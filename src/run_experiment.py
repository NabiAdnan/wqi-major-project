"""Run the full experiment (or a quick smoke test).

Usage, from the project root:
    python -m src.run_experiment --mode quick
    python -m src.run_experiment --mode full
    python -m src.run_experiment --mode full --data-source raw

Modes:
    quick — 1 outer fold, 1 seed (17), max 200 epochs. A smoke test
            only: it checks the whole pipeline runs end to end, it
            does NOT produce publishable results.
    full  — 5 outer folds x 3 seeds (17, 41, 73) x 5 ANN methods =
            75 ANN fits, plus 1 XGBoost (for SHAP) and the 5 classical
            baselines per fold. This is the experiment in the
            technical document. It takes a while; do not run it
            until the quick mode passes.

Data source (default: audited):
    audited — data/processed/water_quality_1572_audited_recomputed.csv,
              the team's audited file, verified on load (see data.py).
    raw     — data/raw/water_dataX.csv, cleaned by the pipeline in
              data.py. The cross-check against the audited file is
              still run and saved, whichever source is primary.

For every outer fold the order of operations is fixed, and it
matters — it is what prevents leakage:
    1. Split off the test fold by station group (GroupKFold).
    2. Split 15% of the remaining groups off as validation
       (GroupShuffleSplit, seed 42).
    3. Fit the X and y StandardScalers on the TRAIN partition only.
    4. Fit XGBoost on the TRAIN partition only (raw X) and compute
       TreeSHAP -> S, D, A for this fold.
    5. Train each ANN method on train, early-stop on validation,
       average the seed predictions, evaluate on the test fold.
    6. Fit the classical baselines on train (RAW inputs, not
       scaled — Ridge needs raw inputs, see baselines.py),
       evaluate on test.
    7. Compute the exact-formula predictions from the raw test inputs.

Outputs (in --output-dir, default outputs/<mode>/):
    pooled_oof_predictions.csv — one row per sample, y_true plus
        one prediction column per method / baseline / exact formula.
    pooled_metrics.csv         — MAE / RMSE / R2 / median AE / p90 AE
        for every method, pooled over all test folds.
    per_fold_mae.csv           — MAE per method per fold.
    station_stats.csv          — paired station-level tests of every
        ANN method vs SHAP-only (t-test, Wilcoxon, bootstrap CI).
    importance.csv             — per-fold S / D / A vectors.
    learned_gates.csv          — final gates per fold and seed.
    run_summary.json           — data audit, cross-check, config,
        timing, and the pooled metrics in one place.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from . import ann, gated
from .baselines import exact_formula_predictions, fit_baselines, predict_baselines
from .config import (
    BOOTSTRAP_RESAMPLES,
    FEATURES,
    MAX_EPOCHS,
    N_FOLDS,
    OUTPUT_DIR,
    SEEDS,
    VALIDATION_GROUP_FRACTION,
    VALIDATION_SPLIT_SEED,
)
from .data import (
    audit,
    cross_check_raw_vs_audited,
    load_audited,
    load_cleaned,
)
from .evaluate import paired_station_test, regression_metrics
from .importance import compute_importance

ANN_METHODS = ["xavier", "domain", "shap", "adaptive", "gated"]
# Which importance vector initialises each non-gated method.
METHOD_IMPORTANCE = {"domain": "D", "shap": "S", "adaptive": "A"}


def _split_train_val(X, y, groups, seed=VALIDATION_SPLIT_SEED):
    """Split the training groups into train / validation (15% of groups)."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_GROUP_FRACTION,
        random_state=seed,
    )
    train_idx, val_idx = next(splitter.split(X, y, groups))
    return train_idx, val_idx


def _run_one_fold(df, fold, train_idx, test_idx, seeds, max_epochs):
    """Run one outer fold. Returns (test predictions dict, records)."""
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["wqi"].to_numpy(dtype=float)
    groups = df["station_group_id"].to_numpy()

    X_train_full, y_train_full = X[train_idx], y[train_idx]
    groups_train_full = groups[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Validation: 15% of the remaining training groups.
    tr_idx, va_idx = _split_train_val(
        X_train_full, y_train_full, groups_train_full
    )
    X_train, y_train = X_train_full[tr_idx], y_train_full[tr_idx]
    X_val, y_val = X_train_full[va_idx], y_train_full[va_idx]

    # Scalers fitted on TRAIN only, for X and for y.
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    X_train_s = x_scaler.transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)
    y_train_s = y_scaler.transform(y_train.reshape(-1, 1))
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1))

    def inverse_y(y_scaled):
        return y_scaler.inverse_transform(
            np.asarray(y_scaled).reshape(-1, 1)
        ).ravel()

    # --- Importance: XGBoost + TreeSHAP on TRAIN only (raw X). ---
    importance = compute_importance(X_train, y_train)
    importance_record = {
        "fold": fold,
        **{f"S_{f}": importance["S"][i] for i, f in enumerate(FEATURES)},
        **{f"D_{f}": importance["D"][i] for i, f in enumerate(FEATURES)},
        **{f"A_{f}": importance["A"][i] for i, f in enumerate(FEATURES)},
    }

    # --- ANN methods, seed-averaged per fold. ---
    predictions = {}
    gate_records = []
    for method in ANN_METHODS:
        seed_predictions = []
        for seed in seeds:
            if method == "gated":
                params, history = gated.train_gated(
                    X_train_s, y_train_s, X_val_s, y_val_s,
                    adaptive=importance["A"], seed=seed,
                    max_epochs=max_epochs,
                )
                y_pred_s = gated.predict_gated(params, X_test_s)
                gate_records.append({
                    "fold": fold,
                    "seed": seed,
                    "best_epoch": history["best_epoch"],
                    "n_epochs": history["n_epochs"],
                    **{
                        f"g_{f}": gated.final_gates(params)[i]
                        for i, f in enumerate(FEATURES)
                    },
                })
            else:
                importance_vec = None
                if method in METHOD_IMPORTANCE:
                    importance_vec = importance[METHOD_IMPORTANCE[method]]
                params, history = ann.train_ann(
                    X_train_s, y_train_s, X_val_s, y_val_s,
                    importance=importance_vec, method=method,
                    seed=seed, max_epochs=max_epochs,
                )
                y_pred_s = ann.predict(params, X_test_s)
            seed_predictions.append(inverse_y(y_pred_s))
        # Average the seed predictions (in original WQI units; the
        # inverse transform is linear, so this equals averaging
        # before the transform).
        predictions[method] = np.mean(seed_predictions, axis=0)

    # --- Classical baselines (RAW X and y, same train split). ---
    # Baselines are fitted on raw inputs on purpose: Ridge is
    # scale-sensitive, and on standardised y it degrades to
    # MAE ~36 on fold 1 instead of ~0.01 (see baselines.py).
    # The XGBoost baseline reuses the model already fitted for
    # SHAP above — it was fitted on exactly this training data.
    baseline_models = fit_baselines(
        X_train, y_train, xgboost_model=importance["model"]
    )
    baseline_preds = predict_baselines(baseline_models, X_test)
    for name, pred in baseline_preds.items():
        predictions[name] = pred

    # --- Exact formula reference (raw test inputs, no model). ---
    predictions["exact_formula"] = exact_formula_predictions(X_test)

    return {
        "test_idx": test_idx,
        "y_test": y_test,
        "predictions": predictions,
        "importance_record": importance_record,
        "gate_records": gate_records,
    }


def run(mode="quick", data_source="audited", output_dir=None):
    """Run the experiment and write all outputs. Returns the summary dict."""
    started = time.time()
    if mode == "quick":
        n_folds, seeds, max_epochs = 1, [SEEDS[0]], 200
    elif mode == "full":
        n_folds, seeds, max_epochs = N_FOLDS, list(SEEDS), MAX_EPOCHS
    else:
        raise ValueError(f"Unknown mode {mode!r}; use 'quick' or 'full'")

    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    # Primary input (audited by default), plus the raw-file cross-check.
    if data_source == "audited":
        df = load_audited()
    elif data_source == "raw":
        df = load_cleaned()
    else:
        raise ValueError(
            f"Unknown data source {data_source!r}; use 'audited' or 'raw'"
        )
    data_audit = audit(df)
    cross_check = cross_check_raw_vs_audited(load_cleaned(), load_audited())

    X = df[FEATURES].to_numpy(dtype=float)
    y = df["wqi"].to_numpy(dtype=float)
    groups = df["station_group_id"].to_numpy()

    oof_predictions = {name: np.full(len(df), np.nan) for name in
                       ANN_METHODS + ["ridge", "xgboost", "random_forest",
                                      "extra_trees", "hist_gb",
                                      "exact_formula"]}
    fold_of_row = np.full(len(df), -1, dtype=int)
    per_fold_rows = []
    importance_rows = []
    gate_rows = []

    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups))
    for fold, (train_idx, test_idx) in enumerate(folds[:n_folds], start=1):
        print(f"[fold {fold}/{n_folds}] train={len(train_idx)} "
              f"test={len(test_idx)} ...", flush=True)
        result = _run_one_fold(df, fold, train_idx, test_idx,
                               seeds, max_epochs)
        fold_of_row[test_idx] = fold
        for name, pred in result["predictions"].items():
            oof_predictions[name][test_idx] = pred
        importance_rows.append(result["importance_record"])
        gate_rows.extend(result["gate_records"])
        for name, pred in result["predictions"].items():
            per_fold_rows.append({
                "fold": fold,
                "method": name,
                **regression_metrics(result["y_test"], pred),
            })
        print(f"[fold {fold}/{n_folds}] done.", flush=True)

    # --- Pooled OOF metrics (only rows that were actually predicted). ---
    mask = fold_of_row > 0
    pooled_rows = []
    for name, pred in oof_predictions.items():
        pooled_rows.append({
            "method": name,
            **regression_metrics(y[mask], pred[mask]),
        })
    pooled = pd.DataFrame(pooled_rows)

    # --- Station-level paired tests vs SHAP-only. ---
    station_rows = []
    for method in ANN_METHODS:
        if method == "shap":
            continue
        test = paired_station_test(
            y[mask], oof_predictions[method][mask],
            oof_predictions["shap"][mask], groups[mask],
            n_bootstrap=BOOTSTRAP_RESAMPLES,
        )
        station_rows.append({"method": method, "reference": "shap", **test})

    # --- Save everything. ---
    oof_df = df[["sample_id", "station_group_id", "year"]].copy() \
        if "sample_id" in df.columns else df[["station_group_id", "year"]].copy()
    oof_df["fold"] = fold_of_row
    oof_df["y_true"] = y
    for name, pred in oof_predictions.items():
        oof_df[f"pred_{name}"] = pred
    oof_df.to_csv(output_dir / "pooled_oof_predictions.csv", index=False)
    pooled.to_csv(output_dir / "pooled_metrics.csv", index=False)
    pd.DataFrame(per_fold_rows).to_csv(
        output_dir / "per_fold_mae.csv", index=False
    )
    pd.DataFrame(station_rows).to_csv(
        output_dir / "station_stats.csv", index=False
    )
    pd.DataFrame(importance_rows).to_csv(
        output_dir / "importance.csv", index=False
    )
    pd.DataFrame(gate_rows).to_csv(
        output_dir / "learned_gates.csv", index=False
    )

    summary = {
        "mode": mode,
        "data_source": data_source,
        "n_folds": n_folds,
        "seeds": seeds,
        "max_epochs": max_epochs,
        "methods": ANN_METHODS,
        "data_audit": data_audit,
        "cross_check_raw_vs_audited": cross_check,
        "pooled_metrics": pooled.to_dict(orient="records"),
        "station_stats": station_rows,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nPooled out-of-fold metrics (original WQI scale):")
    print(pooled.to_string(index=False))
    print(f"\nOutputs written to {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run the WQI major-project experiment."
    )
    parser.add_argument(
        "--mode", choices=["quick", "full"], default="quick",
        help="quick = 1 fold, 1 seed, 200 epochs (smoke test); "
             "full = 5 folds x 3 seeds x 5 methods (75 ANN fits).",
    )
    parser.add_argument(
        "--data-source", choices=["audited", "raw"], default="audited",
        help="Primary input file. Default: audited (verified on load).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory. Default: outputs/<mode>/",
    )
    args = parser.parse_args()
    run(mode=args.mode, data_source=args.data_source,
        output_dir=args.output_dir)


if __name__ == "__main__":
    main()
