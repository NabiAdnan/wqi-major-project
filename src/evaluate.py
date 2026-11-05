"""Metrics and station-level statistical tests.

All metrics are computed on the ORIGINAL WQI scale. The ANN trains
on a standardised target, so the caller must inverse-transform its
predictions before calling these functions — run_experiment.py does
that, and the unit tests check a known case.

Metrics:
    MAE, RMSE, R2, median absolute error, 90th-percentile absolute
    error. The last two are reported because the WQI distribution
    is extremely right-skewed (max > 6,000,000): a single huge
    error can dominate RMSE, and R2 can stay near 1 even when MAE
    is in the thousands. Never quote R2 on its own for this data.

Pooled out-of-fold (OOF) metrics:
    Each of the 1,572 rows is predicted exactly once, by a model
    whose test fold excluded that row's station group. Pooling
    those predictions gives the headline MAE / RMSE / R2.

Station-level paired tests (vs SHAP-only, the reference method):
    For each station group g, compute MAE_g = mean |y - y_hat|
    over that group's rows, then the paired difference
    d_g = MAE_g(method) - MAE_g(SHAP). Three views of the same
    differences, because they are highly skewed:
      * paired t-test (tests the mean difference),
      * Wilcoxon signed-rank test (non-parametric, uses signed ranks),
      * bootstrap 95% CI for the mean difference (5,000 resamples
        of the 356 station-level differences, percentile method).
    A method has not "proven" superiority when the CI includes 0
    or the p-values disagree — report all three, as the technical
    document does, and do not cherry-pick the one that looks best.
"""

import numpy as np
from scipy import stats


def regression_metrics(y_true, y_pred):
    """MAE, RMSE, R2, median AE and p90 AE on the original scale."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "median_ae": float(np.median(abs_errors)),
        "p90_ae": float(np.percentile(abs_errors, 90)),
        "n": int(len(y_true)),
    }


def station_mae(y_true, y_pred, groups):
    """Per-station-group MAE. Returns a DataFrame-free dict.

    ``groups`` is an array of station_group_id, one per row.
    Returns {group: MAE over that group's rows}.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    groups = np.asarray(groups)
    result = {}
    for group in np.unique(groups):
        mask = groups == group
        result[group] = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
    return result


def paired_station_test(y_true, y_pred_method, y_pred_reference,
                        groups, n_bootstrap=5000, seed=42):
    """Paired station-level comparison of one method vs a reference.

    Returns mean paired difference, paired t-test p-value,
    Wilcoxon signed-rank p-value, and the bootstrap 95% CI of the
    mean difference. Positive mean difference = method is worse
    (higher MAE) than the reference on average across stations.
    """
    method_mae = station_mae(y_true, y_pred_method, groups)
    ref_mae = station_mae(y_true, y_pred_reference, groups)
    common = sorted(set(method_mae) & set(ref_mae))
    if len(common) < 10:
        raise ValueError("Too few common station groups for a paired test")
    differences = np.array(
        [method_mae[g] - ref_mae[g] for g in common], dtype=float
    )
    t_result = stats.ttest_rel(
        [method_mae[g] for g in common], [ref_mae[g] for g in common]
    )
    # Wilcoxon needs non-zero differences; if all are zero, the
    # methods are identical on every station and p = 1.
    non_zero = differences[differences != 0]
    if len(non_zero) == 0:
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = float(stats.wilcoxon(non_zero).pvalue)
    rng = np.random.default_rng(seed)
    bootstrap_means = np.array([
        differences[rng.integers(0, len(differences), size=len(differences))].mean()
        for _ in range(n_bootstrap)
    ])
    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
    return {
        "n_groups": int(len(common)),
        "mean_diff": float(differences.mean()),
        "t_pvalue": float(t_result.pvalue),
        "wilcoxon_pvalue": wilcoxon_p,
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "bootstrap_resamples": int(n_bootstrap),
    }
