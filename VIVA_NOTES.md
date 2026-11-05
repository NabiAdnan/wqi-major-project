# Viva Notes — WQI Major Project

Team: Mohammed Nabi Adnan Ahmed (B23AI065), Sidda Samith (B23AI068),
Sai Charith Chekurthi (B23AI087), Komandla Meghana (B23AI129).
Supervisor: Smt. K. Rama Devi. Title is frozen — do not propose changing it.

These notes are for the viva. They describe **this code**
(`~/workspace/projects/wqi-major-project/`) and **its full run**
(`outputs/full/`, 6 Nov 2026), and they say where that run agrees
with, and where it differs from, the technical document's reference
numbers. Quote this run's numbers as this code's results.

## The one-paragraph answer

We study whether *how you initialise a neural network* changes how
well it learns to reconstruct the Water Quality Index from 7 water
parameters. WQI is computed from those same 7 parameters with the
Weighted Arithmetic formula, so it is a **linear function of the
inputs** — the ANN is reconstructing a known formula, not discovering
a new relationship and not forecasting. The exact formula gets ~0
error and Ridge gets MAE 0.29. The ANNs are in the thousands. That
is the result, not a failure: the contribution is the study of
initialisation (SHAP / domain / adaptive / gating), with the
limitation stated openly.

## What is implemented, and what is not

Implemented in this code:
- WAWQI target, D, S (XGBoost + TreeSHAP, training fold only),
  A = 0.8·S + 0.2·D (β = 0.2, frozen).
- Custom NumPy ANN 7–16–8–1, ReLU, linear output, MSE, Adam
  lr 0.001, up to 1,800 epochs, patience 140, full-batch,
  seeds 17/41/73, 5 methods × 5 folds × 3 seeds = 75 fits.
- Gated variant with exp gates (see Q18 — the diagrams were wrong).
- 5-fold station-grouped CV (GroupKFold, zero group overlap) +
  15% of training groups held out for validation per fold.
- Baselines: Ridge, XGBoost, Random Forest, ExtraTrees,
  HistGradientBoosting + exact-formula reference.
- Pooled OOF metrics + per-fold metrics + station-level paired
  tests vs SHAP-only (t-test, Wilcoxon, 5,000 bootstrap, 95% CI).

NOT implemented (say so if asked):
- No XGBoost late fusion / stacking. "Hybrid" = XGBoost-informed
  initialisation (SHAP → ANN), not a late-fusion ensemble.
- No SVR in the final comparison (it is in the base paper only).
- No SHAP explanations of the trained ANN — only XGBoost SHAP
  (before training) and learned gate values (after training).
- No missing-parameter model, no forecasting, no deployment.
  Station-grouped CV tests transfer to held-out stations, not
  to future years.

## Q3 — Why is the target a linear function, and why use an ANN at all?

q_i = 100·(V_i − ideal_i)/(Std_i − ideal_i) is linear in V_i, and
WQI = Σ D_i·q_i with fixed D_i, so WQI is linear in the 7 inputs.
The ANN is therefore learning to approximate a known mapping.
That is deliberate: the experiment is a *controlled comparison of
initialisation methods* on a problem where the answer is known,
so differences between methods are attributable to the
initialisation, not to an unknown ground truth. The exact formula
and Ridge are the honest baselines that keep the comparison honest.

## Q4 — The honest limitation

Do not claim forecasting, missing-parameter prediction, or a
deployed system. Do not quote R² alone: pooled R² is 0.99+ for
the ANNs even with MAE in the thousands, because 3 extreme
observations (WQI up to 6,145,091) dominate the variance. Report
MAE (and median AE / p90 AE) alongside R². Per-fold R² is
meaningless here (e.g. −158 on fold 1) because a test fold with
no extreme rows has tiny variance; pooled R² over all 1,572 rows
is the meaningful one.

## Q7 — What is novel?

Primary: Adaptive Domain-Aware initialisation, A_i = 0.8·S_i + 0.2·D_i.
Secondary: trainable adaptive feature gating. Station-grouped
validation is experimental design, not novelty. Do not claim
first-of-its-kind without a systematic literature review. The
base paper (Al Bataineh et al., IEEE JSTARS 2026) is binary
potability classification on a different dataset (3,276 rows,
9 features), with SHAP-init + XGBoost late fusion — it has no D_i,
no A_i, no gates, no WQI regression, no station-grouped CV.
Those are this project's adaptations.

## Q17 — How does the non-gated initialisation work?

W1 is 7×16, one row per feature. Start from Xavier (Glorot uniform,
limit √(6/(fan_in+fan_out)), biases zero), then for Domain / SHAP /
Adaptive scale row i by r_i = √(I_i / mean(I)) and restore the
Frobenius norm of W1 to the Xavier value. Scaling a row changes the
*distribution* of importance across inputs; restoring the norm
keeps the overall weight scale unchanged. Adaptive modifies the
first-layer weights — it does NOT multiply the raw features by A_i.
That is what the gated variant does instead.

## Q18 — The gates (the diagrams were wrong)

The diagrams showed sigmoid gates in [0, 1] with g_i^(0) = A_i.
That is wrong. The real implementation (`src/gated.py`):
g_i^(0) = √(A_i / mean(A)), θ_i = log(g_i), g_i = exp(θ_i).
So gates are **positive and unbounded** — they can exceed 1 and
they can go below 1. No sigmoid, no logit, no [0, 1] constraint.
In this run the learned gates average: DO 0.90, pH 0.63,
conductivity 0.39, BOD 1.26, nitrate 0.54, fecal coliform 1.54,
total coliform 0.96 — the network amplifies fecal coliform and
BOD and suppresses conductivity and nitrate. That pattern agrees
with the SHAP importance (fecal coliform is the largest mean S,
0.415), which is a nice consistency to point to — but do not force
the two signals to match exactly: S is computed before training
from XGBoost, g is learned inside the ANN during training.

## Q19 (related) — Gate training

No separate learning rate, no L1/sparsity penalty, no weight decay,
no separate optimiser. Gates and network are trained jointly with
the same Adam (lr 0.001). The gate gradient is
dL/d(log g_i) = g_i · Σ_n dL/d(x̃) · x, where x is the standardised
input *before* gating — `ann.forward` stores the pre-gating input
as `cache["X"]` and the gated input as `cache["X_gated"]` precisely
so both gradients are correct. This is covered by a unit test that
checks the analytic gradient against finite differences
(`test_log_gate_gradient_matches_finite_differences`).

## Q20–Q21 — Training setup

7→16→8→1, ReLU hidden, linear output, Adam lr 0.001, β1 0.9,
β2 0.999, ε 1e-8, MSE, max 1,800 epochs, patience 140 on
validation MSE, full-batch, no dropout / batch norm / weight decay.
Seeds 17, 41, 73; predictions averaged over the 3 seeds per fold.
Architecture 16/8 follows the base paper's 9→16→8→1 — it was not
chosen by an architecture search, so do not claim it is optimal.
Note: 7 of the 15 gated fits in this run hit the 1,800-epoch cap
(best_epoch = 1800), i.e. validation loss was still improving —
a longer cap might change the gated numbers slightly.

## Q22 — Why is S unstable across folds? (expect this question)

Mean |TreeSHAP| on raw WQI is dominated by the 3 extreme rows
(WQI up to 6.1M), so which extreme rows land in a training fold
moves S a lot: across this run's 5 folds, BOD S ranges 0.002–0.31
and fecal coliform S ranges 0.14–0.63. The fold-mean S in this run
(DO 0.113, pH 0.020, conductivity 0.045, BOD 0.155, nitrate 0.057,
fecal 0.415, total 0.195) differs from the document's fold-mean S
(DO 0.187, pH 0.019, conductivity 0.031, BOD 0.207, nitrate 0.060,
fecal 0.278, total 0.219), most notably for fecal coliform. The
method was followed exactly as specified; the instability is a
property of the data (a handful of extreme rows), not a coding
choice. If asked what would stabilise it: computing SHAP on a
transformed target (e.g. asinh) or reporting median |SHAP| instead
of the mean — but that would be a *different* method, and it was
not done here. Do not quietly change the method to chase the
document's numbers.

## Q25 — Results of THIS run (outputs/full/, 6 Nov 2026)

Pooled OOF MAE over all 1,572 rows, original WQI scale:

| Method | MAE | RMSE | R² | Median AE |
|---|---|---|---|---|
| Xavier | 2406.9 | 17407.4 | 0.9927 | 305.4 |
| Domain-only | 2297.7 | 16467.5 | 0.9935 | 306.3 |
| SHAP-only | 1368.8 | 6181.3 | 0.9991 | 278.8 |
| Adaptive | 1829.7 | 14618.2 | 0.9948 | 311.9 |
| Gated | 1787.3 | 10189.3 | 0.9975 | 339.5 |
| Ridge | 0.29 | 6.44 | ~1.0000 | 0.016 |
| XGBoost | 8774.8 | 203734.4 | −0.0015 | 3.04 |
| Random Forest | 8902.8 | 203800.9 | −0.0022 | 2.39 |
| ExtraTrees | 8610.2 | 203682.3 | −0.0010 | 1.49 |
| HistGradientBoosting | 20329.5 | 213622.1 | −0.1011 | 22.16 |
| Exact formula | ~0 | ~0 | 1.0000 | ~0 |

Why the tree models look bad pooled but great on some folds:
fold 2's test set contains the WQI = 6,145,091 row. Trees cannot
extrapolate beyond the training range, so they miss it by ~6M,
which alone adds ~19,000 to a fold's MAE; the ANNs extrapolate
(linearly) and do better on that fold. On fold 1 (test max WQI
1,700, no extremes) the trees are far better than the ANNs
(ExtraTrees MAE 5.3 vs ANN ~1,000–2,000), because the ANN trains
on a standardised target whose scale is dominated by the extreme
rows, so small-WQI errors barely register in its loss. Both
patterns are visible in `outputs/full/per_fold_mae.csv` — quote
per-fold numbers when asked, not just the pooled ones.

## Q26 — Station-level statistics (this run, 356 stations)

Paired station-MAE differences vs SHAP-only (positive = worse
than SHAP):

| Method | Mean diff | t p | Wilcoxon p | Bootstrap 95% CI |
|---|---|---|---|---|
| Xavier | +2089.6 | 0.031 | 0.019 | [516, 4245] |
| Domain | +1349.3 | 0.049 | 0.016 | [371, 2965] |
| Adaptive | +632.8 | 0.270 | 0.0006 | [−108, 1949] |
| Gated | +735.8 | 0.035 | 0.0005 | [182, 1500] |

In this run SHAP-only is significantly better than Xavier, Domain
and Gated on station-mean MAE; Adaptive vs SHAP is mixed
(t-test and CI not significant, Wilcoxon significant) — do not
claim Adaptive beats SHAP. The Domain-vs-SHAP Wilcoxon and CI
were missing (truncated) in the source document; this run fills
them in: Wilcoxon p 0.016, CI [371, 2965].

## The discrepancy with the document — how to answer it

The document's reference numbers say Gated is the best ANN
(pooled MAE 1468.9) and Domain the worst (4750.0). This clean
re-implementation says SHAP-only is best (1368.8) and Domain is
second-best among the ANNs (2297.7). Several numbers match
closely (Xavier 2407 vs 2389, ExtraTrees 8610 vs 8622, Ridge
0.29 vs 0.22), so the implementation is in the right ballpark;
the ranking between ANN variants is fragile, for three stated
reasons: (1) pooled MAE is dominated by the single 6.1M row in
fold 2's test set; (2) S is unstable across folds (see Q22);
(3) Xavier draws, the validation-group split and early stopping
differ in detail from the original notebook. The defensible
claim from this run is: SHAP-informed initialisation helps vs
Xavier; the differences between SHAP / Adaptive / Gated are
small and fold-dependent; and every ANN is far behind the exact
formula and Ridge on this reconstruction task. Present the
document's numbers as the reference notebook's results and this
run's numbers as this code's results — do not merge them into
one table without saying which is which.

## Data facts to have ready

- Kaggle Indian Water Quality Data, `water_dataX.csv`: 1,991 rows;
  after exact-duplicate removal 1,990; after dropping rows missing
  any of the 7 features 1,625; after dropping pH outside [0, 14]
  **1,572**. The 53 dropped rows are all 2003–2004 with impossible
  pH (e.g. 239, 1148) and pH-like conductivity — they look swapped
  in the source, but they are dropped, not repaired.
- 356 station groups (CODE: 1,479 rows, LOC: 93), years 2005–2014.
- QC flags: fecal > 100: 1,008; fecal > 1M: 3; total coliform
  > 1M: 3; conductivity > 10,000: 103; BOD > 100: 7; DO = 0: 1;
  pH outside 6.5–8.5: 116. WQI: min −10.56, median ≈ 70.74,
  max 6,145,091.29, 3 negative values (pH < 7, one-sided formula).
- D values: DO 0.2213, pH 0.2604, conductivity 0.0022, BOD 0.4426,
  nitrate 0.0492, fecal coliform 0.0221, total coliform 0.0022.
- The audited file (`data/processed/water_quality_1572_audited_recomputed.csv`)
  is the primary input; the code recomputes WQI from its 7 raw
  features and fails loudly if it does not match (max diff in
  this run: ~1.4e-9). Primary `WQI` column only — never the
  two-sided-pH or asinh sensitivity columns.

## How to run it (for a demo)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v        # 22 tests
python -m src.run_experiment --mode quick       # ~8 s smoke test
python -m src.run_experiment --mode full        # ~62 s, 75 ANN fits
```
