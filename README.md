# A Hybrid Machine Learning Framework for Water Quality Index Prediction Using Adaptive Domain-Aware Neural Network Initialization

**Team (Batch 03, KITSW CSE — AI & ML):** Mohammed Nabi Adnan Ahmed (B23AI065),
Sidda Samith (B23AI068), Sai Charith Chekurthi (B23AI087),
Komandla Meghana (B23AI129)
**Supervisor:** Smt. K. Rama Devi

> The project title is frozen and is used exactly as submitted in Phase 1.

## The honest story, in one paragraph

This project studies whether *how you initialise a neural network*
changes how well it learns — specifically, whether guiding the
first-layer weights with feature importance (from XGBoost + SHAP)
and with domain knowledge (from the WQI formula itself) helps.
The target, WQI, is calculated from the same 7 water parameters
that are fed to the network, using the Weighted Arithmetic formula.
That means WQI is a **linear function of the inputs**, so the ANN
is reconstructing a known formula, not discovering a new
relationship and not forecasting future water quality. The exact
formula reproduces the target with ~0 error, and Ridge regression
gets very close (MAE ≈ 0.2 in the technical document). The ANNs
are far behind that — and that is the result, not a failure to
hide. The contribution is the *study of initialisation and
gating*, with that limitation stated openly. Do not describe this
project as forecasting, as missing-parameter prediction, or as a
deployed water-quality system: none of those were demonstrated.

## What the code does

- Loads the team's audited 1,572-row dataset (primary input) and
  verifies it by recomputing WQI from the 7 raw features — it
  fails loudly if the file does not match the formula.
- Also reproduces the 1,572 rows from the raw Kaggle file, and
  cross-checks the two files against each other.
- Computes the WAWQI target, the domain relevance D, the SHAP
  importance S (XGBoost + TreeSHAP, training fold only), and the
  adaptive score A = 0.8·S + 0.2·D (β = 0.2, frozen).
- Trains a custom NumPy ANN (7–16–8–1, ReLU, linear output, MSE,
  Adam lr 0.001, up to 1,800 epochs, early stopping patience 140,
  full-batch) with five initialisations: Xavier, Domain-only,
  SHAP-only, Adaptive, and Gated.
- Compares them with Ridge, XGBoost, Random Forest, ExtraTrees and
  HistGradientBoosting, plus the exact-formula reference, using
  5-fold station-grouped cross-validation (GroupKFold, with 15%
  of the training groups held out for validation in each fold).
- Reports MAE / RMSE / R² / median AE / p90 AE on the original
  WQI scale, pooled out-of-fold and per fold, plus paired
  station-level tests of every ANN method against SHAP-only
  (paired t-test, Wilcoxon, 5,000 bootstrap resamples, 95% CI).

## What is NOT implemented (say so in the viva)

- **No XGBoost late fusion / stacking on top of the ANN.** The
  Phase-1 proposal discussed it; the final primary run does not
  include it. "Hybrid" in the title means XGBoost-informed
  initialisation (SHAP → ANN), not a late-fusion ensemble.
- **No SVR** in the final comparison (it is in the base paper,
  but not in the final expanded experiment).
- **No SHAP explanations of the trained ANN** — only XGBoost SHAP
  (before training) and the learned gate values (after training).
- **No missing-parameter model, no time-ahead forecasting, no
  real-time / IoT deployment.** Station-grouped validation tests
  transfer to held-out stations, not forecasting.

## The gated variant — read this, the diagrams were wrong

The architecture diagrams showed the gates as sigmoid-constrained
to [0, 1], with g_i^(0) = A_i. **That is not what the code does.**

Real implementation (`src/gated.py`):
`g_i^(0) = sqrt(A_i / mean(A))`, `theta_i = log(g_i)`,
`g_i = exp(theta_i)` — so gates are **positive and unbounded**,
can exceed 1, and are trained jointly with the same Adam
(lr 0.001), with no separate learning rate and no sparsity penalty.
There is no sigmoid and no logit anywhere in the gated code.

## Dataset

**Primary input:** `data/processed/water_quality_1572_audited_recomputed.csv`
— the team's audited file, 1,572 rows × 47 columns, 356 station
groups (CODE: 1,479 rows, LOC: 93), years 2005–2014.

**Source of truth:** `data/raw/water_dataX.csv` — Kaggle,
anbarivan/indian-water-quality-data. Download it with:

```bash
# Kaggle CLI (needs a Kaggle account + API token):
kaggle datasets download anbarivan/indian-water-quality-data \
  --file water_dataX.csv -p data/raw --unzip
# or download water_dataX.csv from
# https://www.kaggle.com/datasets/anbarivan/indian-water-quality-data
# and place it at data/raw/water_dataX.csv
```

**Expected counts (the unit tests check these):**

| Stage | Rows |
|---|---|
| Raw CSV | 1,991 |
| After exact-duplicate removal | 1,990 |
| After dropping rows missing any of the 7 features | 1,625 |
| After dropping pH outside [0, 14] | **1,572** |

The 53 pH-dropped rows are all from 2003–2004 and have
impossible pH values (e.g. 239, 1148, 9416) with pH-like
conductivity values — they look as if pH and conductivity were
swapped in the source file. They are **dropped, not repaired**:
swapping them back would be guessing at the data. See
`src/data.py` for the full explanation.

QC flags on the 1,572 rows: fecal coliform > 100: 1,008;
fecal > 1,000,000: 3; total coliform > 1,000,000: 3;
conductivity > 10,000: 103; BOD > 100: 7; DO = 0: 1;
pH outside 6.5–8.5: 116. WQI: min −10.56, median ≈ 70.74,
max 6,145,091.29, 3 negative values (pH < 7 gives a negative
sub-index under the one-sided pH formula — kept as specified).

The 2025 Scientific Reports study on the same dataset reported
1,987 samples; our copy has 1,991 rows. Document that as a
dataset-version / source-copy discrepancy, not as an error.

## Notebook version (whole project in one file)

[`complete ipynb/WQI_Major_Project_Complete.ipynb`](complete%20ipynb/WQI_Major_Project_Complete.ipynb)
contains the complete project — every line of `src/`, in the same order,
with the same values — plus explanations and the full-run results and
plots. Run it top to bottom (about a minute) and it reproduces the full
experiment exactly: its pooled MAE matches the `src/` full run to 0.0
on every method. It also runs on its own (e.g. Colab): the setup cell
downloads the two data files from this repo if they are not present.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Smoke test (1 fold, 1 seed, 200 epochs) — run this first:
python -m src.run_experiment --mode quick

# Full experiment (5 folds x 3 seeds x 5 methods = 75 ANN fits):
python -m src.run_experiment --mode full

# Unit tests:
python -m unittest discover -s tests -v

# Single-sample WQI (exact formula, no trained model):
python -m src.predict_wqi --do 6.7 --ph 7.5 --conductivity 203 \
  --bod 2.0 --nitrate 0.1 --fecal-coliform 11 --total-coliform 27
```

Options: `--mode quick|full`, `--data-source audited|raw`,
`--output-dir <dir>`. Outputs go to `outputs/<mode>/` by default:
`pooled_oof_predictions.csv`, `pooled_metrics.csv`,
`per_fold_mae.csv`, `station_stats.csv`, `importance.csv`,
`learned_gates.csv`, `run_summary.json`.

Versions used while developing this code: Python 3.10+, numpy
2.5.4, pandas 3.1.0, scikit-learn 1.9.1, xgboost 3.4.1,
shap 0.54.1, scipy 1.18.1. `requirements.txt` lists lower
bounds, not pins. Note: `matplotlib` is listed for optional
plotting but no module in `src/` imports it.

## Will my run reproduce the document's exact numbers?

**No — not exactly, and that is expected.** This is a clean
re-implementation of the technical document, not a copy of the
original notebook. The pooled MAE numbers in the document
(Xavier 2388.7, Domain 4750.0, SHAP 1587.9, Adaptive 1698.9,
Gated 1468.9, Ridge 0.2162) are the *reference* results to
compare against. The honest conclusion is preserved — exact
formula (~0) and Ridge (~0.3) beat the ANNs by orders of
magnitude — but the ANN ordering is **not** preserved, and
that is a finding, not a bug to hide.

**Full run of this code (6 Nov 2026, `outputs/full/`, 62 s,
5 folds × 3 seeds × 5 methods = 75 ANN fits):**

| Method | Pooled MAE (this code) | Pooled MAE (document) |
|---|---|---|
| Xavier | 2406.9 | 2388.7 |
| Domain-only | 2297.7 | 4750.0 |
| SHAP-only | 1368.8 | 1587.9 |
| Adaptive | 1829.7 | 1698.9 |
| Gated | 1787.3 | 1468.9 |
| Ridge | 0.29 | 0.22 |
| XGBoost | 8774.8 | 16658.8 |
| Random Forest | 8902.8 | 10830.4 |
| ExtraTrees | 8610.2 | 8622.2 |
| HistGradientBoosting | 20329.5 | 28663.3 |
| Exact formula | ~0 | ~0 |

In this run SHAP-only is the best ANN, not Gated, and Domain
is far better than the document's Domain. Several numbers match
closely (Xavier, ExtraTrees, Ridge), so the implementation is
in the right ballpark — but the ranking between the ANN
variants is fragile. Why: (1) the pooled numbers are dominated
by fold 2, whose test set contains the single WQI = 6,145,091
row — one extreme row moves pooled MAE by thousands;
(2) the SHAP vector S is unstable across folds (e.g. BOD S
ranges 0.002–0.31, fecal coliform 0.14–0.63 across the 5 folds,
because mean |TreeSHAP| on raw WQI is dominated by the 3
extreme rows and by which of them land in the training fold);
(3) the Xavier draws, the validation-group split, and early
stopping all differ in detail from the original notebook.
Station-level tests in this run (see `outputs/full/station_stats.csv`):
SHAP-only beats Xavier (mean station-MAE diff +2089.6, t p 0.031,
Wilcoxon p 0.019, 95% CI [516, 4245]) and beats Domain
(+1349.3, t p 0.049, Wilcoxon p 0.016, CI [371, 2965]) and
Gated (+735.8, t p 0.035, Wilcoxon p 0.0005, CI [182, 1500]);
Adaptive vs SHAP is mixed (+632.8, t p 0.270, Wilcoxon p 0.0006,
CI [-108, 1949] — not significant). The Domain-vs-SHAP
Wilcoxon and CI were missing (truncated) in the source
document; this run fills them in (Wilcoxon p 0.016, CI [371, 2965]).

**What to report:** quote *this run's* numbers as the results
of *this code*, and cite the document's numbers as the reference
notebook's results. Do not claim "Gated is best" from this code —
in this run it is not. The defensible claim is: among the ANN
variants, SHAP-informed initialisation helps vs Xavier, the
differences between SHAP / Adaptive / Gated are small and
fold-dependent, and every ANN is far behind the exact formula
and Ridge on this reconstruction task. See `VIVA_NOTES.md`
for how to defend this in the viva.

## Choices I had to make (document was ambiguous)

1. **Validation split:** the document says "15% of the remaining
   training groups" for validation. This code uses
   `GroupShuffleSplit(test_size=0.15, random_state=42)` on the
   remaining training groups — deterministic, seed fixed, and
   documented in `src/run_experiment.py`.
2. **Baselines use raw inputs, the ANN uses scaled inputs.**
   Ridge is scale-sensitive: on the standardised target (SD
   ~200,000, dominated by 3 extreme WQI values) with alpha=1.0
   it degrades to MAE ~36 on fold 1, instead of ~0.01 on raw
   inputs — and the document's pooled Ridge MAE is 0.2162, so raw
   inputs are what reproduces the reference result. Tree models
   are invariant to monotone scaling, so the choice does not
   affect them. The XGBoost baseline reuses the model fitted for
   SHAP (same training data, same hyperparameters), rather than
   being fitted twice.
3. **Seed averaging:** the 3 seed predictions per method per fold
   are averaged in original WQI units after inverse-transforming.
   (The inverse transform is linear, so averaging before or after
   it gives the same result.)
4. **Bootstrap CI:** percentile method on 5,000 resamples of the
   356 station-level paired differences, seed 42.
5. **Q26 in the source document was truncated** (Domain vs SHAP
   Wilcoxon / CI missing). This code computes the full table, so
   the missing values will be filled in by your run — do not
   quote the truncated values as complete.

## File map (one job per file)

| File | Job |
|---|---|
| `src/config.py` | All constants, with comments |
| `src/wqi.py` | WAWQI formula only |
| `src/data.py` | Load / clean / verify / cross-check the data |
| `src/importance.py` | XGBoost + TreeSHAP → S, D, A |
| `src/ann.py` | NumPy ANN, Xavier, row-scaling init, Adam |
| `src/gated.py` | Gated ANN (exp gates — see above) |
| `src/baselines.py` | Ridge / XGBoost / RF / ExtraTrees / HistGB + exact formula |
| `src/evaluate.py` | Metrics + station-level paired tests |
| `src/run_experiment.py` | CLI experiment runner (`--mode quick|full`) |
| `src/predict_wqi.py` | Single-sample exact-WQI demo |
| `tests/test_wqi_project.py` | Unit tests (data, formula, gates, ANN, CV) |

## Base paper (for reference only — do not copy its numbers)

Al Bataineh, Vamsi & Smith, "A Hybrid Machine Learning Framework
for Water Quality Index Prediction Using Feature-Based Neural
Network Initialization", IEEE JSTARS Vol. 19, 2026,
DOI 10.1109/JSTARS.2026.3654017. That paper is **binary potability
classification** on the Kaggle water-potability set (3,276 rows,
9 features), with a 9–16–8–1 sigmoid ANN and XGBoost late fusion
(86.9% accuracy). This project adapts its SHAP-initialisation
idea to WQI regression on a different dataset, with domain
relevance, adaptive scores, and gating added — none of which are
in the paper. Never quote the paper's accuracy or features as
this project's results.
