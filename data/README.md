# Data

Two files. Do not confuse them.

## Primary input — the team's audited file (used by the experiment)

`processed/water_quality_1572_audited_recomputed.csv`
— 1,572 rows, 47 columns. Copied from the team's audited file.

This is what `src/run_experiment.py` loads by default. It contains
the 7 raw features (`do`, `ph`, `conductivity`, `bod`,
`nitrate_nitrite`, `fecal_coliform`, `total_coliform`), the primary
`WQI` column, `station_group_id`, the 7 sub-index columns
(`*_q`), and flag / QC columns.

On load, `src/data.py` verifies it: it recomputes WQI from the 7
raw features with the WAWQI formula and fails loudly unless the
maximum difference from the file's `WQI` column is < 1e-3
(in practice ~1e-9), there are 1,572 rows, and there are 356
station groups.

Columns that are **sensitivity variants only** — never the target:
- `WQI_two_sided_pH_sensitivity` (pH treated two-sidedly)
- `WQI_asinh_scale100`, `WQI_original_asinh_scale100`,
  `WQI_two_sided_pH_asinh_scale100` (asinh-transformed WQI)
- `log1p_*` (log-transformed features)

Temperature has 46 nulls in this file; it is irrelevant, because
temperature is not one of the 7 model features.

## Source of truth — the raw Kaggle file

`raw/water_dataX.csv` — 1,991 rows, 12 columns.
Source: Kaggle, anbarivan/indian-water-quality-data,
https://www.kaggle.com/datasets/anbarivan/indian-water-quality-data

Encoding note: the Kaggle download is latin-1; the copy in this
repo was re-encoded to UTF-8 (same characters, including the
micro sign in the conductivity header) so it can be stored on
GitHub. `src/data.py` (`load_raw()`) accepts both encodings, so
a fresh latin-1 download works too.

`src/data.py` (`clean()`) reproduces the 1,572 rows from this file:
drop exact duplicates (1991 → 1990), drop rows missing any of the
7 features (→ 1625), drop physically impossible negatives (none),
drop pH outside [0, 14] (→ 1572). The 53 pH-dropped rows are all
from 2003–2004 and look as if pH and conductivity were swapped in
the source file; they are dropped, not repaired — see `src/data.py`
and the main README.

`src/data.py` (`cross_check_raw_vs_audited()`) checks the two
files agree on row count, group count, QC flags, and WQI
min/max/median, plus the grouping partition and a sorted
row-multiset comparison on year + 7 features + WQI. There is no
`sample_id` in the raw file, so no sample_id join is possible
(the audited file's `sample_id` is `INDWQ_00001`, …).

One label difference to know about: the audited file's LOC
fallback group IDs append `|STATE:NAN` and collapse double
spaces in 3 location strings, e.g. raw `LOC:KAKKADAVU,  KERALA`
vs audited `LOC:KAKKADAVU, KERALA|STATE:NAN`. The grouping
*partition* is identical (same 356 groups, same per-group row
counts) — only the label text differs. `cross_check_raw_vs_audited()`
normalises this before comparing.

## What is NOT in this folder

- No synthetic data. The real CSVs above are the only data.
- No `outputs/` results — those are generated, in `outputs/`.
