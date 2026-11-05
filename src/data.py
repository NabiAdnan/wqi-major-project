"""Load and clean the Indian Water Quality dataset.

Two input files, two jobs — do not confuse them:

PRIMARY INPUT (used by run_experiment.py):
    data/processed/water_quality_1572_audited_recomputed.csv
    The team's audited, prepared file: 1,572 rows, 47 columns.
    ``load_audited()`` reads it, renames ``nitrate_nitrite`` to the
    internal name ``nitrate``, takes the 7 raw features plus
    ``station_group_id``, and uses the ``WQI`` column as the target.
    It then VERIFIES the file instead of trusting it: it recomputes
    WQI from the 7 raw features with the WAWQI formula and fails
    loudly unless the maximum difference is < 1e-3, there are 1,572
    rows, and there are 356 station groups.

    Columns you must NOT use as the target:
      * ``WQI_two_sided_pH_sensitivity`` — a sensitivity variant in
        which pH is treated two-sidedly. Not the primary formula.
      * ``WQI_asinh_scale100``, ``WQI_original_asinh_scale100``,
        ``WQI_two_sided_pH_asinh_scale100`` — asinh-transformed
        versions, also sensitivity variants only.
      * ``log1p_*`` — log-transformed features, also sensitivity
        variants only, not used in the primary experiment.
    Temperature has 46 nulls in this file; it is irrelevant because
    temperature is not one of the 7 model features.

SOURCE OF TRUTH (kept, and cross-checked):
    data/raw/water_dataX.csv
    The original Kaggle file (anbarivan/indian-water-quality-data,
    https://www.kaggle.com/datasets/anbarivan/indian-water-quality-data).
    The copy in this repo was re-encoded from the Kaggle download's
    latin-1 to UTF-8 (same characters, including the micro sign in
    the conductivity header) so it can be stored on GitHub, which
    requires UTF-8 text. ``load_raw()`` accepts both encodings, so
    a fresh latin-1 download from Kaggle still loads. ``clean()``
    reproduces the 1,572 rows from it, and
    ``cross_check_raw_vs_audited()`` checks the two files agree on
    counts, groups, QC flags, and WQI summary statistics.
    Exact row-by-row joining is NOT done by sample_id: the audited
    file's sample_id (INDWQ_00001, ...) does not exist in the raw
    file. The cross-check therefore compares aggregates, and also
    joins on station + location + year + the 7 features where a
    row-level check is wanted (see the tests).

Cleaning pipeline for the raw file (in order):
    1. Read the CSV (UTF-8 in this repo; a fresh Kaggle download is
       latin-1 — ``load_raw()`` accepts both, because the
       conductivity column header contains a micro sign).
    2. Drop exact duplicate rows: 1991 -> 1990.
    3. Convert the 7 feature columns to numeric (anything that does
       not parse becomes NaN).
    4. Drop rows with a missing / non-finite value in any of the
       7 features: 1990 -> 1625.
    5. Drop rows with a negative value in any feature (a negative
       concentration is physically impossible). In this file there
       are none, so the count stays 1625 — the step is kept because
       it is part of the documented procedure, not because it bites.
    6. Drop rows with pH outside [0, 14]: 1625 -> 1572.

Step 6 needs a clear explanation, because it drops 53 rows:
    pH is physically defined on 0-14. The 53 dropped rows all have
    pH values like 239, 86, 101, 1148, 9416 — while their conductivity
    is a plausible pH-like number (6.2, 7.8, ...). All 53 are from
    2003-2004, i.e. before the 2005-2014 window of the kept data.
    It looks as if pH and conductivity were swapped in the source
    file for those rows, but we do NOT swap them back: that would be
    guessing at the data. We drop them as a data-quality filter and
    document it here and in the README.

What we do NOT do: we do not remove extreme values just because
they are large. A fecal coliform of >1,000,000 may be genuine
contamination, and removing polluted samples without justification
would bias the dataset. Extreme values are flagged (see
``qc_flags``), not deleted.

Station groups:
    station_group_id = "CODE:<station code>" if the station code is
    present, else "LOC:<location>". In the raw file the code column
    stores the string 'NAN' for a missing code. Result: 356 groups,
    1479 CODE: rows, 93 LOC: rows. "356 groups" means 356 distinct
    grouping IDs, not 356 independently verified physical sampling
    stations.
"""

import numpy as np
import pandas as pd

from .config import (
    AUDITED_DATA_PATH,
    EXPECTED_CLEAN_ROWS,
    EXPECTED_N_GROUPS,
    EXPECTED_QC_FLAGS,
    FEATURES,
    RAW_DATA_PATH,
)
from .wqi import exact_wqi

# Raw column names, matched by prefix so the micro sign in the
# conductivity header cannot break the load on a different encoding.
_RAW_COLUMN_PREFIXES = {
    "do": "D.O.",
    "ph": "PH",
    "conductivity": "CONDUCTIVITY",
    "bod": "B.O.D.",
    "nitrate": "NITRATENAN",
    "fecal_coliform": "FECAL COLIFORM",
    "total_coliform": "TOTAL COLIFORM",
}

# Column name in the audited file -> internal feature name.
# Only nitrate differs (audited: nitrate_nitrite).
_AUDITED_FEATURE_COLUMNS = {
    "do": "do",
    "ph": "ph",
    "conductivity": "conductivity",
    "bod": "bod",
    "nitrate": "nitrate_nitrite",
    "fecal_coliform": "fecal_coliform",
    "total_coliform": "total_coliform",
}

# How close the recomputed WQI must be to the audited file's WQI
# column. In practice the difference is ~1e-9 (floating-point
# rounding); 1e-3 is a generous tolerance that still catches a wrong
# formula, a wrong column, or a swapped standard.
WQI_VERIFICATION_TOLERANCE = 1e-3


def _find_raw_column(columns, prefix):
    """Return the one column whose name starts with ``prefix``."""
    matches = [c for c in columns if str(c).startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one column starting with {prefix!r}, "
            f"found {matches!r}"
        )
    return matches[0]


# ------------------------------------------------------------ audited file
def load_audited(path=AUDITED_DATA_PATH, verify=True):
    """Load the team's audited 1,572-row file (the primary input).

    Returns a dataframe with the 7 features (internal names, in
    FEATURES order), ``station_group_id``, ``year``, ``sample_id``,
    and ``wqi`` (the file's primary ``WQI`` column).

    If ``verify`` is True (the default), the file is checked before
    it is used: 1,572 rows, 356 groups, and the WQI column must
    match a fresh recomputation from the 7 raw features to within
    1e-3. A mismatch raises ValueError instead of silently training
    on a wrong target.
    """
    df = pd.read_csv(path)
    if len(df) != EXPECTED_CLEAN_ROWS:
        raise ValueError(
            f"Audited file has {len(df)} rows, expected "
            f"{EXPECTED_CLEAN_ROWS}"
        )
    if "WQI" not in df.columns:
        raise ValueError("Audited file has no 'WQI' column")
    if "station_group_id" not in df.columns:
        raise ValueError("Audited file has no 'station_group_id' column")

    out = pd.DataFrame()
    for feature in FEATURES:
        out[feature] = pd.to_numeric(
            df[_AUDITED_FEATURE_COLUMNS[feature]], errors="coerce"
        )
    out["station_group_id"] = df["station_group_id"].astype(str)
    out["year"] = pd.to_numeric(df["year"], errors="coerce")
    out["sample_id"] = df["sample_id"].astype(str)
    # PRIMARY target only. Two-sided-pH and asinh columns are
    # sensitivity variants and are deliberately never read here.
    out["wqi"] = pd.to_numeric(df["WQI"], errors="coerce")

    if out[FEATURES + ["wqi"]].isna().any().any():
        raise ValueError(
            "Audited file has missing values in the 7 features or WQI"
        )

    if verify:
        verify_audited(out, df)

    return out.reset_index(drop=True)


def verify_audited(cleaned, raw_audited=None):
    """Check the audited file's WQI column against the WAWQI formula.

    Recomputes WQI from the 7 raw features and compares it with the
    file's ``WQI`` column. Also checks the group count and, when the
    original audited frame is supplied, that the 7 q_i columns in
    the file match the formula's sub-indices exactly.
    Raises ValueError on any mismatch. Returns the max |difference|.
    """
    recomputed = exact_wqi(cleaned[FEATURES].to_numpy(dtype=float))
    max_diff = float(
        np.max(np.abs(recomputed - cleaned["wqi"].to_numpy(dtype=float)))
    )
    if max_diff >= WQI_VERIFICATION_TOLERANCE:
        raise ValueError(
            f"Audited WQI does not match recomputation: "
            f"max |difference| = {max_diff}, "
            f"tolerance = {WQI_VERIFICATION_TOLERANCE}"
        )
    n_groups = int(cleaned["station_group_id"].nunique())
    if n_groups != EXPECTED_N_GROUPS:
        raise ValueError(
            f"Audited file has {n_groups} station groups, "
            f"expected {EXPECTED_N_GROUPS}"
        )
    if raw_audited is not None and "qc_formula_match_original_WQI" in raw_audited:
        if not bool(raw_audited["qc_formula_match_original_WQI"].all()):
            raise ValueError(
                "Audited file's qc_formula_match_original_WQI is "
                "False for some rows"
            )
    return max_diff


# --------------------------------------------------------------- raw file
def load_raw(path=RAW_DATA_PATH):
    """Read the raw Kaggle CSV and return it unchanged (except header).

    The copy in this repo is UTF-8; a fresh download from Kaggle is
    latin-1. Try UTF-8 first and fall back to latin-1, so both load.
    """
    try:
        return pd.read_csv(path, encoding="utf-8", keep_default_na=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", keep_default_na=False)


def clean(df):
    """Apply the cleaning pipeline to the raw frame; see module docstring.

    Returns a frame with the 7 features as float columns named after
    FEATURES, plus ``station_group_id``, ``year``, and ``wqi``
    (recomputed from the 7 raw inputs with the WAWQI formula).
    """
    df = df.drop_duplicates().copy()

    # 7 feature columns, converted to numeric.
    for feature in FEATURES:
        raw_col = _find_raw_column(df.columns, _RAW_COLUMN_PREFIXES[feature])
        df[feature] = pd.to_numeric(df[raw_col], errors="coerce")

    # Drop rows missing any required input.
    df = df.dropna(subset=FEATURES)

    # Drop physically impossible negative measurements (none in this
    # file, but the check is part of the documented procedure).
    df = df[(df[FEATURES] >= 0).all(axis=1)]

    # Data-quality filter: pH must lie in [0, 14]. See the module
    # docstring for why these 53 rows are dropped, not repaired.
    df = df[(df["ph"] >= 0) & (df["ph"] <= 14)]

    # Station group: CODE:<code>, or LOC:<location> if code is 'NAN'.
    code = df["STATION CODE"].astype(str).str.strip()
    location = df["LOCATIONS"].astype(str).str.strip()
    df["station_group_id"] = np.where(
        code != "NAN", "CODE:" + code, "LOC:" + location
    )

    # Drop rows whose group identifier is missing/empty. (None in
    # this file; kept so a future data version fails loudly, not
    # silently, if a group cannot be formed.)
    df = df[df["station_group_id"].str.strip() != ""]
    df = df[~df["station_group_id"].isin(["CODE:", "LOC:"])]

    # Target: recompute WQI from the 7 raw inputs, every time.
    # We never trust a precomputed WQI column from the file.
    df["wqi"] = exact_wqi(df[FEATURES].to_numpy(dtype=float))

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df.reset_index(drop=True)


def load_cleaned(path=RAW_DATA_PATH):
    """Load the raw file and return the cleaned analysis dataframe."""
    return clean(load_raw(path))


# ------------------------------------------------------------- QC + audit
def qc_flags(df):
    """Count the quality-control flags on a cleaned dataframe.

    Flags overlap (one row can be flagged several times), so the
    counts must NOT be added together. These are the values the
    unit tests check against the technical document.
    """
    return {
        "fecal_coliform_gt_100": int((df["fecal_coliform"] > 100).sum()),
        "fecal_coliform_gt_1M": int((df["fecal_coliform"] > 1_000_000).sum()),
        "total_coliform_gt_1M": int((df["total_coliform"] > 1_000_000).sum()),
        "conductivity_gt_10000": int((df["conductivity"] > 10000).sum()),
        "bod_gt_100": int((df["bod"] > 100).sum()),
        "do_eq_0": int((df["do"] == 0).sum()),
        "ph_outside_6.5_8.5": int(
            ((df["ph"] < 6.5) | (df["ph"] > 8.5)).sum()
        ),
    }


def audit(df):
    """Return a dict summarising a cleaned dataset (for the report)."""
    return {
        "n_rows": int(len(df)),
        "n_groups": int(df["station_group_id"].nunique()),
        "n_code_rows": int(
            df["station_group_id"].str.startswith("CODE:").sum()
        ),
        "n_loc_rows": int(
            df["station_group_id"].str.startswith("LOC:").sum()
        ),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "wqi_min": float(df["wqi"].min()),
        "wqi_max": float(df["wqi"].max()),
        "wqi_median": float(df["wqi"].median()),
        "wqi_negative": int((df["wqi"] < 0).sum()),
        "qc_flags": qc_flags(df),
        "qc_flags_expected": dict(EXPECTED_QC_FLAGS),
    }


def _normalise_group_id(group_id):
    """Normalise a station_group_id for cross-file comparison.

    The audited file's LOC fallback IDs differ textually from the
    raw file's in two small ways, although they denote the same
    groups:
      * audited appends "|STATE:NAN" to LOC ids
        (e.g. "LOC:...|STATE:NAN" vs raw "LOC:...");
      * audited collapses double spaces in 3 location strings
        ("KAKKADAVU,  KERALA" -> "KAKKADAVU, KERALA", and two
        similar cases).
    Stripping the "|STATE:..." suffix and collapsing whitespace
    makes the two group label sets identical, and the per-group
    row counts are then equal for all 356 groups — i.e. the
    grouping *partition* is the same, only the labels differ.
    (CODE: ids are identical in both files.)
    """
    import re

    text = re.sub(r"\|STATE:.*$", "", str(group_id))
    return re.sub(r"\s+", " ", text).strip()


def cross_check_raw_vs_audited(raw_cleaned, audited):
    """Compare the raw-cleaned frame with the audited file.

    Exact row-by-row joining on sample_id is impossible: the
    audited file's sample_id (INDWQ_00001, ...) does not exist in
    the raw file. So this compares:
      * aggregates — row count, group count, QC flags, and WQI
        min/max/median;
      * the grouping partition — group label sets and per-group
        row counts, after normalising the LOC label differences
        described in ``_normalise_group_id``;
      * the row multiset — both frames sorted by year + the 7
        features + WQI must match row for row. A join on
        group+year+features is NOT used: two rows can share the
        same 7 features and year while belonging to different
        stations, so that join is not 1:1.

    Returns a dict with the comparison. Raises ValueError if any
    check disagrees beyond tolerance.
    """
    checks = {
        "n_rows_raw": int(len(raw_cleaned)),
        "n_rows_audited": int(len(audited)),
        "n_groups_raw": int(raw_cleaned["station_group_id"].nunique()),
        "n_groups_audited": int(audited["station_group_id"].nunique()),
        "qc_flags_raw": qc_flags(raw_cleaned),
        "qc_flags_audited": qc_flags(audited),
        "wqi_min_raw": float(raw_cleaned["wqi"].min()),
        "wqi_min_audited": float(audited["wqi"].min()),
        "wqi_max_raw": float(raw_cleaned["wqi"].max()),
        "wqi_max_audited": float(audited["wqi"].max()),
        "wqi_median_raw": float(raw_cleaned["wqi"].median()),
        "wqi_median_audited": float(audited["wqi"].median()),
    }
    if checks["n_rows_raw"] != checks["n_rows_audited"]:
        raise ValueError(
            f"Row count mismatch: raw-cleaned {checks['n_rows_raw']} "
            f"vs audited {checks['n_rows_audited']}"
        )
    if checks["n_groups_raw"] != checks["n_groups_audited"]:
        raise ValueError(
            f"Group count mismatch: raw-cleaned {checks['n_groups_raw']} "
            f"vs audited {checks['n_groups_audited']}"
        )
    if checks["qc_flags_raw"] != checks["qc_flags_audited"]:
        raise ValueError(
            f"QC flag mismatch: raw {checks['qc_flags_raw']} "
            f"vs audited {checks['qc_flags_audited']}"
        )
    for key in ("wqi_min", "wqi_max", "wqi_median"):
        raw_val = checks[key + "_raw"]
        aud_val = checks[key + "_audited"]
        if abs(raw_val - aud_val) > 1e-6 * max(1.0, abs(aud_val)):
            raise ValueError(
                f"{key} mismatch: raw {raw_val} vs audited {aud_val}"
            )
    # Grouping partition: same groups, same per-group row counts,
    # after normalising the LOC label differences.
    raw_groups = raw_cleaned["station_group_id"].map(_normalise_group_id)
    aud_groups = audited["station_group_id"].map(_normalise_group_id)
    if set(raw_groups) != set(aud_groups):
        raise ValueError(
            "Station group sets differ after normalising LOC labels: "
            f"only in raw {sorted(set(raw_groups) - set(aud_groups))[:5]}, "
            f"only in audited {sorted(set(aud_groups) - set(raw_groups))[:5]}"
        )
    raw_counts = raw_groups.value_counts().sort_index()
    aud_counts = aud_groups.value_counts().sort_index()
    if not raw_counts.equals(aud_counts):
        raise ValueError("Per-group row counts differ between raw and audited")
    checks["n_groups_normalised"] = int(len(set(raw_groups)))

    # Row multiset: sort both frames by year + 7 features + WQI and
    # compare row for row. Same features give the same WQI (the
    # formula is deterministic), so a sorted comparison is exact
    # even when two rows share the same feature vector.
    sort_cols = ["year"] + FEATURES + ["wqi"]
    raw_sorted = raw_cleaned.sort_values(sort_cols).reset_index(drop=True)
    aud_sorted = audited.sort_values(sort_cols).reset_index(drop=True)
    max_feature_diff = float(np.max(np.abs(
        raw_sorted[FEATURES].to_numpy(dtype=float)
        - aud_sorted[FEATURES].to_numpy(dtype=float)
    )))
    max_wqi_diff = float(np.max(np.abs(
        raw_sorted["wqi"].to_numpy(dtype=float)
        - aud_sorted["wqi"].to_numpy(dtype=float)
    )))
    checks["max_feature_diff_sorted"] = max_feature_diff
    checks["max_wqi_diff_sorted"] = max_wqi_diff
    checks["n_rows_sorted_multiset_match"] = int(len(raw_sorted))
    if max_feature_diff > 0 or max_wqi_diff >= WQI_VERIFICATION_TOLERANCE:
        raise ValueError(
            "Sorted row multiset mismatch: max feature diff "
            f"{max_feature_diff}, max WQI diff {max_wqi_diff}"
        )
    return checks
