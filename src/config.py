"""All constants for the WQI major project, in one place.

Why one file: so a viva examiner (or a teammate) can see every
number the experiment depends on without hunting through code.
Nothing in the other modules should hard-code a magic number
that lives here instead.

Title (frozen, do not change):
    A Hybrid Machine Learning Framework for Water Quality Index
    Prediction Using Adaptive Domain-Aware Neural Network Initialization
Team: Mohammed Nabi Adnan Ahmed (B23AI065), Sidda Samith (B23AI068),
Sai Charith Chekurthi (B23AI087), Komandla Meghana (B23AI129).
Supervisor: Smt. K. Rama Devi, KITSW CSE (AI & ML), Batch 03.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "water_dataX.csv"
# Team's audited, prepared 1,572-row file. This is the PRIMARY input
# for run_experiment.py. The raw Kaggle file above is kept as the
# source of truth, and data.py cross-checks the two against each other.
AUDITED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "water_quality_1572_audited_recomputed.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------- data
# Raw CSV columns in the Kaggle file (anbarivan/indian-water-quality-data).
# The conductivity column contains a micro sign (µ) whose exact bytes
# depend on the file encoding, so data.py matches it by prefix instead
# of by this exact string. The other names are given here for reference.
RAW_COLUMNS = {
    "station_code": "STATION CODE",
    "location": "LOCATIONS",
    "year": "year",
    "do": "D.O. (mg/l)",
    "ph": "PH",
    "conductivity": "CONDUCTIVITY (µmhos/cm)",
    "bod": "B.O.D. (mg/l)",
    "nitrate": "NITRATENAN N+ NITRITENANN (mg/l)",
    "fecal_coliform": "FECAL COLIFORM (MPN/100ml)",
    "total_coliform": "TOTAL COLIFORM (MPN/100ml)Mean",
}

# The seven model features, in the order used everywhere in the code.
# Keep this order fixed: W1 rows, gates, S/D/A vectors all follow it.
FEATURES = [
    "do",
    "ph",
    "conductivity",
    "bod",
    "nitrate",
    "fecal_coliform",
    "total_coliform",
]

FEATURE_LABELS = {
    "do": "Dissolved Oxygen (mg/L)",
    "ph": "pH",
    "conductivity": "Conductivity (µmhos/cm)",
    "bod": "Biochemical Oxygen Demand (mg/L)",
    "nitrate": "Nitrate/Nitrite (mg/L)",
    "fecal_coliform": "Fecal Coliform (MPN/100 mL)",
    "total_coliform": "Total Coliform (MPN/100 mL)",
}

# ------------------------------------------------------- WAWQI constants
# Weighted Arithmetic Water Quality Index.
# q_i = 100 * (V_i - ideal_i) / (Std_i - ideal_i)
# D_i = (1 / Std_i) / sum_j (1 / Std_j)      (domain relevance)
# WQI = sum_i D_i * q_i
#
# NOTE: these standard / ideal values are the ones encoded in the
# project implementation. They are NOT claimed here to be an official
# CPCB weight table; the report must cite an authoritative reference
# for each value before that claim is made.
STANDARDS = {
    "do": 10.0,
    "ph": 8.5,
    "conductivity": 1000.0,
    "bod": 5.0,
    "nitrate": 45.0,
    "fecal_coliform": 100.0,
    "total_coliform": 1000.0,
}

IDEALS = {
    "do": 14.6,
    "ph": 7.0,
    "conductivity": 0.0,
    "bod": 0.0,
    "nitrate": 0.0,
    "fecal_coliform": 0.0,
    "total_coliform": 0.0,
}

# ------------------------------------------------- adaptive importance
# A_i = (1 - beta) * S_i + beta * D_i
# beta = 0.2 was selected on the earlier 390-record track (beta in
# {0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}) and then FROZEN
# before the expanded 1,572-record experiment. It was not retuned
# against the outer test folds, and it is not claimed to be globally
# optimal. Do not change it without saying so in the report.
BETA = 0.2

# ------------------------------------------------------------- ANN
# Custom NumPy ANN only. No PyTorch / TensorFlow is used for the ANN.
HIDDEN_1 = 16
HIDDEN_2 = 8
LEARNING_RATE = 0.001
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8
MAX_EPOCHS = 1800
EARLY_STOPPING_PATIENCE = 140
SEEDS = (17, 41, 73)

# ------------------------------------------------- cross-validation
N_FOLDS = 5
VALIDATION_GROUP_FRACTION = 0.15  # of the remaining training groups
VALIDATION_SPLIT_SEED = 42

# ------------------------------------------------------- XGBoost / SHAP
# One XGBoost regressor is fitted per outer fold, on the training
# partition only. Its TreeSHAP values give S_i.
XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "random_state": 42,
    "n_jobs": -1,
}

# ------------------------------------------------------------- stats
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 42

# --------------------------------------------------- expected data audit
# These are the counts the cleaning pipeline must reproduce. The unit
# tests check them, so a change in the data or in data.py will fail
# loudly instead of silently changing the experiment.
EXPECTED_RAW_ROWS = 1991
EXPECTED_AFTER_DEDUP = 1990
EXPECTED_CLEAN_ROWS = 1572
EXPECTED_N_GROUPS = 356
EXPECTED_CODE_ROWS = 1479
EXPECTED_LOC_ROWS = 93
EXPECTED_QC_FLAGS = {
    "fecal_coliform_gt_100": 1008,
    "fecal_coliform_gt_1M": 3,
    "total_coliform_gt_1M": 3,
    "conductivity_gt_10000": 103,
    "bod_gt_100": 7,
    "do_eq_0": 1,
    "ph_outside_6.5_8.5": 116,
}
