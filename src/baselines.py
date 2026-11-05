"""Classical baselines and the exact-formula reference.

Baselines are fitted on the training partition only, on the RAW
(unscaled) inputs and target — NOT on the standardised data the
ANN uses. That is deliberate, and it matters for Ridge:
    * Ridge regression — the most important baseline in this project.
      Because WQI is a linear function of the 7 inputs, Ridge can
      reconstruct the formula almost exactly. If Ridge beats the
      ANNs by a large margin, that is the expected result, not a bug.
      But Ridge is scale-sensitive: fitting it on the standardised
      target (whose SD, ~200,000, is dominated by 3 extreme WQI
      values) with alpha=1.0 shrinks the coliform coefficients and
      degrades it to MAE ~36 on fold 1, instead of ~0.01 on raw
      inputs. The document's Ridge result (pooled MAE 0.2162) is
      reproduced with raw inputs + alpha=1.0, so raw inputs it is.
      (Tree models are invariant to monotone scaling of X and y,
      so the choice does not affect them.)
    * XGBoost regressor — the same model that supplies SHAP, fitted
      here as a standalone predictor.
    * Random Forest, ExtraTrees, HistGradientBoosting — standard
      tree ensembles, included because the technical document's
      final comparison includes them.

Exact-formula reference:
    ``exact_wqi`` from wqi.py is not a model. It computes WQI
    directly from the 7 raw inputs and is expected to give ~0 error.
    It is included so the report can say, honestly, how far every
    learned model is from the formula that generated the target.

NOT included, deliberately:
    * SVR — it appears in the base paper and in the literature, but
      it is NOT in the final expanded experiment's comparison table.
    * XGBoost late fusion / stacking on top of the ANN — the Phase-1
      proposal discussed it, but the final primary run does not
      include a separately trained XGBoost refinement stage. The
      word "Hybrid" in the project title therefore refers to the
      XGBoost-informed initialisation (SHAP -> ANN), not to a
      late-fusion ensemble. Say so in the report; do not claim a
      fusion stage was run.
"""

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge

from .config import XGBOOST_PARAMS
from .importance import fit_xgboost
from .wqi import exact_wqi


def fit_baselines(X_train, y_train, xgboost_model=None):
    """Fit all classical baselines on RAW training data.

    Returns {name: fitted model}. ``X_train`` / ``y_train`` must be
    the raw (unscaled) training inputs / target — see the module
    docstring for why Ridge needs raw inputs.

    ``xgboost_model``: if the caller already fitted an XGBoost
    regressor on exactly this training data (the one fitted for
    SHAP in importance.py), pass it in and it is reused as the
    XGBoost baseline instead of being fitted a second time.
    """
    models = {
        "ridge": Ridge(alpha=1.0),
        "xgboost": xgboost_model if xgboost_model is not None
        else fit_xgboost(X_train, y_train),
        "random_forest": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "hist_gb": HistGradientBoostingRegressor(random_state=42),
    }
    for name, model in models.items():
        if name != "xgboost":
            model.fit(X_train, y_train)
    return models


def predict_baselines(models, X):
    """Return {name: predictions of shape (n,)} for RAW inputs X."""
    return {name: model.predict(X) for name, model in models.items()}


def exact_formula_predictions(X_raw):
    """WQI from the 7 RAW inputs. X_raw must be unscaled, in FEATURES order."""
    return exact_wqi(np.asarray(X_raw, dtype=float))
