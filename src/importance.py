"""Feature importance: XGBoost + TreeSHAP, and the adaptive score.

Two sources of "how important is this feature", combined into one:

* S_i — data-driven importance. Train an XGBoost regressor on the
  TRAINING partition only, compute TreeSHAP values on the training
  rows, take the mean absolute SHAP value per feature, and
  normalise by the sum. S_i sums to 1.
* D_i — domain relevance. The normalised WAWQI weights, D_i =
  (1/Std_i) / sum_j (1/Std_j), from wqi.domain_relevance(). D_i
  sums to 1. This is domain knowledge from the WQI formula itself,
  not an expert rating and not a learned model.

Adaptive importance:
    A_i = (1 - beta) * S_i + beta * D_i,  beta = 0.2 (frozen).
Since both parts sum to 1, A_i also sums to 1 and 0 <= A_i <= 1.

Why XGBoost for S_i and not the ANN itself? Using the ANN's own
explanations to initialise that same ANN would be circular. The
XGBoost model is a separate, independent prior, computed before
any ANN training. It is also why the SHAP calculation must use
training data only — computing it on the whole dataset would leak
test information into the initialisation.

Note on SHAP versions: shap's TreeExplainer returns values of
shape (n_samples, n_features) for a regressor. If a future version
returns an extra dimension, ``_shap_values`` squeezes it; the
unit tests do not depend on SHAP, so a change here fails in the
experiment run, not silently.
"""

import numpy as np
import shap
from xgboost import XGBRegressor

from .config import BETA, FEATURES, XGBOOST_PARAMS
from .wqi import domain_relevance


def fit_xgboost(X_train, y_train):
    """Fit the XGBoost regressor used for SHAP (training data only)."""
    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X_train, y_train)
    return model


def shap_importance(model, X_train):
    """Mean |TreeSHAP| per feature, normalised to sum to 1.

    ``X_train`` must be the same (raw, unscaled) training rows the
    model was fitted on. The SHAP base value (bias) column, if
    present, is ignored — only the 7 per-feature contributions count.
    """
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(X_train), dtype=float)
    # Guard against a (n, 7, 1) or similar return shape.
    while values.ndim > 2:
        values = values[:, :, 0]
    if values.shape[1] != len(FEATURES):
        raise ValueError(
            f"SHAP values have {values.shape[1]} columns, "
            f"expected {len(FEATURES)}"
        )
    mean_abs = np.mean(np.abs(values), axis=0)
    total = mean_abs.sum()
    if total <= 0:
        raise ValueError("SHAP importances sum to zero")
    return mean_abs / total


def adaptive_importance(shap_scores, beta=BETA):
    """Combine S_i and D_i into A_i = (1-beta)*S_i + beta*D_i."""
    shap_scores = np.asarray(shap_scores, dtype=float)
    domain = domain_relevance()
    adaptive = (1.0 - beta) * shap_scores + beta * domain
    return adaptive


def compute_importance(X_train, y_train):
    """Fit XGBoost on the training partition and return S, D, A.

    Returns a dict with keys ``S``, ``D``, ``A`` (each a length-7
    array in FEATURES order, summing to 1) and ``model`` (the fitted
    XGBoost regressor, returned so the caller can reuse it as a
    baseline instead of fitting it twice).
    """
    model = fit_xgboost(X_train, y_train)
    shap_scores = shap_importance(model, X_train)
    return {
        "S": shap_scores,
        "D": domain_relevance(),
        "A": adaptive_importance(shap_scores),
        "model": model,
    }
