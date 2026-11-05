"""Weighted Arithmetic Water Quality Index (WAWQI).

This module contains only the WQI formula. It knows nothing about
files, models, or cross-validation, so it can be tested on its own.

Formula (for each of the 7 parameters):
    q_i  = 100 * (V_i - ideal_i) / (Std_i - ideal_i)   (quality rating)
    D_i  = (1 / Std_i) / sum_j (1 / Std_j)             (domain relevance)
    WQI  = sum_i D_i * q_i

Two things to understand before using it:
1. The standards, ideals and weights are fixed, so WQI is a *linear*
   function of the 7 inputs. The ANN in this project is therefore
   learning to reconstruct a known formula, not discovering a new
   relationship. The exact formula (``exact_wqi`` below) reproduces
   the target with ~0 error, and Ridge regression gets very close.
   That is the honest baseline every other method is judged against.
2. The pH sub-index is one-sided: ideal pH is 7.0, standard is 8.5,
   so a pH below 7 gives a *negative* q_i. That is why the cleaned
   dataset contains 3 negative WQI values. We keep the formula as
   specified; a two-sided pH version is a sensitivity variant only
   and must not be silently substituted for this one.
"""

import numpy as np

from .config import FEATURES, IDEALS, STANDARDS

# STANDARDS / IDEALS are dicts in config; the arrays below are built
# in FEATURES order so index i always means feature i.


def domain_relevance():
    """Return D_i for the 7 features, in FEATURES order.

    D_i = (1 / Std_i) / sum_j (1 / Std_j).  Sums to 1 by construction.
    BOD gets the largest weight (0.4426) because its standard is the
    smallest (5); conductivity and total coliform get the smallest
    (0.0022 each) because their standards are the largest (1000).
    """
    standards = np.array([STANDARDS[f] for f in FEATURES], dtype=float)
    reciprocal = 1.0 / standards
    return reciprocal / reciprocal.sum()


def quality_ratings(values):
    """Return the sub-index q_i for each feature.

    ``values`` is an array of shape (..., 7) in FEATURES order, or a
    single row of shape (7,). The return has the same shape.
    """
    values = np.asarray(values, dtype=float)
    standards = np.array([STANDARDS[f] for f in FEATURES], dtype=float)
    ideals = np.array([IDEALS[f] for f in FEATURES], dtype=float)
    return 100.0 * (values - ideals) / (standards - ideals)


def exact_wqi(values):
    """Compute WQI directly from the 7 raw measurements.

    ``values`` is an array of shape (n, 7) or (7,) in FEATURES order.
    Returns the WQI for each row. This is the mathematical reference:
    it is not a model, and it is expected to reproduce the target
    with (numerically) zero error.
    """
    values = np.asarray(values, dtype=float)
    single_row = values.ndim == 1
    if single_row:
        values = values.reshape(1, -1)
    wqi = quality_ratings(values) @ domain_relevance()
    return float(wqi[0]) if single_row else wqi
