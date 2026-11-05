"""Single-sample demo: compute WQI from 7 raw measurements.

This script does NOT ship a pretrained model, and it does not
pretend to. There is no trained ANN in this repository — the
models are trained by ``src/run_experiment.py`` and their
predictions are written to the outputs folder. What this script
does is the honest, useful part: it computes the exact WAWQI
value for one water sample, shows every sub-index and weight,
and explains what a trained model would add (and what it would
not: for a complete sample, the formula is already the answer).

Usage:
    python -m src.predict_wqi --do 6.7 --ph 7.5 --conductivity 203 \\
        --bod 2.0 --nitrate 0.1 --fecal-coliform 11 --total-coliform 27
"""

import argparse

import numpy as np

from .config import FEATURES, FEATURE_LABELS
from .wqi import domain_relevance, exact_wqi, quality_ratings


def main():
    parser = argparse.ArgumentParser(
        description="Compute the exact WAWQI for one water sample."
    )
    parser.add_argument("--do", type=float, required=True,
                        help="Dissolved oxygen (mg/L)")
    parser.add_argument("--ph", type=float, required=True,
                        help="pH (0-14)")
    parser.add_argument("--conductivity", type=float, required=True,
                        help="Conductivity (µmhos/cm)")
    parser.add_argument("--bod", type=float, required=True,
                        help="Biochemical oxygen demand (mg/L)")
    parser.add_argument("--nitrate", type=float, required=True,
                        help="Nitrate/Nitrite (mg/L)")
    parser.add_argument("--fecal-coliform", type=float, required=True,
                        help="Fecal coliform (MPN/100 mL)")
    parser.add_argument("--total-coliform", type=float, required=True,
                        help="Total coliform (MPN/100 mL)")
    args = parser.parse_args()

    values = np.array([
        args.do, args.ph, args.conductivity, args.bod,
        args.nitrate, args.fecal_coliform, args.total_coliform,
    ], dtype=float)
    if not (0.0 <= args.ph <= 14.0):
        raise ValueError("pH must lie in [0, 14]")

    weights = domain_relevance()
    ratings = quality_ratings(values)
    wqi = exact_wqi(values)

    print("Input (raw measurements):")
    for i, feature in enumerate(FEATURES):
        print(f"  {FEATURE_LABELS[feature]:35s} {values[i]:>12.4g}")
    print("\nSub-index q_i and domain weight D_i:")
    for i, feature in enumerate(FEATURES):
        print(f"  {feature:16s} q = {ratings[i]:>12.4f}   "
              f"D = {weights[i]:.6f}   contribution = {weights[i] * ratings[i]:.4f}")
    print(f"\nExact WAWQI = {wqi:.4f}")
    print("\nNote: this is the formula value, not a model prediction.")
    print("A trained ANN would try to reconstruct this same number")
    print("from the same 7 inputs. Run the full experiment first:")
    print("    python -m src.run_experiment --mode full")


if __name__ == "__main__":
    main()
