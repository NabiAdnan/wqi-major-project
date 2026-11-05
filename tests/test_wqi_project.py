"""Unit tests for the WQI major project.

Run from the project root with the test venv, e.g.:
    python -m unittest discover -s tests -v

These tests check the facts the whole experiment rests on. They
do NOT run the full experiment and they do NOT fit XGBoost/SHAP
(that is slow and belongs in the experiment run, not in a unit
test). The ANN tests use tiny synthetic data.

If a test fails after a data or code change, do not "fix" the
expected numbers in the test to make it pass — find out whether
the data changed or the code is wrong, and say so in the report.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import ann, gated  # noqa: E402
from src.config import (  # noqa: E402
    EXPECTED_CLEAN_ROWS,
    EXPECTED_CODE_ROWS,
    EXPECTED_LOC_ROWS,
    EXPECTED_N_GROUPS,
    EXPECTED_QC_FLAGS,
    FEATURES,
)
from src.data import (  # noqa: E402
    audit,
    cross_check_raw_vs_audited,
    load_audited,
    load_cleaned,
    qc_flags,
)
from src.evaluate import regression_metrics  # noqa: E402
from src.importance import adaptive_importance  # noqa: E402
from src.wqi import domain_relevance, exact_wqi, quality_ratings  # noqa: E402


class TestWqiFormula(unittest.TestCase):
    def test_domain_relevance_sums_to_one(self):
        d = domain_relevance()
        self.assertEqual(len(d), 7)
        self.assertAlmostEqual(float(d.sum()), 1.0, places=12)
        # BOD has the largest weight, conductivity/total the smallest.
        self.assertAlmostEqual(d[FEATURES.index("bod")], 0.442606, places=5)
        self.assertAlmostEqual(d[FEATURES.index("ph")], 0.260356, places=5)
        self.assertAlmostEqual(d[FEATURES.index("do")], 0.221303, places=5)

    def test_wqi_is_100_when_every_value_equals_its_standard(self):
        # q_i = 100 for every feature, weights sum to 1, so WQI = 100.
        values = np.array([10.0, 8.5, 1000.0, 5.0, 45.0, 100.0, 1000.0])
        self.assertAlmostEqual(exact_wqi(values), 100.0, places=9)

    def test_wqi_is_0_when_every_value_equals_its_ideal(self):
        values = np.array([14.6, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(exact_wqi(values), 0.0, places=9)

    def test_ph_below_7_gives_negative_sub_index(self):
        # One-sided pH formula: pH 6 -> q_ph = 100*(6-7)/(8.5-7) < 0.
        ratings = quality_ratings(
            np.array([14.6, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
        self.assertLess(ratings[FEATURES.index("ph")], 0.0)

    def test_adaptive_importance_sums_to_one(self):
        # Any S that sums to 1 gives an A that sums to 1 (beta = 0.2).
        fake_s = np.array([0.2, 0.1, 0.05, 0.25, 0.1, 0.2, 0.1])
        a = adaptive_importance(fake_s)
        self.assertAlmostEqual(float(a.sum()), 1.0, places=12)
        self.assertTrue(np.all(a >= 0) and np.all(a <= 1))


class TestAuditedDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_audited()

    def test_shape_is_1572_by_47_in_source_file(self):
        import pandas as pd
        from src.config import AUDITED_DATA_PATH
        raw = pd.read_csv(AUDITED_DATA_PATH)
        self.assertEqual(raw.shape, (1572, 47))

    def test_cleaned_frame_has_1572_rows_and_356_groups(self):
        self.assertEqual(len(self.df), EXPECTED_CLEAN_ROWS)
        self.assertEqual(
            int(self.df["station_group_id"].nunique()), EXPECTED_N_GROUPS
        )
        self.assertEqual(
            int(self.df["station_group_id"].str.startswith("CODE:").sum()),
            EXPECTED_CODE_ROWS,
        )
        self.assertEqual(
            int(self.df["station_group_id"].str.startswith("LOC:").sum()),
            EXPECTED_LOC_ROWS,
        )

    def test_wqi_column_matches_recomputation(self):
        # load_audited() already verifies this and would have raised;
        # check it explicitly here so the test name says what it is.
        from src.data import verify_audited
        max_diff = verify_audited(self.df)
        self.assertLess(max_diff, 1e-3)

    def test_wqi_summary_statistics(self):
        self.assertAlmostEqual(float(self.df["wqi"].min()),
                               -10.56410760643171, places=6)
        self.assertAlmostEqual(float(self.df["wqi"].max()),
                               6145091.292130445, places=3)
        self.assertAlmostEqual(float(self.df["wqi"].median()),
                               70.74234447723006, places=4)
        self.assertEqual(int((self.df["wqi"] < 0).sum()), 3)

    def test_qc_flags_match_the_document(self):
        self.assertEqual(qc_flags(self.df), EXPECTED_QC_FLAGS)

    def test_primary_wqi_is_not_a_sensitivity_column(self):
        # The audited file also carries two-sided-pH and asinh
        # variants. The primary WQI must differ from them, so a
        # test that accidentally read the wrong column would fail
        # the recomputation check above — this test just makes the
        # distinction visible.
        import pandas as pd
        from src.config import AUDITED_DATA_PATH
        raw = pd.read_csv(AUDITED_DATA_PATH)
        self.assertFalse(
            np.allclose(raw["WQI"], raw["WQI_two_sided_pH_sensitivity"])
        )


class TestRawCleaning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_cleaned()

    def test_cleaning_reproduces_1572_rows_and_356_groups(self):
        self.assertEqual(len(self.df), EXPECTED_CLEAN_ROWS)
        self.assertEqual(
            int(self.df["station_group_id"].nunique()), EXPECTED_N_GROUPS
        )

    def test_qc_flags_match_the_document(self):
        self.assertEqual(qc_flags(self.df), EXPECTED_QC_FLAGS)

    def test_audit_matches_expected_counts(self):
        summary = audit(self.df)
        self.assertEqual(summary["n_rows"], EXPECTED_CLEAN_ROWS)
        self.assertEqual(summary["n_groups"], EXPECTED_N_GROUPS)
        self.assertEqual(summary["year_min"], 2005)
        self.assertEqual(summary["year_max"], 2014)
        self.assertEqual(summary["wqi_negative"], 3)


class TestCrossCheck(unittest.TestCase):
    def test_raw_cleaned_and_audited_agree(self):
        # Aggregates + grouping partition + sorted row multiset.
        # No sample_id join: the raw file has no sample_id column.
        # No group+year+features join either: two rows can share
        # the same features and year at different stations, and
        # the audited LOC group labels differ textually from the
        # raw ones ("|STATE:NAN" suffix, collapsed double spaces).
        result = cross_check_raw_vs_audited(load_cleaned(), load_audited())
        self.assertEqual(result["n_rows_raw"], 1572)
        self.assertEqual(result["n_rows_audited"], 1572)
        self.assertEqual(result["n_groups_normalised"], 356)
        self.assertEqual(result["n_rows_sorted_multiset_match"], 1572)
        self.assertLess(result["max_wqi_diff_sorted"], 1e-3)


class TestAnn(unittest.TestCase):
    def test_forward_shape(self):
        rng = np.random.default_rng(0)
        params = ann.init_params(rng)
        X = rng.normal(size=(5, 7))
        y_hat, _ = ann.forward(params, X)
        self.assertEqual(y_hat.shape, (5, 1))

    def test_frobenius_restoration_preserves_norm(self):
        rng = np.random.default_rng(0)
        W1 = ann.xavier_uniform(rng, 7, 16)
        importance = np.array([0.2, 0.1, 0.05, 0.25, 0.1, 0.2, 0.1])
        scaled = ann.scale_rows_by_importance(W1, importance)
        self.assertAlmostEqual(
            float(np.linalg.norm(scaled)), float(np.linalg.norm(W1)),
            places=10,
        )
        # ... but the rows are no longer equal: importance changed
        # the distribution, which is the whole point.
        self.assertFalse(np.allclose(scaled, W1))

    def test_gates_are_positive_and_unbounded(self):
        # exp(theta) is always > 0, and can exceed 1 — this is the
        # correction to the sigmoid diagrams. A gate of, say, 1.5
        # must be representable, not clipped.
        adaptive = np.array([0.19, 0.07, 0.02, 0.25, 0.06, 0.23, 0.18])
        gates = gated.initial_gates(adaptive)
        self.assertTrue(np.all(gates > 0))
        log_gate = np.log(gates)
        self.assertTrue(np.all(gated.gates_from_log_gate(log_gate) > 0))
        self.assertTrue(np.allclose(
            gated.gates_from_log_gate(np.log(1.5)), 1.5
        ))

    def test_log_gate_gradient_matches_finite_differences(self):
        # The gate gradient is easy to get wrong by a factor of g_i:
        # it must use the PRE-gating input, not the gated input.
        # Check the analytic gradient against a finite-difference
        # gradient on a tiny synthetic batch.
        rng = np.random.default_rng(2)
        params = ann.init_params(rng)
        X = rng.normal(size=(8, 7))
        y = rng.normal(size=(8, 1))
        adaptive = np.array([0.19, 0.07, 0.02, 0.25, 0.06, 0.23, 0.18])
        log_gate = np.log(gated.initial_gates(adaptive))
        params["log_gate"] = log_gate.copy()
        gates = gated.gates_from_log_gate(log_gate)
        y_hat, cache = ann.forward(params, X, gates=gates)
        grads = ann.backward(params, cache, y_hat, y)
        analytic = gates * np.sum(
            grads["dX_gated"] * cache["X"], axis=0
        )
        eps = 1e-6
        numeric = np.zeros(7)
        for i in range(7):
            for sign in (+1, -1):
                lg = log_gate.copy()
                lg[i] += sign * eps
                g = gated.gates_from_log_gate(lg)
                yh, _ = ann.forward(params, X, gates=g)
                loss = ann.mse_loss(yh, y)
                numeric[i] += sign * loss
            numeric[i] /= 2 * eps
        self.assertTrue(np.allclose(analytic, numeric, rtol=1e-4, atol=1e-6),
                        f"analytic {analytic} vs numeric {numeric}")

    def test_one_training_step_reduces_loss_on_tiny_synthetic_batch(self):
        # Tiny learnable problem: y = sum of the 7 inputs. A few
        # full-batch Adam steps must reduce the MSE, or backprop
        # is broken.
        rng = np.random.default_rng(1)
        X = rng.normal(size=(32, 7))
        y = X.sum(axis=1, keepdims=True)
        X_val, y_val = X[:8], y[:8]
        params, history = ann.train_ann(
            X, y, X_val, y_val, method="xavier", seed=17,
            max_epochs=60, patience=60,
        )
        self.assertLess(history["train_mse"][-1], history["train_mse"][0])

    def test_groupkfold_has_zero_group_overlap(self):
        from sklearn.model_selection import GroupKFold
        df = load_audited()
        groups = df["station_group_id"].to_numpy()
        X = df[FEATURES].to_numpy(dtype=float)
        y = df["wqi"].to_numpy(dtype=float)
        for train_idx, test_idx in GroupKFold(n_splits=5).split(X, y, groups):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            self.assertEqual(train_groups & test_groups, set())

    def test_metrics_on_a_known_case(self):
        metrics = regression_metrics(
            np.array([0.0, 10.0, 20.0]), np.array([0.0, 10.0, 20.0])
        )
        self.assertAlmostEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["r2"], 1.0)


if __name__ == "__main__":
    unittest.main()
