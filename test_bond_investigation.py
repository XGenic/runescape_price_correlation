"""Regression checks for timing, lag-search calibration, and holdout isolation."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from bond_investigation_data import align_prices, prepare_controls
from bond_investigation_stats import (
    returns_stationary, lag_analysis, regime_analysis, granger_table,
)
from bond_investigation_validation import walk_forward, run_holdout_once


def returns_fixture(n=300):
    rng = np.random.default_rng(812)
    bond = rng.normal(0, .02, n)
    equity = np.roll(bond, 3) * .6 + rng.normal(0, .006, n)
    return pd.DataFrame({"bond_return": bond, "sp500_return": equity},
                        index=pd.bdate_range("2001-01-01", periods=n))


class InvestigationIntegrityTests(unittest.TestCase):
    def test_bucket_availability_and_missing_sessions_preserve_time(self):
        rs = pd.DataFrame({"price": [100., 110., 121.]},
                          index=pd.to_datetime(["2024-01-04", "2024-01-07", "2024-01-09"], utc=True))
        spx = pd.DataFrame({"spx_price": [100., 102., 103., 104.]},
                           index=pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"]))
        frame = align_prices(rs, spx)
        # Sunday's completed bucket is known Monday, including weekend movement.
        self.assertAlmostEqual(frame.loc["2024-01-08", "bond_return"], .1)
        self.assertAlmostEqual(frame.loc["2024-01-08", "sp500_return"], .02)
        self.assertTrue(np.isnan(frame.loc["2024-01-09", "bond_return"]))
        self.assertTrue(np.isnan(frame.loc["2024-01-10", "bond_return"]))

    def test_delayed_monthly_change_survives_weekend_release_proxy(self):
        frame = pd.DataFrame(index=pd.bdate_range("2023-03-30", "2023-04-04"))
        vix = pd.DataFrame({"spx_price": [10., 11., 12., 13., 14.]},
                           index=pd.bdate_range("2023-03-29", periods=5))
        sentiment = pd.DataFrame({"UMCSENT": [100., 120.]},
                                 index=pd.to_datetime(["2023-01-01", "2023-02-01"]))
        controls = prepare_controls(frame, vix, sentiment)
        self.assertTrue(np.isnan(controls.loc["2023-03-31", "umcsent_change"]))
        # February + two months lands on Saturday April 1; usable Monday April 3.
        self.assertAlmostEqual(controls.loc["2023-04-03", "umcsent_change"], .2)
        self.assertAlmostEqual(controls.loc["2023-04-03", "vix_return"], 12 / 11 - 1)

    def test_stationarity_requires_both_numeric_decisions_for_both_returns(self):
        rows = [{"series": name, "test": test, "pvalue": p,
                 "interpretation": "Non-stationary not rejected"}
                for name in ["bond_return", "sp500_return"]
                for test, p in [("ADF", .01), ("KPSS", .2)]]
        table = pd.DataFrame(rows)
        self.assertTrue(returns_stationary(table))
        table.loc[3, "pvalue"] = .01
        self.assertFalse(returns_stationary(table))
        table.loc[3, "pvalue"] = np.nan
        self.assertFalse(returns_stationary(table))

    def test_permutation_search_matches_pearson_at_signed_lags_with_gaps(self):
        frame = returns_fixture(120)
        frame.iloc[40, 0] = np.nan
        result = lag_analysis(frame, max_lag=5, iterations=12, seed=9)
        self.assertEqual(result["best_lag"], 3)
        for row in result["table"].itertuples():
            pairs = pd.concat([frame.bond_return.shift(row.lag), frame.sp500_return], axis=1).dropna()
            self.assertAlmostEqual(row.correlation, pearsonr(pairs.iloc[:, 0], pairs.iloc[:, 1]).statistic)
        x = frame.bond_return.to_numpy()
        finite = np.isfinite(x)
        rng = np.random.default_rng(9)
        expected = []
        for _ in range(12):
            shuffled = x.copy()
            shuffled[finite] = rng.permutation(x[finite] - x[finite].mean())
            correlations = []
            for lag in range(-5, 6):
                pairs = pd.concat([pd.Series(shuffled, index=frame.index).shift(lag), frame.sp500_return], axis=1).dropna()
                correlations.append(abs(pearsonr(pairs.iloc[:, 0], pairs.iloc[:, 1]).statistic))
            expected.append(max(correlations))
        np.testing.assert_allclose(result["null_max"], expected, atol=1e-12)
        self.assertEqual(result["shuffle_pvalue"], (1 + np.count_nonzero(np.array(expected) >= abs(result["best_corr"]))) / 13)

    def test_unseen_test_labels_cannot_change_locked_rule_or_first_forecasts(self):
        frame = returns_fixture()
        result = walk_forward(frame, 180, 240, lag=3)
        altered = frame.copy()
        altered.iloc[240:, altered.columns.get_loc("sp500_return")] *= -7
        counterfactual = walk_forward(altered, 180, 240, lag=3)
        self.assertEqual(result["locked_parameters"], counterfactual["locked_parameters"])
        pd.testing.assert_frame_equal(result["validation_scored"], counterfactual["validation_scored"])
        columns = ["fitted_return", "prediction_direction", "baseline_direction"]
        pd.testing.assert_frame_equal(result["test_scored"].iloc[:20][columns],
                                      counterfactual["test_scored"].iloc[:20][columns])
        windows = result["window_scores"]
        self.assertTrue((windows.history_end < windows.batch_start).all())
        self.assertEqual(result["test_scored"].index.min(), frame.index[240])

    def test_persisted_holdout_reproduces_report_and_rejects_retuning(self):
        context = {"frame": returns_fixture(), "train_end": 180,
                   "valid_end": 240, "fingerprint": "synthetic-integrity-check"}
        with tempfile.TemporaryDirectory() as directory:
            first = run_holdout_once(context, 3, Path(directory))
            again = run_holdout_once(context, 3, Path(directory))
            pd.testing.assert_frame_equal(first["test_scored"], again["test_scored"])
            with self.assertRaisesRegex(ValueError, "holdout is spent"):
                run_holdout_once(context, 4, Path(directory))

    def test_infeasible_granger_order_is_reported_not_fitted(self):
        result = granger_table(returns_fixture(120), best_lag=90)
        skipped = result.loc[result.lag_order == 90]
        self.assertEqual(len(skipped), 2)
        self.assertTrue(skipped.pvalue.isna().all())
        self.assertTrue(skipped.status.str.startswith("skipped:").all())

    def test_segmentation_can_choose_zero_or_detect_a_slope_reversal(self):
        rng = np.random.default_rng(103)
        x = rng.normal(0, .01, 240)
        noise = rng.normal(0, .0005, 240)
        frame = pd.DataFrame({"bond_return": x, "sp500_return": 2 * x + noise},
                             index=pd.bdate_range("2001-01-01", periods=240))
        stable = regime_analysis(frame)
        self.assertEqual(stable["break_dates"], [])
        frame.iloc[120:, 1] = -2 * x[120:] + noise[120:]
        changed = regime_analysis(frame)
        self.assertEqual(changed["break_dates"], [frame.index[120]])


if __name__ == "__main__":
    unittest.main()
