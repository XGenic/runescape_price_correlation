"""Regression defenses for archive identity, gaps, and cross-source timing."""
import unittest

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

from bond_historical_data import audit_archive
from bond_historical_stats import granger_with_gaps, historical_regimes, stationarity_by_block
from bond_source_comparison import compare_sources


def fixture(n=300):
    rng = np.random.default_rng(390)
    x = rng.normal(0, .02, n)
    y = np.roll(x, 3) * .7 + rng.normal(0, .006, n)
    return pd.DataFrame({"bond_return": x, "sp500_return": y,
                         "bond_price": 100 * np.cumprod(1 + x),
                         "spx_price": 100 * np.cumprod(1 + y)},
                        index=pd.bdate_range("2002-01-01", periods=n))


class HistoricalIntegrityTests(unittest.TestCase):
    def test_archive_excludes_prelaunch_and_keeps_final_daily_quote_without_fill(self):
        dates = pd.to_datetime(["2015-03-28", "2015-03-30T10:00:00", "2015-03-30T20:00:00", "2015-04-02"], format="mixed", utc=True)
        raw = pd.DataFrame({"id": "13190", "timestamp": [int(d.timestamp() * 1000) for d in dates],
                            "price": [2000000, 1900000, 1850000, 1750000], "volume": None})
        daily, audit, gaps, excluded, _ = audit_archive(raw)
        self.assertEqual(daily.loc["2015-03-30", "price"], 1850000)
        self.assertNotIn(pd.Timestamp("2015-03-31"), daily.index)
        self.assertEqual(excluded.price.tolist(), [2000000])
        self.assertEqual(gaps.missing_days.tolist(), [2])
        self.assertEqual(audit["prelaunch_excluded"], 1)
        ambiguous = pd.concat([raw, raw.iloc[[1]].assign(price=1800000)], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Conflicting prices"):
            audit_archive(ambiguous)

    def test_regular_granger_matches_established_nested_f_test(self):
        frame = fixture(220)
        result = granger_with_gaps(frame, 3)
        got = result.loc[(result.direction == "Bond -> S&P 500") & (result.lag_order == 3)].iloc[0]
        expected = grangercausalitytests(frame[["sp500_return", "bond_return"]], maxlag=[3])[3][0]["ssr_ftest"]
        self.assertAlmostEqual(got.fstat, expected[0], places=9)
        self.assertAlmostEqual(got.pvalue, expected[1], places=12)

    def test_granger_never_treats_gap_neighbors_as_consecutive(self):
        frame = fixture(220)
        frame.iloc[50:55, frame.columns.get_loc("bond_return")] = np.nan
        result = granger_with_gaps(frame, 3)
        actual = result.loc[(result.direction == "Bond -> S&P 500") & (result.lag_order == 3)].iloc[0]
        # Independent chronological construction: only rows whose actual three
        # predecessor sessions exist may enter either nested fit.
        pairs = []
        for t in range(3, len(frame)):
            past = frame.iloc[t-3:t]
            if past[["bond_return", "sp500_return"]].notna().all().all():
                pairs.append([frame.sp500_return.iloc[t], *past.sp500_return.iloc[::-1], *past.bond_return.iloc[::-1]])
        values = np.array(pairs)
        restricted = sm.OLS(values[:, 0], sm.add_constant(values[:, 1:4])).fit()
        augmented = sm.OLS(values[:, 0], sm.add_constant(values[:, 1:])).fit()
        f, p, _ = augmented.compare_f_test(restricted)
        self.assertAlmostEqual(actual.fstat, f, places=9)
        self.assertAlmostEqual(actual.pvalue, p, places=12)

    def test_stationarity_coverage_does_not_bridge_missing_sessions(self):
        frame = fixture(120)
        frame.iloc[59, frame.columns.get_loc("bond_return")] = np.nan
        result = stationarity_by_block(frame, min_size=60)
        self.assertEqual(result["blocks"].n_obs.tolist(), [59, 60])
        self.assertAlmostEqual(result["coverage_fraction"], .5)
        self.assertEqual(set(result["table"].block_start), {frame.index[60]})

    def test_source_lag_sign_and_inherited_shift_keep_original_calendar(self):
        transaction = fixture(180)
        guide = transaction.copy()
        guide["bond_return"] = transaction.bond_return.shift(2)
        # Begin guide coverage later without erasing transaction predecessors.
        guide = guide.iloc[20:]
        guide.loc[transaction.index[60], "bond_return"] = np.nan
        result = compare_sources(guide, transaction, max_lag=5)
        self.assertEqual(result["best_lag"], 2)
        self.assertAlmostEqual(result["best_corr"], 1.)
        calendar = result["calendar"]
        # Missing source session 60 remains missing at target session 149.
        self.assertFalse(calendar.inherited_lag89_complete.iloc[149])
        self.assertAlmostEqual(calendar.guide_return_lag89.iloc[150], guide.bond_return.loc[transaction.index[61]])
        self.assertAlmostEqual(calendar.transaction_return_lag89.iloc[100], transaction.bond_return.iloc[11])

    def test_historical_segmentation_detects_slope_reversal_without_forcing_breaks(self):
        rng = np.random.default_rng(333)
        x = rng.normal(0, .01, 520)
        noise = rng.normal(0, .0005, 520)
        frame = pd.DataFrame({"bond_return": x, "sp500_return": 2*x+noise},
                             index=pd.bdate_range("2002-01-01", periods=520))
        self.assertEqual(historical_regimes(frame)["break_dates"], [])
        frame.iloc[260:, 1] = -2*x[260:]+noise[260:]
        self.assertEqual(historical_regimes(frame)["break_dates"], [frame.index[260]])


if __name__ == "__main__":
    unittest.main()
