"""Descriptive GE guide-price versus transaction-average comparisons.

Inputs already encode conservative next-UTC-day availability. Archive guide
quotes are not RuneLite bucket starts; this module neither retimestamps quotes
nor recomputes returns. Missing equity-session positions survive every shift.
All selected lags are descriptive, without significance or validation claims.
"""

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = ("bond_price", "bond_return", "sp500_return")
_INHERITED_LAG = 89


def _check_frame(frame, name):
    """Require an uncompressed, chronological daily equity-session frame."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must have a DatetimeIndex.")
    if (frame.index.tz is not None or frame.index.hasnans
            or frame.index.has_duplicates or not frame.index.is_monotonic_increasing
            or not frame.index.equals(frame.index.normalize())):
        raise ValueError(f"{name} must have unique chronological naive daily dates.")
    missing = set(_REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    for column in _REQUIRED_COLUMNS:
        values = frame[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}.{column} contains nonfinite observations.")
        if column == "bond_price" and (values <= 0).any():
            raise ValueError(f"{name}.bond_price must be positive where observed.")


def _correlation(sample, left, right, method="pearson", min_obs=2):
    """Sample must already be complete on the comparison's common columns."""
    if (len(sample) < min_obs or sample[left].nunique() < 2
            or sample[right].nunique() < 2):
        return np.nan
    return float(sample[left].corr(sample[right], method=method))


def _dates(sample):
    return ((sample.index[0], sample.index[-1]) if len(sample)
            else (pd.NaT, pd.NaT))


def compare_sources(guide_frame, transaction_frame, max_lag=20):
    """Return paired, summary, lags, best_lag, best_corr, calendar and status.

    ``paired`` retains the union of supplied equity-session labels within their
    shared date range, including missing prices/returns and one-source-only
    labels. Inputs must retain their original full equity calendars: dates
    missing from BOTH inputs cannot be recovered here. Existing return values
    are preserved, including a boundary return using a preceding source date.

    Positive cross-source lag k means transaction returns LEAD guide returns:
    corr(transaction_return.shift(k), guide_return). Each signed lag requires
    at least 30 complete pairs. Maximum absolute correlation is an exploratory
    descriptive selection only; ties resolve to the first (smallest) lag.

    SPX comparisons use transaction_frame.sp500_return as a common reference,
    with both bond sources on identical complete rows. The inherited +89 lag
    means each bond source leads SPX, not the cross-source lag convention.
    Both bond returns are first aligned to the ENTIRE original transaction
    study calendar, shifted there, then restricted to overlapping target dates
    and common complete rows. This is an exposed prior-study choice, not an
    independently selected validation or confirmatory test.

    ``calendar`` is that full original transaction calendar, with unshifted
    returns, shifted returns, overlap and common-row flags. Fractions, sample
    SDs (ddof=1), and return correlations in ``summary`` use paired complete
    returns unless explicitly marked source-available. Prices use their own
    common complete sample. Sign is -1/0/+1; zero is flat, never positive.
    """
    _check_frame(guide_frame, "guide_frame")
    _check_frame(transaction_frame, "transaction_frame")
    if isinstance(max_lag, (bool, np.bool_)) or not isinstance(max_lag, (int, np.integer)) or max_lag < 0:
        raise ValueError("max_lag must be a nonnegative integer.")

    if guide_frame.empty or transaction_frame.empty:
        index = pd.DatetimeIndex([], name="date")
    else:
        start = max(guide_frame.index[0], transaction_frame.index[0])
        end = min(guide_frame.index[-1], transaction_frame.index[-1])
        index = guide_frame.index.union(transaction_frame.index).sort_values()
        index = index[(index >= start) & (index <= end)].rename("date")
    guide = guide_frame.reindex(index)
    transaction = transaction_frame.reindex(index)
    paired = pd.DataFrame({
        "guide_price": guide["bond_price"],
        "transaction_price": transaction["bond_price"],
        "guide_return": guide["bond_return"],
        "transaction_return": transaction["bond_return"],
    }, index=index)
    returns = paired[["guide_return", "transaction_return"]].dropna()
    prices = paired[["guide_price", "transaction_price"]].dropna()
    price_discrepancy = (prices["guide_price"] / prices["transaction_price"] - 1).abs()
    signs = np.sign(returns)
    both_nonflat = (signs != 0).all(axis=1)
    both_flat = (signs == 0).all(axis=1)
    one_flat = (signs == 0).sum(axis=1) == 1
    opposite = signs["guide_return"] * signs["transaction_return"] == -1
    return_difference = (returns["guide_return"] - returns["transaction_return"]).abs()
    guide_sd = float(returns["guide_return"].std(ddof=1))
    transaction_sd = float(returns["transaction_return"].std(ddof=1))

    lag_rows = []
    for lag in range(-max_lag, max_lag + 1):
        sample = pd.concat([
            paired["transaction_return"].shift(lag), paired["guide_return"],
        ], axis=1).dropna()
        lag_rows.append({
            "lag": lag,
            "correlation": _correlation(sample, "transaction_return", "guide_return", min_obs=30),
            "n_obs": len(sample),
        })
    lags = pd.DataFrame(lag_rows, columns=["lag", "correlation", "n_obs"])
    estimable = lags["correlation"].notna()
    best_lag, best_corr = None, np.nan
    if estimable.any():
        best = lags.loc[lags.loc[estimable, "correlation"].abs().idxmax()]
        best_lag, best_corr = int(best["lag"]), float(best["correlation"])

    # Never shift the shared-range slice: that would erase original lag-89
    # predecessors, and never drop incomplete rows until after both shifts.
    calendar = pd.DataFrame({
        "guide_return": guide_frame["bond_return"].reindex(transaction_frame.index),
        "transaction_return": transaction_frame["bond_return"],
        "sp500_return": transaction_frame["sp500_return"],
    }, index=transaction_frame.index).rename_axis("date")
    calendar["guide_return_lag89"] = calendar["guide_return"].shift(_INHERITED_LAG)
    calendar["transaction_return_lag89"] = calendar["transaction_return"].shift(_INHERITED_LAG)
    calendar["in_overlap"] = calendar.index.isin(index)
    contemporaneous_columns = ["guide_return", "transaction_return", "sp500_return"]
    inherited_columns = ["guide_return_lag89", "transaction_return_lag89", "sp500_return"]
    calendar["contemporaneous_complete"] = (
        calendar["in_overlap"] & calendar[contemporaneous_columns].notna().all(axis=1))
    calendar["inherited_lag89_complete"] = (
        calendar["in_overlap"] & calendar[inherited_columns].notna().all(axis=1))
    contemporaneous = calendar.loc[calendar["contemporaneous_complete"], contemporaneous_columns]
    inherited = calendar.loc[calendar["inherited_lag89_complete"], inherited_columns]
    spx_overlap = pd.DataFrame({
        "guide_sp500_return": guide["sp500_return"],
        "transaction_sp500_return": transaction["sp500_return"],
    }).dropna()
    spx_discrepancy = (spx_overlap["guide_sp500_return"] - spx_overlap["transaction_sp500_return"]).abs()

    summary = {
        "interpretation": "Exploratory source/smoothing diagnostics; no significance or confirmatory claims.",
        "return_units": "Fractional equity-session changes, retained from each input; no fill or recomputation.",
        "overlap_calendar": "Union of supplied equity sessions inside the shared date range; missing positions retained.",
        "cross_source_lag_convention": "Positive k: corr(transaction_return.shift(k), guide_return); transaction leads guide.",
        "lag_selection": "Maximum absolute Pearson over all signed lags; descriptive only, no multiplicity-adjusted inference.",
        "lag_minimum_pairs": 30,
        "max_lag": int(max_lag),
        "sign_convention": "-1 negative, 0 flat, +1 positive; sign disagreement includes exactly one flat.",
        "fraction_denominator": "Paired complete returns except opposite_nonflat_fraction (both nonflat) and source_available metrics.",
        "spx_reference": "Original transaction-study SPX returns; both bond sources use identical complete target rows.",
        "spx_inherited_lag89_interpretation": "Inherited exposed prior-study +89-session bond-leading-SPX lag; not independently selected validation.",
        "spx_inherited_lag89_calendar": "Reindex guide to full original transaction calendar, shift BOTH by 89, then filter shared targets/common complete rows.",
        "guide_input_sessions": len(guide_frame),
        "transaction_input_sessions": len(transaction_frame),
        "guide_input_start": _dates(guide_frame)[0],
        "guide_input_end": _dates(guide_frame)[1],
        "transaction_input_start": _dates(transaction_frame)[0],
        "transaction_input_end": _dates(transaction_frame)[1],
        "overlap_sessions": len(paired),
        "overlap_start": _dates(paired)[0],
        "overlap_end": _dates(paired)[1],
        "guide_missing_calendar_labels": int((~index.isin(guide_frame.index)).sum()),
        "transaction_missing_calendar_labels": int((~index.isin(transaction_frame.index)).sum()),
        "guide_available_price_n": int(paired["guide_price"].notna().sum()),
        "transaction_available_price_n": int(paired["transaction_price"].notna().sum()),
        "guide_available_return_n": int(paired["guide_return"].notna().sum()),
        "transaction_available_return_n": int(paired["transaction_return"].notna().sum()),
        "paired_return_n": len(returns),
        "paired_return_start": _dates(returns)[0],
        "paired_return_end": _dates(returns)[1],
        "paired_price_n": len(prices),
        "paired_price_start": _dates(prices)[0],
        "paired_price_end": _dates(prices)[1],
        "return_pearson": _correlation(returns, "guide_return", "transaction_return"),
        "return_spearman": _correlation(returns, "guide_return", "transaction_return", method="spearman"),
        "return_unequal_fraction": float((return_difference != 0).mean()),
        "return_abs_difference_mean": float(return_difference.mean()),
        "return_abs_difference_median": float(return_difference.median()),
        "sign_disagreement_fraction": float((signs["guide_return"] != signs["transaction_return"]).mean()),
        "opposite_sign_fraction": float(opposite.mean()),
        "both_nonflat_n": int(both_nonflat.sum()),
        "opposite_nonflat_fraction": float(opposite.loc[both_nonflat].mean()),
        "exactly_one_flat_fraction": float(one_flat.mean()),
        "both_flat_fraction": float(both_flat.mean()),
        "guide_zero_return_fraction": float((returns["guide_return"] == 0).mean()),
        "transaction_zero_return_fraction": float((returns["transaction_return"] == 0).mean()),
        "guide_source_available_zero_return_fraction": float((paired["guide_return"].dropna() == 0).mean()),
        "transaction_source_available_zero_return_fraction": float((paired["transaction_return"].dropna() == 0).mean()),
        "guide_return_sd": guide_sd,
        "transaction_return_sd": transaction_sd,
        "guide_to_transaction_return_sd_ratio": guide_sd / transaction_sd if transaction_sd > 0 else np.nan,
        "price_abs_relative_discrepancy_median": float(price_discrepancy.median()),
        "price_abs_relative_discrepancy_p95": float(price_discrepancy.quantile(.95)),
        "best_cross_source_lag": best_lag,
        "best_cross_source_correlation": best_corr,
        "spx_reference_comparison_n": len(spx_overlap),
        "spx_reference_absolute_difference_max": float(spx_discrepancy.max()),
        "spx_contemporaneous_n": len(contemporaneous),
        "spx_contemporaneous_start": _dates(contemporaneous)[0],
        "spx_contemporaneous_end": _dates(contemporaneous)[1],
        "spx_contemporaneous_guide_correlation": _correlation(contemporaneous, "guide_return", "sp500_return"),
        "spx_contemporaneous_transaction_correlation": _correlation(contemporaneous, "transaction_return", "sp500_return"),
        "spx_inherited_lag89_n": len(inherited),
        "spx_inherited_lag89_start": _dates(inherited)[0],
        "spx_inherited_lag89_end": _dates(inherited)[1],
        "spx_inherited_lag89_guide_correlation": _correlation(inherited, "guide_return_lag89", "sp500_return"),
        "spx_inherited_lag89_transaction_correlation": _correlation(inherited, "transaction_return_lag89", "sp500_return"),
    }
    if paired.empty:
        status = "not_estimable: no overlapping equity sessions"
    elif returns.empty:
        status = "not_estimable: no paired complete returns"
    elif best_lag is None:
        status = "partial: no cross-source lag has 30 complete nonconstant pairs"
    else:
        status = "ok: descriptive exploratory comparison; inherited lag89 is exposed"
    summary["status"] = status
    return {
        "paired": paired,
        "summary": pd.DataFrame(summary.items(), columns=["metric", "value"]),
        "lags": lags,
        "best_lag": best_lag,
        "best_corr": best_corr,
        "calendar": calendar,
        "status": status,
    }
