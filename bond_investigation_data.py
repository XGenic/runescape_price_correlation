"""Cache the existing fetchers' outputs and prepare availability-aware sessions.

The v1 timeseries API is a rolling snapshot, not a paginated historical archive.
Raw caches are immutable by default: choose a new cache directory to acquire a
new snapshot. No synthetic prices or forward-filled market returns are used.
"""
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr

from bond_sp500_correlation import fetch_rs_timeseries, fetch_sp500

BOND_ITEM_ID = 13190
DATA_PROTOCOL = "osrs-13190-24h-available-next-utc-day-sentiment-two-month-delay-v1"


def cached_frame(path, fetch):
    """Load an existing parquet verbatim; only a cache miss invokes fetch."""
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    frame = fetch()
    if frame.empty:
        raise ValueError(f"No observations for {path.name}; an empty response is not cached.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary)
    temporary.replace(path)
    return frame


def daily_index(frame):
    """Normalize provider date labels without moving local market calendar dates."""
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    if frame.index.has_duplicates:
        raise ValueError("Duplicate daily observations; check the provider's interval.")
    return frame.sort_index().rename_axis("date")


def align_prices(rs_raw, sp500_raw):
    """Use completed RS daily buckets; retain equity sessions before differencing.

    A wiki timestamp labels the START of an averaging bucket. Its 24h average
    cannot be used until the following UTC midnight. Positive lag 1 is therefore
    available before the beginning of the predicted equity close-to-close period.
    Weekend bond movement is incorporated between consecutive equity sessions.
    Missing RS sessions and the returns across them remain missing; dropping
    them first would turn a one-session lag into an irregular-time lag.
    """
    bond = daily_index(rs_raw).rename(columns={"price": "bond_price"})
    bond.index = bond.index + pd.Timedelta(days=1)
    spx = daily_index(sp500_raw)
    start = max(bond.index.min(), spx.index.min())
    end = min(bond.index.max(), spx.index.max())
    frame = spx.loc[start:end, ["spx_price"]].join(bond[["bond_price"]])
    if frame.empty:
        raise ValueError("No overlapping market sessions between RS and S&P data.")
    for column in ("bond_price", "spx_price"):
        values = frame[column].dropna()
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"Nonpositive or nonfinite observations in {column}.")
    frame["bond_return"] = frame.bond_price.pct_change(fill_method=None)
    frame["sp500_return"] = frame.spx_price.pct_change(fill_method=None)
    return frame.iloc[1:]


def prepare_controls(frame, vix_raw, sentiment_raw):
    """Lag VIX one session; map monthly FRED observations by availability date.

    FRED UMCSENT contains revised/latest-vintage values, not point-in-time
    releases. Delaying the monthly change to month-start + two months avoids
    using its period label as a release date, but does NOT remove revision bias.
    Controls are for retrospective training diagnostics, never the holdout rule.
    """
    vix = daily_index(vix_raw).spx_price
    # Difference on the complete fetched exchange calendar, then shift there.
    prior_vix_return = vix.pct_change(fill_method=None).shift(1)
    sentiment = daily_index(sentiment_raw).UMCSENT
    monthly_change = sentiment.pct_change(fill_method=None)
    monthly_change.index = monthly_change.index + pd.DateOffset(months=2)
    # Union first: a release proxy falling on a weekend must not be discarded.
    available = monthly_change.reindex(monthly_change.index.union(frame.index)).sort_index().ffill()
    return pd.DataFrame({
        "vix_return": prior_vix_return.reindex(frame.index),
        "umcsent_change": available.reindex(frame.index),
    }, index=frame.index)


def load_context(cache_dir=Path("cache"), rs_fetch=fetch_rs_timeseries, market_fetch=fetch_sp500):
    """Fetch every raw source once, then reconstruct all sections from parquet.

    Callers may supply the directly imported fetchers (as the notebook does).
    The RS snapshot has no date-range request parameter; the other cache keys
    include provider, symbol, and inclusive requested range. Keep this directory
    unchanged to reproduce an investigation. All partition boundaries precede
    missing-value filtering, lag search, or inspection of holdout statistics.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = cache_dir / "snapshot_utc.txt"
    if not snapshot_path.exists():
        snapshot_path.write_text(pd.Timestamp.now(tz="UTC").isoformat())
    snapshot_day = pd.Timestamp(snapshot_path.read_text()).tz_convert(None).normalize()
    rs_raw = cached_frame(
        cache_dir / f"rs_raw_osrs_{BOND_ITEM_ID}_24h.parquet",
        lambda: rs_fetch(item_id=BOND_ITEM_ID, market="osrs", timestep="24h"),
    )
    rs_dates = daily_index(rs_raw).index
    start, end = rs_dates.min(), rs_dates.max() + pd.Timedelta(days=1)
    range_key = f"{start.date()}_{end.date()}"
    sp500_raw = cached_frame(
        cache_dir / f"market_raw_GSPC_{range_key}.parquet",
        lambda: market_fetch("^GSPC", start=start, end=end),
    )
    # A daily Yahoo row for the acquisition date may still be intraday. Freeze
    # that cutoff with the cache, so tomorrow cannot promote a partial close.
    completed_spx = daily_index(sp500_raw)
    completed_spx = completed_spx.loc[completed_spx.index < snapshot_day]
    frame = align_prices(rs_raw, completed_spx)
    if len(frame) < 100:
        raise ValueError(f"Only {len(frame)} equity sessions: need at least 100 for this investigation.")
    # Warm-up is derived from the data, not a historical cutoff.
    control_start = start - pd.DateOffset(months=4)
    control_key = f"{control_start.date()}_{end.date()}"
    vix_raw = cached_frame(
        cache_dir / f"market_raw_VIX_{control_key}.parquet",
        lambda: market_fetch("^VIX", start=control_start, end=end),
    )
    sentiment_raw = cached_frame(
        cache_dir / f"fred_raw_UMCSENT_{control_key}.parquet",
        lambda: pdr.DataReader("UMCSENT", "fred", start=control_start, end=end),
    )
    controls = prepare_controls(frame, vix_raw, sentiment_raw)
    digest = sha256(DATA_PROTOCOL.encode())
    # The holdout identity intentionally excludes model configuration. Changing
    # the model cannot turn already-viewed dates into an untouched holdout.
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return {
        "frame": frame, "controls": controls,
        "train_end": int(len(frame) * 0.6), "valid_end": int(len(frame) * 0.8),
        "fingerprint": digest.hexdigest(),
        "rs_raw": rs_raw, "sp500_raw": sp500_raw,
        "vix_raw": vix_raw, "sentiment_raw": sentiment_raw,
        "snapshot_day": snapshot_day,
    }
