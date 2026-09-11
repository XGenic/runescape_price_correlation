"""Separate, immutable GE-guide-price archive; never splice into RuneLite averages."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import requests

from bond_sp500_correlation import USER_AGENT, fetch_sp500
from bond_investigation_data import BOND_ITEM_ID, align_prices, cached_frame, daily_index, prepare_controls

ARCHIVE_URL = "https://api.weirdgloop.org/exchange/history/osrs/all"
# Instrument metadata, NOT an arbitrary sample cutoff: bonds launched on this day.
BOND_LAUNCH = pd.Timestamp("2015-03-30", tz="UTC")
PROTOCOL = "ge-guide-13190-launch-exclusion-last-utc-quote-next-day-no-fill-v1"


def fetch_archive(cache_dir):
    """Preserve the original JSON response, then expose all raw records as a frame.

    Current API returns objects with millisecond Unix timestamps, unlike some
    older documentation's array schema. An unexpected schema is an error, not a
    guessed conversion. This archive contains GE updates, not RuneLite trades.
    """
    path = Path(cache_dir) / "archive_raw_osrs_13190.json"
    if path.exists():
        payload = json.loads(path.read_text())
    else:
        response = requests.get(ARCHIVE_URL, params={"id": BOND_ITEM_ID},
                                headers={"User-Agent": USER_AGENT}, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not payload.get(str(BOND_ITEM_ID)):
            raise ValueError("Archive returned no bond records; refusing to cache an error.")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload) + "\n")
        temporary.replace(path)
    rows = payload.get(str(BOND_ITEM_ID))
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Unexpected archive schema: expected nonempty JSON object records.")
    return pd.DataFrame(rows)


def audit_archive(raw):
    """Audit records before selecting the final quote of each post-launch UTC day.

    Never fill missing guide dates. Same-day updates are collapsed to the last
    timestamp, not averaged. Pre-launch entries remain available in the excluded
    table; their initial values must not be treated as observed bond trading.
    """
    required = {"id", "timestamp", "price", "volume"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Archive missing fields: {sorted(required - set(raw.columns))}")
    if not raw.id.astype(str).eq(str(BOND_ITEM_ID)).all():
        raise ValueError("Archive contains a different item ID.")
    quotes = raw.copy()
    quotes["timestamp_utc"] = pd.to_datetime(quotes.timestamp, unit="ms", utc=True, errors="raise")
    quotes["price"] = pd.to_numeric(quotes.price, errors="raise")
    if quotes.timestamp_utc.isna().any() or not np.isfinite(quotes.price).all() or (quotes.price <= 0).any():
        raise ValueError("Invalid timestamp or nonpositive/nonfinite guide price.")
    if (quotes.groupby("timestamp_utc").price.nunique() > 1).any():
        raise ValueError("Conflicting prices at the same timestamp; manual audit required.")
    quotes = quotes.sort_values("timestamp_utc", kind="stable")
    excluded = quotes.loc[quotes.timestamp_utc < BOND_LAUNCH].copy()
    excluded["reason"] = "Dated before documented OSRS bond launch; excluded, not imputed"
    valid = quotes.loc[quotes.timestamp_utc >= BOND_LAUNCH].copy()
    if valid.empty:
        raise ValueError("No valid post-launch guide observations.")
    valid["date"] = valid.timestamp_utc.dt.tz_localize(None).dt.normalize()
    duplicate_days = valid.groupby("date").size()
    daily = valid.drop_duplicates("date", keep="last").set_index("date")[["price"]]
    daily.index.name = "date"
    gap_rows = []
    for before, after in zip(daily.index[:-1], daily.index[1:]):
        missing = (after - before).days - 1
        if missing > 0:
            gap_rows.append({"last_observed": before, "next_observed": after,
                             "first_missing": before + pd.Timedelta(days=1),
                             "last_missing": after - pd.Timedelta(days=1), "missing_days": missing})
    gaps = pd.DataFrame(gap_rows, columns=["last_observed", "next_observed", "first_missing", "last_missing", "missing_days"])
    years = valid.assign(year=valid.date.dt.year).groupby("year").agg(
        observations=("price", "size"), observed_days=("date", "nunique"),
        first_date=("date", "min"), last_date=("date", "max"))
    years["calendar_span_days"] = (years.last_date - years.first_date).dt.days + 1
    years["missing_days_within_year_span"] = years.calendar_span_days - years.observed_days
    audit = {
        "item_id": BOND_ITEM_ID, "source": f"{ARCHIVE_URL}?id={BOND_ITEM_ID}",
        "price_definition": "Archived GE guide-price updates, not RuneLite transaction averages",
        "raw_records": len(raw), "raw_unique_days": quotes.timestamp_utc.dt.normalize().nunique(),
        "raw_start": quotes.timestamp_utc.min().isoformat(), "raw_end": quotes.timestamp_utc.max().isoformat(),
        "prelaunch_excluded": len(excluded), "postlaunch_records": len(valid),
        "postlaunch_daily_quotes": len(daily),
        "postlaunch_start": str(daily.index.min().date()), "postlaunch_end": str(daily.index.max().date()),
        "duplicate_timestamp_records": int(quotes.timestamp_utc.duplicated().sum()),
        "multiple_update_days": int((duplicate_days > 1).sum()),
        "extra_same_day_updates_collapsed": len(valid) - len(daily),
        "missing_calendar_days": int(gaps.missing_days.sum()),
        "largest_gap_days": int(gaps.missing_days.max()) if not gaps.empty else 0,
        "null_volume_records": int(raw.volume.isna().sum()),
        "midnight_timestamp_fraction": float(quotes.timestamp_utc.eq(quotes.timestamp_utc.dt.normalize()).mean()),
        "availability_policy": "Last quote dated D available at D+1 UTC midnight; positive lag >=1 only",
        "timestamp_limitation": "Reported GE update times, not verified first-publication or collection times",
    }
    return daily, audit, gaps, excluded, years


def load_historical_context(cache_dir=Path("cache/historical")):
    """Rebuild a distinct daily-session research sample from immutable caches.

    The imported alignment function adds one UTC day before matching equities.
    For guide prices this is a CONSERVATIVE availability convention, not a claim
    that the archive timestamps denote the start of transaction-average buckets.
    Missing sessions are retained so every subsequent shift has a fixed horizon.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot = cache_dir / "snapshot_utc.txt"
    if not snapshot.exists():
        snapshot.write_text(pd.Timestamp.now(tz="UTC").isoformat())
    snapshot_day = pd.Timestamp(snapshot.read_text()).tz_convert(None).normalize()
    raw = cached_frame(cache_dir / "archive_raw_osrs_13190.parquet", lambda: fetch_archive(cache_dir))
    guide, audit, gaps, excluded, annual = audit_archive(raw)
    start, end = guide.index.min(), guide.index.max() + pd.Timedelta(days=1)
    key = f"{start.date()}_{end.date()}"
    spx_raw = cached_frame(cache_dir / f"market_GSPC_{key}.parquet",
                           lambda: fetch_sp500("^GSPC", start=start, end=end))
    completed_spx = daily_index(spx_raw)
    completed_spx = completed_spx.loc[completed_spx.index < snapshot_day]
    frame = align_prices(guide, completed_spx)
    if len(frame) < 300:
        raise ValueError("Historical study requires at least 300 equity sessions.")
    control_start = start - pd.DateOffset(months=4)
    control_key = f"{control_start.date()}_{end.date()}"
    vix = cached_frame(cache_dir / f"market_VIX_{control_key}.parquet",
                       lambda: fetch_sp500("^VIX", start=control_start, end=end))
    sentiment = cached_frame(cache_dir / f"fred_UMCSENT_{control_key}.parquet",
                             lambda: pdr.DataReader("UMCSENT", "fred", start=control_start, end=end))
    controls = prepare_controls(frame, vix, sentiment)
    digest = sha256(PROTOCOL.encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    audit.update(equity_sessions=len(frame), complete_return_pairs=len(frame[["bond_return", "sp500_return"]].dropna()),
                 missing_bond_returns=int(frame.bond_return.isna().sum()))
    return {"frame": frame, "controls": controls, "train_end": int(len(frame) * .6),
            "valid_end": int(len(frame) * .8), "fingerprint": digest.hexdigest(),
            "audit": audit, "gaps": gaps, "excluded": excluded, "annual": annual,
            "raw_archive": raw, "guide_daily": guide, "snapshot_day": snapshot_day,
            "evaluation_status": "Retrospective historical evaluation; recent dates overlap the previously exposed study"}
