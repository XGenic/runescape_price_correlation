"""Training-selected directional forecasts and a durable, one-shot holdout ledger.

Rows are trading sessions. A positive lag shifts the bond return BEFORE splitting;
no row is removed before the chronological partition/window boundaries are fixed.
Rows missing either return, and zero SPX target returns, are excluded consistently
from fitting, scoring, and the majority baseline. The gate is calibrated only on
initial training signals. Validation chooses a gate, not a lag or orientation.
OLS (including its sign) is refit only on strictly preceding eligible observations.

Intervals are nominal Wilson binomial intervals, not dependence-adjusted inference.
Rerunning a persisted frozen report is not fresh validation. Local JSON ledgers
must be retained: deleting one does not make previously exposed dates fresh.
"""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
from io import StringIO
import json
import math
import operator
import os
from pathlib import Path
import tempfile

import fcntl
import numpy as np
import pandas as pd


GATE_QUANTILES = (0.0, 0.5, 0.75)
MIN_TRAINING_OBSERVATIONS = 20
DEFAULT_WINDOW = 20
LEDGER_VERSION = 1
_TABLE_KEYS = (
    "validation_candidates", "validation_scored", "test_scored", "window_scores"
)
_REPORT_NOTE = (
    "Frozen one-shot holdout report; loading it again is not fresh validation. "
    "Wilson intervals are nominal and do not account for serial dependence."
)


def _integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean.")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def _wilson(correct: int, count: int) -> list[float | None]:
    if count == 0:
        return [None, None]
    z = 1.959963984540054
    proportion = correct / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / count + z * z / (4 * count * count)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _inputs(frame, train_end, valid_end, lag, window):
    lag = _integer(lag, "lag")
    window = _integer(window, "window")
    train_end = _integer(train_end, "train_end")
    valid_end = _integer(valid_end, "valid_end")
    if lag <= 0:
        raise ValueError("lag must be strictly positive and selected on TRAIN ONLY.")
    if window <= 0:
        raise ValueError("window must be strictly positive.")
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("frame must be a pandas DataFrame.")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("frame must have a chronological DatetimeIndex.")
    if frame.index.hasnans or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("frame dates must be nonmissing, unique, and chronological.")
    if not 0 < train_end < valid_end < len(frame):
        raise ValueError("Require 0 < train_end < valid_end < len(frame).")
    if not {"bond_return", "sp500_return"}.issubset(frame.columns):
        raise ValueError("frame must contain bond_return and sp500_return.")
    # Shift the original session grid, not a dropna-compressed sample.
    x = frame["bond_return"].shift(lag).to_numpy(dtype=float, na_value=np.nan)
    y = frame["sp500_return"].to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(x).any() or np.isinf(y).any():
        raise ValueError("Returns must be finite or missing; infinite returns are invalid.")
    eligible = np.isfinite(x) & np.isfinite(y) & (y != 0)
    training = eligible[:train_end]
    training_x = x[:train_end][training]
    if len(training_x) < MIN_TRAINING_OBSERVATIONS:
        raise ValueError(
            f"Insufficient training for lag {lag}: {len(training_x)} eligible observations; "
            f"at least {MIN_TRAINING_OBSERVATIONS} are required after lagging, missing-value "
            "exclusions, and exclusion of zero SPX returns. No alternate lag was chosen."
        )
    if np.ptp(training_x) == 0:
        raise ValueError("Insufficient training signal variation to fit an intercept and slope.")
    return x, y, eligible, train_end, valid_end, lag, window


def _fit(x, y, eligible, stop):
    history = eligible[:stop]
    hx = x[:stop][history]
    hy = y[:stop][history]
    # Centering avoids an unnecessary design matrix and improves conditioning.
    mean_x = hx.mean()
    mean_y = hy.mean()
    centered = hx - mean_x
    denominator = np.dot(centered, centered)
    if not denominator > 0:
        raise ValueError("Historical bond signal has no estimable variation.")
    slope = np.dot(centered, hy - mean_y) / denominator
    intercept = mean_y - slope * mean_x
    majority = 1 if np.count_nonzero(hy > 0) * 2 >= len(hy) else -1
    return float(intercept), float(slope), majority, len(hy)


def _score_partition(frame, x, y, eligible, start, stop, window, threshold, partition):
    scored_parts = []
    windows = []
    for window_id, left in enumerate(range(start, stop, window), start=1):
        right = min(left + window, stop)
        intercept, slope, majority, history_n = _fit(x, y, eligible, left)
        positions = np.flatnonzero(eligible[left:right]) + left
        fitted = intercept + slope * x[positions]
        if not np.isfinite(fitted).all():
            raise ValueError("OLS produced nonfinite forecasts; no fallback fit was substituted.")
        gated = np.abs(x[positions]) >= threshold
        direction = np.where(fitted > 0, 1, np.where(fitted < 0, -1, majority))
        prediction = np.where(gated, direction, majority)
        target_direction = np.where(y[positions] > 0, 1, -1)
        correct = prediction == target_direction
        baseline_correct = majority == target_direction
        count = len(positions)
        interval = _wilson(int(correct.sum()), count)
        baseline_interval = _wilson(int(baseline_correct.sum()), count)
        scored_parts.append(pd.DataFrame({
            "partition": partition,
            "window_id": window_id,
            "history_n": history_n,
            "signal": x[positions],
            "target_return": y[positions],
            "fitted_return": fitted,
            "signal_gate_passed": gated,
            "prediction_direction": prediction,
            "baseline_direction": majority,
            "target_direction": target_direction,
            "signal_correct": correct,
            "baseline_correct": baseline_correct,
        }, index=frame.index[positions].rename("date")))
        windows.append({
            "partition": partition,
            "window_id": window_id,
            "batch_start": frame.index[left],
            "batch_end": frame.index[right - 1],
            "history_end": frame.index[left - 1],
            "history_n": history_n,
            "n": count,
            "signal_correct": int(correct.sum()),
            "baseline_correct": int(baseline_correct.sum()),
            "accuracy": float(correct.mean()) if count else np.nan,
            "baseline_accuracy": float(baseline_correct.mean()) if count else np.nan,
            "wilson_low": interval[0],
            "wilson_high": interval[1],
            "baseline_wilson_low": baseline_interval[0],
            "baseline_wilson_high": baseline_interval[1],
        })
    scored = pd.concat(scored_parts)
    scored["cumulative_signal_correct"] = scored["signal_correct"].astype(int).cumsum()
    scored["cumulative_baseline_correct"] = scored["baseline_correct"].astype(int).cumsum()
    denominator = np.arange(1, len(scored) + 1)
    scored["cumulative_signal_accuracy"] = scored["cumulative_signal_correct"] / denominator
    scored["cumulative_baseline_accuracy"] = scored["cumulative_baseline_correct"] / denominator
    return scored, pd.DataFrame(windows)


def _summary(scored):
    n = len(scored)
    correct = int(scored["signal_correct"].sum())
    baseline_correct = int(scored["baseline_correct"].sum())
    return {
        "n": n,
        "accuracy": correct / n if n else None,
        "baseline_accuracy": baseline_correct / n if n else None,
        "wilson_95": _wilson(correct, n),
        "baseline_wilson_95": _wilson(baseline_correct, n),
    }


def _prepare_validation(frame, train_end, valid_end, lag, window):
    inputs = _inputs(frame, train_end, valid_end, lag, window)
    x, y, eligible, train_end, valid_end, lag, window = inputs
    train_signal = np.abs(x[:train_end][eligible[:train_end]])
    candidates = []
    best = None
    for quantile in GATE_QUANTILES:
        # q=0 explicitly means no gate, including future signals below the train minimum.
        threshold = 0.0 if quantile == 0 else float(np.quantile(train_signal, quantile))
        scored, windows = _score_partition(
            frame, x, y, eligible, train_end, valid_end, window, threshold, "validation"
        )
        summary = _summary(scored)
        if summary["n"] == 0:
            raise ValueError("Validation has no eligible nonzero target returns; cannot select a gate.")
        candidates.append({
            "quantile": quantile,
            "threshold": threshold,
            "n": summary["n"],
            "accuracy": summary["accuracy"],
            "baseline_accuracy": summary["baseline_accuracy"],
            "wilson_low": summary["wilson_95"][0],
            "wilson_high": summary["wilson_95"][1],
        })
        # The ascending predeclared family and strict comparison implement the tie rule.
        if best is None or summary["accuracy"] > best[0]:
            best = (summary["accuracy"], quantile, threshold, scored, windows)
    _, quantile, threshold, scored, windows = best
    locked = {"lag": lag, "quantile": quantile, "threshold": threshold, "window": window}
    return inputs, pd.DataFrame(candidates), scored, windows, locked


def _finish(frame, prepared):
    inputs, candidates, validation, validation_windows, locked = prepared
    x, y, eligible, _, valid_end, _, window = inputs
    test, test_windows = _score_partition(
        frame, x, y, eligible, valid_end, len(frame), window, locked["threshold"], "test"
    )
    result = {
        "validation_candidates": candidates,
        "validation_scored": validation,
        "test_scored": test,
        "window_scores": pd.concat([validation_windows, test_windows], ignore_index=True),
        "locked_parameters": locked,
        "report_note": _REPORT_NOTE,
        "zero_target_policy": "Exclude from historical fits, majority baseline, and scoring.",
        "fit_policy": "Expanding OLS refits at original-session batch starts; ties predict up.",
        "threshold_policy": "Initial eligible TRAIN absolute signals; quantile zero disables gate.",
    }
    for name, scored in (("validation", validation), ("test", test)):
        result.update({f"{name}_{key}": value for key, value in _summary(scored).items()})
    return result


def walk_forward(frame, train_end, valid_end, lag, window=DEFAULT_WINDOW) -> dict:
    """Score validation gates, freeze their winner, then refit chronologically on test.

    The caller must select the strictly positive lag on TRAIN ONLY. Below-gate
    signals and exactly zero fitted returns use the same expanding-history majority
    baseline (up wins majority ties). Predictions in a batch never use that batch's
    outcomes; earlier test outcomes may enter later test-batch fits. Cumulative
    statistics restart for test. This pure computation has no persistent protection;
    use run_holdout_once for the investigation's protected holdout.
    """
    return _finish(frame, _prepare_validation(frame, train_end, valid_end, lag, window))


def _configuration(lag, train_end, valid_end):
    return {
        "ledger_version": LEDGER_VERSION,
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "lag": lag,
        "window": DEFAULT_WINDOW,
        "gate_quantiles": list(GATE_QUANTILES),
        "minimum_training_observations": MIN_TRAINING_OBSERVATIONS,
        "train_end": train_end,
        "valid_end": valid_end,
        "zero_targets": "excluded_from_fit_baseline_and_score",
        "majority_tie": "up",
        "gate_tie": "smallest_quantile",
    }


def _encode_result(result):
    payload = dict(result)
    for key in _TABLE_KEYS:
        payload[key] = json.loads(result[key].to_json(
            orient="table", date_format="iso", date_unit="ns", double_precision=15
        ))
    return payload


def _decode_result(payload):
    result = dict(payload)
    for key in _TABLE_KEYS:
        result[key] = pd.read_json(StringIO(json.dumps(payload[key])), orient="table")
    return result


def _atomic_json(path, payload):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, allow_nan=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        # Persist the rename, not only the new file's contents.
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _ledger_lock(path):
    # Keep this inode in place: unlinking it would permit concurrent, disjoint locks.
    with path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def run_holdout_once(context, lag, cache_dir=Path("cache")) -> dict:
    """Return one persisted report per data fingerprint, never a new strategy retry.

    Existing identical requests load JSON without fitting or reading test outcomes.
    Changed lag, partitions, window, or module source SHA256 reject a spent holdout.
    A durable reservation is written immediately BEFORE the first test evaluation;
    an interrupted/failed evaluation stays spent, rather than allowing an unnoticed
    second attempt. JSON uses pandas table schemas, not executable pickle payloads.
    The caller must supply a fingerprint of dataset contents/data configuration,
    never of the selected strategy. Cache files are local integrity records, not
    tamper-proof or distributed experiment tracking.
    """
    lag = _integer(lag, "lag")
    if lag <= 0:
        raise ValueError("lag must be strictly positive and selected on TRAIN ONLY.")
    fingerprint = context.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("context fingerprint must be a nonempty dataset fingerprint string.")
    train_end = _integer(context["train_end"], "train_end")
    valid_end = _integer(context["valid_end"], "valid_end")
    config = _configuration(lag, train_end, valid_end)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = sha256(fingerprint.encode("utf-8")).hexdigest()
    path = cache_dir / f"holdout_{digest}.json"
    spent_message = (
        "This dataset's holdout is spent. A changed lag/rule/configuration or an "
        "interrupted evaluation cannot be evaluated afresh on these dates; truly "
        "fresh dates and a new dataset fingerprint are required. Do not delete the ledger."
    )
    with _ledger_lock(path.with_suffix(".lock")):
        if path.exists():
            try:
                ledger = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError(f"Unreadable holdout ledger. {spent_message}") from exc
            if not isinstance(ledger, dict):
                raise ValueError(f"Invalid holdout ledger. {spent_message}")
            if (ledger.get("fingerprint") != fingerprint or ledger.get("configuration") != config
                    or ledger.get("status") != "complete"):
                raise ValueError(spent_message)
            try:
                return _decode_result(ledger["result"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid persisted report. {spent_message}") from exc

        frame = context["frame"]
        # No reservation for an invalid training sample or an unselectable validation gate.
        prepared = _prepare_validation(frame, train_end, valid_end, lag, DEFAULT_WINDOW)
        ledger = {
            "fingerprint": fingerprint,
            "configuration": config,
            "selected_lag": lag,
            "locked_parameters": prepared[-1],
            "status": "reserved",
            "report_note": _REPORT_NOTE,
        }
        _atomic_json(path, ledger)
        # Any exception from this point leaves a reserved/spent ledger on disk.
        result = _finish(frame, prepared)
        result["ledger_path"] = str(path)
        result["dataset_fingerprint"] = fingerprint
        result["algorithm_configuration"] = config
        result["persisted_result"] = True
        ledger["result"] = _encode_result(result)
        ledger["status"] = "complete"
        _atomic_json(path, ledger)
        # Use the same decoder on cold and warm paths (including scalar/table dtypes).
        return _decode_result(ledger["result"])
