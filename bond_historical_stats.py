"""Gap-aware historical diagnostics; callers enforce the training/evaluation split.

These are exploratory diagnostics, including on previously exposed recent dates.
Stationarity on a subset of contiguous training blocks never establishes global
stationarity. Only historical_regimes may receive the post-evaluation full frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ruptures.base import BaseCost
from ruptures.exceptions import NotEnoughPoints
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from bond_investigation_stats import lag_analysis, returns_stationary, stationarity_table


_RETURN_COLUMNS = ["bond_return", "sp500_return"]


def _finite_runs(finite: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open runs on the original grid, including singleton runs."""
    boundaries = np.diff(np.r_[False, finite, False].astype(np.int8))
    return list(zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)))


def stationarity_by_block(frame: pd.DataFrame, min_size: int = 60) -> dict:
    """Diagnose each eligible contiguous finite return-pair TRAINING block.

    coverage_fraction is eligible-block observations / all original sessions,
    not the fraction of blocks passing. Prices are tested by the existing helper
    within each return block; missing prices remain explicit infeasible tests.
    Passing means both returns meet the existing joint ADF/KPSS rule. It is a
    local diagnostic, not a stationarity gate for the entire historical sample.
    """
    if not isinstance(min_size, (int, np.integer)) or min_size < 8:
        raise ValueError("min_size must be an integer of at least eight")
    finite = np.isfinite(frame[_RETURN_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    rows, tables = [], []
    covered = 0
    for start, end in _finite_runs(finite):
        eligible = end - start >= min_size
        row = dict(block_start=frame.index[start], block_end=frame.index[end - 1],
                   n_obs=int(end - start), eligible=eligible, passing=False,
                   status="not_tested: block shorter than min_size")
        if eligible:
            table = stationarity_table(frame.iloc[start:end])
            passing = bool(returns_stationary(table))
            table = table.assign(block_start=row["block_start"], block_end=row["block_end"],
                                 n_obs=row["n_obs"], passing=passing)
            tables.append(table)
            covered += end - start
            row.update(passing=passing, status="tested: local return criteria met" if passing
                       else "tested: local return criteria not met or not estimable")
        rows.append(row)
    table_columns = ["series", "test", "statistic", "pvalue", "interpretation", "status",
                     "block_start", "block_end", "n_obs", "passing"]
    blocks = pd.DataFrame(rows, columns=["block_start", "block_end", "n_obs", "eligible", "passing", "status"])
    status = ("diagnosed_eligible_blocks: local results only; global stationarity not established"
              if tables else "no_eligible_blocks: no contiguous finite return-pair block reaches min_size")
    return dict(table=pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=table_columns),
                blocks=blocks, coverage_fraction=float(covered / len(frame)) if len(frame) else 0.0,
                status=status)


def _block_max_nulls(x: np.ndarray, y: np.ndarray, max_lag: int,
                     iterations: int, block_size: int, seed: np.random.SeedSequence) -> np.ndarray:
    """Batched full-family maxima; run-bounded source chunks, fixed gap mask."""
    n = len(x)
    finite = np.isfinite(x)
    positions = np.arange(n)
    lags = np.arange(-max_lag, max_lag + 1)
    masks = np.zeros((len(lags), n))
    equity = np.zeros_like(masks)
    # Rescaling before centering avoids overflow without changing correlations.
    xvalues = x[finite]
    xvalues = xvalues / max(float(np.max(np.abs(xvalues))), np.finfo(float).tiny)
    xvalues = xvalues - xvalues.mean()
    yfinite = np.isfinite(y)
    ynormal = np.zeros(n)
    ynormal[yfinite] = y[yfinite] / max(float(np.max(np.abs(y[yfinite]))), np.finfo(float).tiny)
    ynormal[yfinite] -= ynormal[yfinite].mean()
    chunks = []
    offset = 0
    for start, end in _finite_runs(finite):
        length = end - start
        for first in range(0, length, block_size):
            chunks.append(xvalues[offset + first:offset + min(first + block_size, length)])
        offset += length
    for j, lag in enumerate(lags):
        targets = positions + lag
        inside = (targets >= 0) & (targets < n)
        source, target = positions[inside], targets[inside]
        keep = finite[source] & yfinite[target]
        source, target = source[keep], target[keep]
        masks[j, source] = 1.0
        equity[j, source] = ynormal[target]
    counts = masks.sum(axis=1)
    denominators = np.maximum(counts, 1)
    sum_y = equity.sum(axis=1)
    var_y = (equity * equity).sum(axis=1) - sum_y * sum_y / denominators
    eligible = (counts >= 30) & (var_y > 0)
    rng = np.random.default_rng(seed)
    nulls = np.full(iterations, np.nan)
    for begin in range(0, iterations, 64):
        size = min(64, iterations - begin)
        shuffled = np.zeros((size, n))
        for row in range(size):
            shuffled[row, finite] = np.concatenate([chunks[i] for i in rng.permutation(len(chunks))])
        sum_x = shuffled @ masks.T
        var_x = (shuffled * shuffled) @ masks.T - sum_x * sum_x / denominators
        numerator = shuffled @ equity.T - sum_x * sum_y / denominators
        valid = eligible[None, :] & (var_x > 0)
        correlations = np.full_like(numerator, np.nan)
        np.divide(numerator, np.sqrt(np.maximum(var_x, 0) * np.maximum(var_y, 0)),
                  out=correlations, where=valid)
        correlations = np.clip(correlations, -1, 1)
        maxima = np.max(np.where(np.isfinite(correlations), np.abs(correlations), -np.inf), axis=1)
        maxima[~np.isfinite(maxima)] = np.nan
        nulls[begin:begin + size] = maxima
    return nulls


def dependence_lag_analysis(frame: pd.DataFrame, max_lag: int = 90,
                            iterations: int = 1000, seed: int = 42,
                            block_sizes: tuple[int, ...] = (5, 20)) -> dict:
    """Training-only signed-lag search with IID and block-permutation sensitivity.

    Positive k is corr(bond_return.shift(k), sp500_return). Every null searches
    the same [-max_lag, max_lag] family, requiring 30 finite pairs at each lag.
    Chunks are made independently within contiguous finite bond runs, including
    shorter terminal chunks. Their order is shuffled globally and values placed
    back in the original finite positions. This is a boundary approximation:
    reassignment can split a chunk across fixed gaps and create artificial joins;
    it cannot preserve all serial dependence, volatility clustering or regimes.
    The +1 p-values address this lag search only, not wider model selection.
    """
    sizes = []
    for size in block_sizes:
        if not isinstance(size, (int, np.integer)) or isinstance(size, bool) or size < 1:
            raise ValueError("block_sizes must contain positive integers")
        if int(size) not in sizes:
            sizes.append(int(size))
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    result = lag_analysis(frame, max_lag=max_lag, iterations=iterations, seed=int(seed))
    sensitivity = [dict(method="IID permutation", block_size=1,
                        pvalue=result["shuffle_pvalue"], threshold=result["threshold"])]
    result["block_nulls"] = {}
    result["block_status"] = {}
    result["dependence_limitations"] = (
        "Run-bounded nonoverlapping source chunks; original missing positions fixed. "
        "Boundary approximation can split reassigned chunks at gaps and creates artificial joins; "
        "cannot preserve all serial dependence, heteroskedasticity or regimes. "
        "Exploratory training-only sensitivity, not a time-series-valid significance guarantee.")
    observed = abs(result["best_corr"])
    x, y = (frame[column].to_numpy(dtype=float) for column in _RETURN_COLUMNS)
    for size in sizes:
        # A size-addressed SeedSequence is distinct from the IID seed and stable
        # if the caller changes the order of the requested block sizes.
        child_seed = np.random.SeedSequence([int(seed), size, 19073])
        nulls = np.full(iterations, np.nan)
        pvalue, threshold = np.nan, np.nan
        status = "not_estimable: observed lag search is unavailable"
        if np.isfinite(observed):
            chunks = sum((end - start + size - 1) // size for start, end in _finite_runs(np.isfinite(x)))
            if chunks < 2:
                status = "not_estimable: fewer than two source chunks; no block-order randomization"
            else:
                nulls = _block_max_nulls(x, y, max_lag, iterations, size, child_seed)
                if np.isfinite(nulls).all():
                    pvalue = float((1 + np.count_nonzero(nulls >= observed)) / (iterations + 1))
                    threshold = float(np.quantile(nulls, 0.95))
                    status = "ok: approximate block dependence sensitivity"
                else:
                    status = "not_estimable: at least one permutation has no finite lag statistic"
        result["block_nulls"][size] = nulls
        result["block_status"][size] = status
        sensitivity.append(dict(method="Block permutation", block_size=size, pvalue=pvalue, threshold=threshold))
    result["sensitivity"] = pd.DataFrame(sensitivity, columns=["method", "block_size", "pvalue", "threshold"])
    if any(not value.startswith("ok:") for value in result["block_status"].values()):
        result["status"] += "; block sensitivity partly/not estimable (see block_status)"
    return result


def granger_with_gaps(frame: pd.DataFrame, best_lag: int | None) -> pd.DataFrame:
    """Exploratory TRAINING nested-OLS F-tests with original-grid lags.

    Restricted outcome autoregression and augmented source-lag model use exactly
    the same complete rows after lag construction. Classical F calibration assumes
    correctly specified stationary linear dynamics and iid homoskedastic Gaussian
    errors; gaps and selected rows do not establish these assumptions. Holm covers
    estimable direction/orders only, not preceding lag selection. No causal or
    global-stationarity claim follows from these exploratory postselection tests.
    """
    orders = {1, 5, 10, 30}
    if best_lag is not None:
        if not isinstance(best_lag, (int, np.integer)) or isinstance(best_lag, bool):
            raise ValueError("best_lag must be an integer or None")
        if best_lag:
            orders.add(abs(int(best_lag)))
    returns = frame[_RETURN_COLUMNS].replace([np.inf, -np.inf], np.nan)
    rows = []
    for source, target, label in [("bond_return", "sp500_return", "Bond -> S&P 500"),
                                  ("sp500_return", "bond_return", "S&P 500 -> Bond")]:
        for order in sorted(orders):
            row = dict(direction=label, lag_order=order, n_obs=0, fstat=np.nan,
                       pvalue=np.nan, adjusted_pvalue=np.nan, status="ok")
            try:
                if order >= len(returns):
                    raise ValueError("lag order reaches or exceeds original session count")
                design = pd.concat([returns[target], *[returns[target].shift(k) for k in range(1, order + 1)],
                                    *[returns[source].shift(k) for k in range(1, order + 1)]], axis=1)
                values = design.to_numpy(dtype=float)
                values = values[np.isfinite(values).all(axis=1)]
                n = len(values)
                row["n_obs"] = n
                if n <= 2 * order + 1:
                    raise ValueError(f"{n} complete original-grid rows give no augmented residual degrees of freedom")
                # Column rescaling changes neither fitted subspaces nor the F-test.
                scale = np.max(np.abs(values), axis=0)
                values = values / np.where(scale > 0, scale, 1.0)
                outcome = values[:, 0]
                if np.ptp(outcome) == 0:
                    raise ValueError("constant outcome")
                augmented = np.column_stack([np.ones(n), values[:, 1:]])
                restricted = augmented[:, :order + 1]
                if np.linalg.matrix_rank(augmented) != augmented.shape[1]:
                    raise ValueError("rank-deficient augmented intercept/lag design")
                if np.linalg.matrix_rank(restricted) != restricted.shape[1]:
                    raise ValueError("rank-deficient restricted autoregression")
                small = sm.OLS(outcome, restricted).fit()
                large = sm.OLS(outcome, augmented).fit()
                if large.df_resid <= 0 or not np.isfinite([small.ssr, large.ssr]).all():
                    raise ValueError("nonpositive residual degrees of freedom or nonfinite residual SSE")
                if large.ssr <= np.finfo(float).eps * float(outcome @ outcome):
                    raise ValueError("perfect or numerically perfect augmented fit")
                fstat, pvalue, df_difference = large.compare_f_test(small)
                if not np.isfinite([fstat, pvalue, df_difference]).all() or fstat < 0 or not 0 <= pvalue <= 1:
                    raise ValueError("invalid nested-model F-test")
                if df_difference != order:
                    raise ValueError("unexpected nested-model degrees-of-freedom difference")
                row.update(fstat=float(fstat), pvalue=float(pvalue))
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                row["status"] = f"skipped: {error}"
            rows.append(row)
    table = pd.DataFrame(rows, columns=["direction", "lag_order", "n_obs", "fstat", "pvalue", "adjusted_pvalue", "status"])
    estimable = table["status"].eq("ok") & np.isfinite(table["pvalue"])
    if estimable.any():
        table.loc[estimable, "adjusted_pvalue"] = multipletests(table.loc[estimable, "pvalue"], method="holm")[1]
    table.attrs["limitations"] = (
        "Classical stationary, correctly specified, iid homoskedastic Gaussian-error F assumptions are not established. "
        "Holm spans estimable directions/orders but not prior lag selection; exploratory predictive association, not causation.")
    return table


class _PrefixLinearCost(BaseCost):
    """O(1) intercept+slope SSE after O(n) prefix preparation.

    Signal columns are outcome and source (intercept implicit). SSE is measured
    in globally rescaled outcome units, a common factor for every segmentation.
    """

    model = "linear"
    min_size = 2

    def fit(self, signal: np.ndarray) -> _PrefixLinearCost:
        values = np.asarray(signal, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all() or not len(values):
            raise ValueError("linear cost requires a nonempty finite [outcome, source] matrix")
        self.signal = values
        scales = np.maximum(np.max(np.abs(values), axis=0), np.finfo(float).tiny)
        scaled = values / scales
        scaled -= scaled.mean(axis=0)
        y, x = scaled.T
        self.prefix = np.vstack([np.zeros(5), np.cumsum(np.column_stack([x, y, x * x, y * y, x * y]), axis=0)])
        return self

    def _errors(self, starts: np.ndarray, end: int) -> np.ndarray:
        count = end - starts
        sums = self.prefix[end] - self.prefix[starts]
        sx, sy, sxx, syy, sxy = sums.T
        vx = sxx - sx * sx / count
        vy = syy - sy * sy / count
        cov = sxy - sx * sy / count
        tolerance = 64 * np.finfo(float).eps
        identified = vx > tolerance * np.maximum(np.abs(sxx), np.abs(sx * sx / count))
        explained = np.zeros_like(vx)
        np.divide(cov * cov, vx, out=explained, where=identified)
        sse = vy - explained
        credible = sse >= -tolerance * np.maximum(np.abs(syy), np.abs(explained))
        return np.where(identified & credible & np.isfinite(sse), np.maximum(sse, 0), np.inf)

    def error(self, start: int, end: int) -> float:
        if end - start < self.min_size:
            raise NotEnoughPoints
        if start < 0 or end > len(self.signal):
            raise ValueError("segment bounds outside fitted signal")
        return float(self._errors(np.array([start]), end)[0])


def historical_regimes(frame: pd.DataFrame) -> dict:
    """Full-range, post-historical-evaluation exploratory BIC segmentation.

    Rolling correlations retain the original grid and require 60/90 complete
    sessions. Regression segmentation alone drops incomplete return pairs; 126
    means retained observations, NOT calendar days, so segments may span gaps.
    Candidate internal boundaries are every five retained observations, with at
    most five breaks and an explicit zero-break candidate. Prefix-sum SSE and
    dynamic programming minimize regression SSE separately for each break count.
    BIC is n*log(SSE/n) + multiplier*[2*(breaks+1)+breaks]*log(n).

    This is not formal Bai-Perron significance testing. Changes in residual
    variance, outliers or gap selection may drive apparent slope/intercept breaks;
    it does not separate those explanations or prove changing correlation.
    """
    returns = frame[_RETURN_COLUMNS].replace([np.inf, -np.inf], np.nan)
    rolling = pd.DataFrame(index=frame.index)
    for window in (60, 90):
        rolling[f"corr_{window}"] = returns["bond_return"].rolling(window, min_periods=window).corr(returns["sp500_return"])
    rolling = rolling.replace([np.inf, -np.inf], np.nan)
    finite = np.isfinite(returns.to_numpy(dtype=float)).all(axis=1)
    clean = returns.loc[finite]
    n = len(clean)
    result = dict(rolling=rolling, break_dates=[],
                  sensitivity=pd.DataFrame(columns=["penalty_multiplier", "break_count"]),
                  segments=pd.DataFrame(columns=["start", "end", "n_obs", "correlation"]),
                  status="not_estimable: requires at least 126 finite return pairs",
                  dropped_nonfinite=int((~finite).sum()), n_obs=n,
                  method="Exploratory BIC intercept+slope regression segmentation; not formal Bai-Perron significance testing",
                  limitations="Segmentation removes gaps; minimum 126 retained pairs, not calendar days. "
                              "Five-pair boundary grid; slope/intercept breaks can reflect variance changes or outliers, "
                              "not necessarily correlation shifts. Full-range post-evaluation analysis is exploratory.")
    if n < 126:
        return result
    signal = clean[["sp500_return", "bond_return"]].to_numpy(dtype=float)
    if np.ptp(signal[:, 1]) == 0 or np.ptp(signal[:, 0]) == 0:
        result["status"] = "not_estimable: constant source or outcome returns"
        return result
    cost = _PrefixLinearCost().fit(signal)
    endpoints = np.r_[np.arange(0, n, 5, dtype=int), n]
    max_segments = min(6, n // 126)
    scores = np.full((max_segments + 1, len(endpoints)), np.inf)
    previous = np.full_like(scores, -1, dtype=int)
    scores[0, 0] = 0.0
    # Each candidate end computes its prefix costs just once for all break counts.
    for j in range(1, len(endpoints)):
        eligible = np.flatnonzero(endpoints[j] - endpoints[:j] >= 126)
        if not len(eligible):
            continue
        costs = cost._errors(endpoints[eligible], int(endpoints[j]))
        for segments in range(1, max_segments + 1):
            totals = scores[segments - 1, eligible] + costs
            best = int(np.argmin(totals))
            if np.isfinite(totals[best]):
                scores[segments, j] = totals[best]
                previous[segments, j] = eligible[best]
    candidates, failures = [], []
    for segments in range(1, max_segments + 1):
        sse = scores[segments, -1]
        if not np.isfinite(sse):
            failures.append(f"{segments - 1} breaks: no full-rank segmentation satisfying grid/minimum size")
            continue
        j, cuts = len(endpoints) - 1, []
        for count in range(segments, 0, -1):
            cuts.append(int(endpoints[j]))
            j = int(previous[count, j])
        cuts.reverse()
        candidates.append(dict(break_count=segments - 1, endpoints=cuts, sse=float(sse),
                               likelihood=float(n * np.log(max(sse / n, np.finfo(float).tiny))),
                               penalty=float((2 * segments + segments - 1) * np.log(n))))
    result["candidate_failures"] = failures
    if not candidates:
        result["status"] = "not_estimable: no numerically identified finite segmentation"
        return result
    selected_by_penalty = [(multiplier, min(candidates, key=lambda item: item["likelihood"] + multiplier * item["penalty"]))
                           for multiplier in (0.5, 1.0, 2.0)]
    selected = selected_by_penalty[1][1]
    rows = []
    cuts = selected["endpoints"]
    for start, end in zip([0, *cuts[:-1]], cuts):
        section = clean.iloc[start:end]
        x, y = section[_RETURN_COLUMNS].to_numpy(dtype=float).T
        corr = float(stats.pearsonr(x, y).statistic) if np.ptp(x) > 0 and np.ptp(y) > 0 else np.nan
        rows.append(dict(start=section.index[0], end=section.index[-1], n_obs=end - start, correlation=corr))
    result.update(break_dates=[pd.Timestamp(clean.index[end]) for end in cuts[:-1]],
                  sensitivity=pd.DataFrame([dict(penalty_multiplier=multiplier, break_count=item["break_count"])
                                            for multiplier, item in selected_by_penalty]),
                  segments=pd.DataFrame(rows), status="ok: exploratory post-evaluation segmentation",
                  bic_candidates=pd.DataFrame([dict(break_count=item["break_count"], scaled_sse=item["sse"],
                                                    bic=item["likelihood"] + item["penalty"]) for item in candidates]))
    return result
