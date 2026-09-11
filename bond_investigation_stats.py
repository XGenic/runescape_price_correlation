"""Training-sample diagnostics and post-holdout exploratory regime analysis.

Returns are fractional close-to-close changes, not percentage points. A positive
lag means bond_return.shift(lag) is paired with sp500_return: the bond observation
precedes the equity return by that many shared trading sessions. Callers enforce
the training/holdout boundary; these functions never fetch data or select a split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ruptures as rpt
from arch.unitroot import ADF, KPSS
from arch.utility.exceptions import InfeasibleTestException
from ruptures.costs import CostLinear
from ruptures.exceptions import BadSegmentationParameters
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import InfeasibleTestError
from statsmodels.tsa.stattools import grangercausalitytests


_RETURN_COLUMNS = ["bond_return", "sp500_return"]
_EXPECTED_ERRORS = (ValueError, np.linalg.LinAlgError, InfeasibleTestException)


def _contiguous_values(frame: pd.DataFrame) -> np.ndarray:
    """Trim missing boundaries, but never silently join nonadjacent sessions."""
    values = frame.to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    positions = np.flatnonzero(finite)
    if not len(positions):
        raise ValueError("no finite observations")
    first, last = positions[0], positions[-1]
    if not finite[first : last + 1].all():
        raise ValueError("interior missing/nonfinite sessions would compress time")
    return values[first : last + 1]


def stationarity_table(frame: pd.DataFrame) -> pd.DataFrame:
    """ADF and KPSS with an intercept, using prices and fractional returns.

ADF's null is a unit root; KPSS's null is level stationarity. Decisions use
numeric p-values at 5%, not test summary strings. Boundary missing values are
trimmed; interior gaps, constant data and infeasible tests get explicit rows with
NaN statistics and a not_estimable status. Tests do not establish stationarity
with certainty, especially in this short training sample.
    """
    rows = []
    for name in ["bond_price", "spx_price", *_RETURN_COLUMNS]:
        for label, implementation in [("ADF", ADF), ("KPSS", KPSS)]:
            row = dict(series=name, test=label, statistic=np.nan, pvalue=np.nan)
            try:
                values = _contiguous_values(frame[[name]])[:, 0]
                if len(values) < 8 or np.ptp(values) == 0:
                    raise ValueError("requires at least eight nonconstant observations")
                result = implementation(values, trend="c")
                statistic, pvalue = float(result.stat), float(result.pvalue)
                if not np.isfinite([statistic, pvalue]).all():
                    raise ValueError("test produced a nonfinite statistic or p-value")
                if label == "ADF":
                    interpretation = (
                        "Reject unit-root null at 5%" if pvalue < 0.05
                        else "Do not reject unit-root null at 5%"
                    )
                else:
                    interpretation = (
                        "Reject level-stationarity null at 5%" if pvalue < 0.05
                        else "Do not reject level-stationarity null at 5%"
                    )
                row.update(statistic=statistic, pvalue=pvalue,
                           interpretation=interpretation, status="ok")
            except _EXPECTED_ERRORS as error:
                row.update(interpretation=f"Not estimable: {error}",
                           status=f"not_estimable: {error}")
            rows.append(row)
    return pd.DataFrame(rows)


def returns_stationary(table: pd.DataFrame) -> bool:
    """Require BOTH return series to have ADF p<.05 and KPSS p>=.05."""
    for name in _RETURN_COLUMNS:
        for test in ["ADF", "KPSS"]:
            selected = table.loc[(table["series"] == name) & (table["test"] == test)]
            if len(selected) != 1:
                return False
            pvalue = selected["pvalue"].iloc[0]
            if not pd.notna(pvalue) or not np.isfinite(float(pvalue)):
                return False
            if not 0 <= float(pvalue) <= 1:
                return False
            if test == "ADF" and not float(pvalue) < 0.05:
                return False
            if test == "KPSS" and not float(pvalue) >= 0.05:
                return False
    return True


def lag_analysis(frame: pd.DataFrame, max_lag: int = 90,
                 iterations: int = 1000, seed: int = 42) -> dict:
    """Scan signed trading-session lags and calibrate the whole lag search.

Positive lag is corr(bond_return.shift(lag), sp500_return). At least 30 finite
pairs are required for each lag; missing positions are retained before shifting.
The best lag maximizes absolute Pearson correlation; forecast_lag is selected
separately from strictly positive lags. Pointwise Pearson p-values are nominal.

The null permutes finite bond returns IID, keeping their missing positions fixed,
and takes the maximum absolute correlation across the SAME full signed lag
family on each iteration. This corrects the lag search under exchangeability,
NOT serial dependence or heteroskedasticity; it is an IID sensitivity diagnostic,
not a time-series-valid proof. Its p-value uses the finite-sample +1 correction.
The caller, not this numerical function, applies the stationarity gate.
    """
    if not isinstance(max_lag, (int, np.integer)) or max_lag < 0:
        raise ValueError("max_lag must be a nonnegative integer")
    if not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    x = frame["bond_return"].to_numpy(dtype=float)
    y = frame["sp500_return"].to_numpy(dtype=float)
    n = len(frame)
    lags = np.arange(-max_lag, max_lag + 1)
    positions = np.arange(n)
    xfinite = np.isfinite(x)
    # All arrays are expressed at the unshifted bond position i; equity is i+lag.
    masks = np.zeros((len(lags), n), dtype=float)
    equity = np.zeros_like(masks)
    rows = []
    for j, lag in enumerate(lags):
        targets = positions + lag
        within = (targets >= 0) & (targets < n)
        source = positions[within]
        target = targets[within]
        keep = xfinite[source] & np.isfinite(y[target])
        source, target = source[keep], target[keep]
        masks[j, source] = 1.0
        equity[j, source] = y[target]
        correlation, pvalue = np.nan, np.nan
        if len(source) >= 30 and np.ptp(x[source]) > 0 and np.ptp(y[target]) > 0:
            result = stats.pearsonr(x[source], y[target])
            correlation, pvalue = float(result.statistic), float(result.pvalue)
        rows.append(dict(lag=int(lag), correlation=correlation,
                         pvalue=pvalue, n_obs=len(source)))
    table = pd.DataFrame(rows)
    result = dict(table=table, best_lag=None, best_corr=np.nan, forecast_lag=None,
                  null_max=np.full(iterations, np.nan), shuffle_pvalue=np.nan,
                  threshold=np.nan, status="not_estimable: no lag has 30 nonconstant pairs",
                  null_method="IID permutation; serial dependence is not preserved")
    available = table.loc[np.isfinite(table["correlation"])]
    if available.empty:
        return result
    best = available.loc[available["correlation"].abs().idxmax()]
    result.update(best_lag=int(best["lag"]), best_corr=float(best["correlation"]))
    positive = available.loc[available["lag"] > 0]
    if not positive.empty:
        result["forecast_lag"] = int(positive.loc[positive["correlation"].abs().idxmax(), "lag"])

    # Center globally to reduce cancellation. Pair-specific means are still
    # removed below because overlap and missingness vary by lag.
    xvalues = x[xfinite] - x[xfinite].mean()
    ymean = y[np.isfinite(y)].mean()
    equity -= masks * ymean
    counts = masks.sum(axis=1)
    denominators = np.maximum(counts, 1)
    sum_y = equity.sum(axis=1)
    var_y = (equity * equity).sum(axis=1) - sum_y * sum_y / denominators
    eligible = (counts >= 30) & (var_y > 0)
    rng = np.random.default_rng(seed)
    null_max = np.full(iterations, np.nan)
    for begin in range(0, iterations, 64):
        size = min(64, iterations - begin)
        shuffled = np.zeros((size, n))
        for row in range(size):
            shuffled[row, xfinite] = rng.permutation(xvalues)
        sum_x = shuffled @ masks.T
        var_x = (shuffled * shuffled) @ masks.T - sum_x * sum_x / denominators
        numerator = shuffled @ equity.T - sum_x * sum_y / denominators
        valid = eligible[None, :] & (var_x > 0)
        correlations = np.full_like(numerator, np.nan)
        denominator = np.sqrt(np.maximum(var_x, 0) * np.maximum(var_y, 0))
        np.divide(numerator, denominator, out=correlations, where=valid)
        correlations = np.clip(correlations, -1, 1)
        maxima = np.max(np.where(np.isfinite(correlations), np.abs(correlations), -np.inf), axis=1)
        maxima[~np.isfinite(maxima)] = np.nan
        null_max[begin : begin + size] = maxima
    result["null_max"] = null_max
    if not np.isfinite(null_max).all():
        result["status"] = "not_estimable: at least one permutation has no finite lag statistic"
        return result
    observed = abs(result["best_corr"])
    result.update(shuffle_pvalue=float((1 + np.count_nonzero(null_max >= observed)) / (iterations + 1)),
                  threshold=float(np.quantile(null_max, 0.95)), status="ok")
    return result


def controlled_regression(frame: pd.DataFrame, controls: pd.DataFrame, lag: int) -> dict:
    """Compare controls-only and bond-augmented OLS on identical finite rows.

Bond returns are shifted by a strictly positive number of shared trading
sessions; controls must ALREADY be availability-adjusted by the data layer.
HAC covariance uses fixed maxlags=5. Fractional-return coefficients and nominal
post-lag-selection p-values are exploratory, not confirmatory causal inference.
Missing rows are excluded AFTER shifting. Singular, constant and insufficient
samples return empty coefficients and an explicit not_estimable status.
    """
    result = dict(coefficients=pd.DataFrame(columns=["coefficient", "pvalue", "tstat"]),
                  rsquared=np.nan, baseline_rsquared=np.nan, bond_pvalue=np.nan,
                  n_obs=0, status="not_estimable")
    if not isinstance(lag, (int, np.integer)) or lag <= 0:
        result["status"] = "not_estimable: lag must be a strictly positive integer"
        return result
    required = ["vix_return", "umcsent_change"]
    missing = [name for name in required if name not in controls.columns]
    if missing:
        result["status"] = f"not_estimable: missing controls {missing}"
        return result
    sample = pd.concat([
        frame["sp500_return"], frame["bond_return"].shift(lag).rename("bond_lag"),
        controls.reindex(frame.index)[required],
    ], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    result["n_obs"] = len(sample)
    if sample.empty:
        result["status"] = "not_estimable: no common finite observations"
        return result
    baseline = sm.add_constant(sample[required], has_constant="add")
    augmented = sm.add_constant(sample[["bond_lag", *required]], has_constant="add")
    if len(sample) <= augmented.shape[1] + 5:
        result["status"] = "not_estimable: residual degrees of freedom must exceed HAC lag 5"
        return result
    if np.ptp(sample["sp500_return"].to_numpy()) == 0:
        result["status"] = "not_estimable: constant dependent return"
        return result
    if (np.linalg.matrix_rank(baseline.to_numpy()) < baseline.shape[1]
            or np.linalg.matrix_rank(augmented.to_numpy()) < augmented.shape[1]):
        result["status"] = "not_estimable: constant or rank-deficient regressors"
        return result
    try:
        base_fit = sm.OLS(sample["sp500_return"], baseline).fit(
            cov_type="HAC", cov_kwds={"maxlags": 5})
        fit = sm.OLS(sample["sp500_return"], augmented).fit(
            cov_type="HAC", cov_kwds={"maxlags": 5})
        coefficients = pd.DataFrame({"coefficient": fit.params,
                                     "pvalue": fit.pvalues, "tstat": fit.tvalues})
        if not np.isfinite(coefficients.to_numpy()).all():
            raise ValueError("nonfinite fitted coefficients or HAC inference")
        coefficients.index.name = "term"
        result.update(coefficients=coefficients, rsquared=float(fit.rsquared),
                      baseline_rsquared=float(base_fit.rsquared),
                      bond_pvalue=float(fit.pvalues["bond_lag"]), status="ok")
    except (ValueError, np.linalg.LinAlgError) as error:
        result["status"] = f"not_estimable: {error}"
    return result


def granger_table(frame: pd.DataFrame, best_lag: int | None) -> pd.DataFrame:
    """Bidirectional predictive Granger F-tests, not evidence of causation.

Only orders 1, 5, 10, 30 and positive abs(best_lag) are requested, with duplicates
removed. An order is skipped unless n > 3*order+1 and residual degrees of freedom
are positive. Interior gaps are not compressed. Holm correction spans every
estimable direction/order in this table; it does not undo prior lag selection.
    """
    orders = {1, 5, 10, 30}
    if best_lag is not None:
        if not isinstance(best_lag, (int, np.integer)):
            raise ValueError("best_lag must be an integer or None")
        if abs(best_lag) > 0:
            orders.add(abs(int(best_lag)))
    rows = []
    for source, target in [("bond_return", "sp500_return"), ("sp500_return", "bond_return")]:
        try:
            values = _contiguous_values(frame[[target, source]])
            unavailable = None
        except ValueError as error:
            values = np.empty((0, 2))
            unavailable = str(error)
        for order in sorted(orders):
            row = dict(direction=f"{source} -> {target}", lag_order=order,
                       fstat=np.nan, pvalue=np.nan, adjusted_pvalue=np.nan, status="ok")
            try:
                if unavailable:
                    raise ValueError(unavailable)
                residual_df = len(values) - order - (2 * order + 1)
                if len(values) <= 3 * order + 1 or residual_df <= 0:
                    raise ValueError(f"n={len(values)} must exceed 3*lag+1={3 * order + 1}")
                tests = grangercausalitytests(values, maxlag=[order])
                fstat, pvalue, df_denom, _ = tests[order][0]["ssr_ftest"]
                if df_denom <= 0 or not np.isfinite([fstat, pvalue]).all():
                    raise ValueError("nonpositive residual degrees of freedom or nonfinite F-test")
                row.update(fstat=float(fstat), pvalue=float(pvalue))
            except (InfeasibleTestError, ValueError, np.linalg.LinAlgError) as error:
                row["status"] = f"skipped: {error}"
            rows.append(row)
    table = pd.DataFrame(rows)
    estimable = table["status"].eq("ok") & np.isfinite(table["pvalue"])
    if estimable.any():
        table.loc[estimable, "adjusted_pvalue"] = multipletests(
            table.loc[estimable, "pvalue"].to_numpy(), method="holm")[1]
    return table


class _FullRankLinearCost(CostLinear):
    """CostLinear with explicit residual SSE and inadmissible singular segments.

NumPy's lstsq residuals array can be empty for rank-deficient designs; blindly
summing it would give a spurious zero-cost segment. Dynp still uses linear
regression cost, but singular intercept+slope segments are not candidates.
    """

    def error(self, start: int, end: int) -> float:
        design = self.covar[start:end]
        outcome = self.signal[start:end]
        beta, _, rank, _ = np.linalg.lstsq(design, outcome, rcond=None)
        if rank < design.shape[1]:
            return float("inf")
        residual = outcome - design @ beta
        return float(np.sum(residual * residual))


def regime_analysis(frame: pd.DataFrame) -> dict:
    """Exploratory multiple-break regression segmentation, NOT Bai-Perron testing.

Rolling 60/90-session correlations retain the original index and missing rows,
requiring a complete window. Segmentation uses finite return pairs, maps every
boundary back to the original date, and reports removed nonfinite pairs. Its
minimum segment size is 60 retained observations (not calendar days).

Dynp/CostLinear fits SPX fractional return ~ intercept + bond fractional return.
Candidates include zero breaks and at most five feasible breaks. The criterion
is n*log(SSE/n) + [2*(k+1)+k]*log(n), with sensitivity multipliers .5, 1 and 2 on
the parameter penalty. Breaks are never forced. No selected break has a formal
significance level; this is an exploratory full-range, post-holdout diagnostic.
    """
    returns = frame[_RETURN_COLUMNS].replace([np.inf, -np.inf], np.nan)
    rolling = pd.DataFrame(index=frame.index)
    for window in [60, 90]:
        rolling[f"corr_{window}"] = returns["bond_return"].rolling(
            window, min_periods=window).corr(returns["sp500_return"])
    finite = np.isfinite(returns.to_numpy(dtype=float)).all(axis=1)
    clean = returns.loc[finite]
    n = len(clean)
    result = dict(rolling=rolling, break_dates=[],
                  sensitivity=pd.DataFrame(columns=["penalty_multiplier", "break_count"]),
                  segments=pd.DataFrame(columns=["start", "end", "n_obs", "correlation", "status"]),
                  status="not_estimable: requires at least 60 finite return pairs",
                  n_obs=n, dropped_nonfinite=int((~finite).sum()),
                  method="Exploratory BIC multiple-break linear regression; not formal Bai-Perron significance testing")
    if n < 60:
        return result
    signal = np.column_stack([clean["sp500_return"], np.ones(n), clean["bond_return"]])
    if np.linalg.matrix_rank(signal[:, 1:]) < 2:
        result["status"] = "not_estimable: constant bond returns make slope unidentified"
        return result
    if np.ptp(clean["sp500_return"].to_numpy()) == 0:
        result["status"] = "not_estimable: constant equity returns have no regime variation"
        return result
    cost = _FullRankLinearCost()
    model = rpt.Dynp(custom_cost=cost, min_size=60, jump=1).fit(signal)
    candidates = []
    failures = []
    for breaks in range(min(5, n // 60 - 1) + 1):
        try:
            endpoints = [n] if breaks == 0 else model.predict(n_bkps=breaks)
            starts = [0, *endpoints[:-1]]
            sse = sum(cost.error(start, end) for start, end in zip(starts, endpoints))
            if not np.isfinite(sse) or sse < 0:
                raise ValueError("candidate contains a singular or nonfinite regression segment")
            likelihood = n * np.log(max(sse / n, np.finfo(float).tiny))
            penalty = (2 * (breaks + 1) + breaks) * np.log(n)
            candidates.append(dict(break_count=breaks, endpoints=endpoints, sse=sse,
                                   likelihood=float(likelihood), penalty=float(penalty)))
        except (BadSegmentationParameters, ValueError, np.linalg.LinAlgError) as error:
            failures.append(f"{breaks} breaks: {error}")
    if not candidates:
        result["status"] = "not_estimable: no finite candidate segmentation; " + "; ".join(failures)
        return result
    sensitivity = []
    for multiplier in [0.5, 1.0, 2.0]:
        selected = min(candidates, key=lambda item: item["likelihood"] + multiplier * item["penalty"])
        sensitivity.append(dict(penalty_multiplier=multiplier, break_count=selected["break_count"]))
    selected = min(candidates, key=lambda item: item["likelihood"] + item["penalty"])
    endpoints = selected["endpoints"]
    segment_rows = []
    for start, end in zip([0, *endpoints[:-1]], endpoints):
        section = clean.iloc[start:end]
        x = section["bond_return"].to_numpy()
        y = section["sp500_return"].to_numpy()
        correlation = float(stats.pearsonr(x, y).statistic) if np.ptp(x) > 0 and np.ptp(y) > 0 else np.nan
        segment_rows.append(dict(start=section.index[0], end=section.index[-1],
                                 n_obs=len(section), correlation=correlation,
                                 status="ok" if np.isfinite(correlation) else "not_estimable: constant return"))
    result.update(break_dates=[pd.Timestamp(clean.index[end]) for end in endpoints[:-1]],
                  sensitivity=pd.DataFrame(sensitivity), segments=pd.DataFrame(segment_rows),
                  status="ok", candidate_failures=failures,
                  bic_candidates=pd.DataFrame([
                      {"break_count": item["break_count"], "sse": item["sse"],
                       "bic": item["likelihood"] + item["penalty"]} for item in candidates]))
    return result
