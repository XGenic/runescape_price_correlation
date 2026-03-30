#!/usr/bin/env python3
"""Runescape bond vs S&P 500 correlation analysis."""

import argparse
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

API_BASE = "https://prices.runescape.wiki/api/v1"
USER_AGENT = "runescape_price_correlation/0.1 (by github.com)"


def fetch_rs_timeseries(item_id: int, market: str = "osrs", timestep: str = "24h") -> pd.DataFrame:
    url = f"{API_BASE}/{market}/timeseries"
    params = {
        "id": item_id,
        "timestep": timestep,
    }
    response = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
    response.raise_for_status()
    payload = response.json()

    rows = []
    for point in payload.get("data", []):
        ts = pd.to_datetime(point["timestamp"], unit="s", utc=True)
        prices = [point.get("avgHighPrice"), point.get("avgLowPrice")]
        prices = [p for p in prices if p is not None]
        if not prices:
            continue
        price = float(np.mean(prices))
        rows.append({"timestamp": ts, "price": price})

    if not rows:
        raise ValueError("No RuneScape price data returned from the API.")

    df = pd.DataFrame(rows).set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df


def fetch_sp500(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    # yfinance end date is exclusive; add one day to include the final row.
    history = yf.download(symbol, start=start.date(), end=(end + timedelta(days=1)).date(), progress=False)  # type: ignore[assignment]
    if history is None or history.empty:
        raise ValueError(f"No data returned for ticker {symbol}.")

    if isinstance(history.columns, pd.MultiIndex):
        if "Ticker" in history.columns.names:
            history.columns = history.columns.droplevel("Ticker")
        else:
            history.columns = history.columns.droplevel(-1)

    if "Adj Close" in history.columns:
        price_col = "Adj Close"
    elif "Close" in history.columns:
        price_col = "Close"
    else:
        raise ValueError(f"Ticker {symbol} did not return a usable close price column.")
    history = history.rename(columns={price_col: "spx_price"})
    return history[["spx_price"]]


def normalize(series: pd.Series) -> pd.Series:
    return (series - series.min()) / (series.max() - series.min())


def compute_cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int = 70):
    lags = np.arange(-max_lag, max_lag + 1)
    correlations = []
    pvalues = []

    for lag in lags:
        if lag > 0:
            x_slice = x[:-lag]
            y_slice = y[lag:]
        elif lag < 0:
            x_slice = x[-lag:]
            y_slice = y[:lag]
        else:
            x_slice = x
            y_slice = y

        if len(x_slice) < 2 or len(y_slice) < 2:
            correlations.append(np.nan)
            pvalues.append(np.nan)
            continue

        corr, pval = stats.pearsonr(x_slice, y_slice)
        correlations.append(corr)
        pvalues.append(pval)

    result = pd.DataFrame({"lag": lags, "correlation": correlations, "pvalue": pvalues})
    return result


def plot_results(output_path: str, joint: pd.DataFrame, corr_df: pd.DataFrame, best_lag: int):
    sns.set(style="whitegrid")
    normalized = joint.copy()
    normalized["bond_norm"] = normalize(normalized["bond_price"])
    normalized["spx_norm"] = normalize(normalized["spx_price"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 18), constrained_layout=True)

    axes[0].plot(normalized.index, normalized["bond_norm"], label="RuneScape bond (normalized)")
    axes[0].plot(normalized.index, normalized["spx_norm"], label="S&P 500 (normalized)")
    axes[0].set_title("Normalized Price Series")
    axes[0].legend(loc="upper left")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Normalized value")

    axes[1].bar(corr_df["lag"], corr_df["correlation"], color="tab:blue", alpha=0.75)
    axes[1].axvline(best_lag, color="tab:red", linestyle="--", linewidth=2, label=f"Best lag = {best_lag}")
    axes[1].set_title("Lagged correlation between RuneScape bond returns and S&P 500 returns")
    axes[1].set_xlabel("Lag (days): positive = bond leads SP500")
    axes[1].set_ylabel("Pearson correlation")
    axes[1].legend()

    rolling_window = 30
    joint["bond_vol"] = joint["bond_return"].rolling(rolling_window).std()
    joint["spx_vol"] = joint["spx_return"].rolling(rolling_window).std()
    axes[2].plot(joint.index, joint["bond_vol"], label=f"RuneScape bond {rolling_window}-day vol")
    axes[2].plot(joint.index, joint["spx_vol"], label=f"S&P 500 {rolling_window}-day vol")
    axes[2].set_title(f"{rolling_window}-Day Rolling Volatility")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Standard deviation of daily returns")
    axes[2].legend()

    plt.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze RuneScape bond prices vs S&P 500.")
    parser.add_argument("--item-id", type=int, default=12532, help="RuneScape item ID for the bond")
    parser.add_argument("--market", type=str, default="osrs", choices=["osrs"], help="RuneScape market to use")
    parser.add_argument("--timestep", type=str, default="24h", choices=["5m", "1h", "6h", "24h"], help="Time interval for RuneScape history")
    parser.add_argument("--symbol", type=str, default="^GSPC", help="Ticker symbol for the equity index")
    parser.add_argument("--max-lag", type=int, default=70, help="Maximum lag in days for cross-correlation")
    parser.add_argument("--output", type=str, default="bond_sp500_analysis.png", help="Filename for the output chart")
    args = parser.parse_args()

    print("Fetching RuneScape bond data...")
    bond = fetch_rs_timeseries(item_id=args.item_id, market=args.market, timestep=args.timestep)
    bond = bond.rename(columns={"price": "bond_price"})
    bond.index = bond.index.tz_convert(None)

    print("Fetching S&P 500 data...")
    spx = fetch_sp500(args.symbol, start=bond.index.min() - timedelta(days=5), end=bond.index.max() + timedelta(days=5))
    spx.index = pd.to_datetime(spx.index).tz_localize(None)

    joint = bond.join(spx, how="inner")
    if joint.empty:
        raise RuntimeError("No overlapping dates between RuneScape and S&P 500 datasets.")

    joint["bond_return"] = joint["bond_price"].pct_change()
    joint["spx_return"] = joint["spx_price"].pct_change()
    joint = joint.dropna(subset=["bond_return", "spx_return"]).copy()

    if joint.empty:
        raise RuntimeError("Not enough overlapping return data after dropping missing values.")

    corr_df = compute_cross_correlation(joint["bond_return"].to_numpy(), joint["spx_return"].to_numpy(), max_lag=args.max_lag)
    corr_df = corr_df.dropna()
    best_idx = corr_df["correlation"].abs().idxmax()
    best_lag = int(corr_df.loc[best_idx, "lag"])
    best_corr = float(corr_df.loc[best_idx, "correlation"])
    best_p = float(corr_df.loc[best_idx, "pvalue"])

    print("\nSummary")
    print("-------")
    print(f"Bond item ID: {args.item_id}")
    print(f"RuneScape data interval: {args.timestep}")
    print(f"Date range: {joint.index.min().date()} to {joint.index.max().date()}")
    print(f"Matching days: {len(joint)}")
    print(f"Best lag: {best_lag} days")
    if best_lag > 0:
        print(f"Interpretation: RuneScape bond returns lead S&P 500 returns by {best_lag} days.")
    elif best_lag < 0:
        print(f"Interpretation: S&P 500 returns lead RuneScape bond returns by {-best_lag} days.")
    else:
        print("Interpretation: No lead/lag detected.")
    print(f"Correlation at best lag: {best_corr:.4f}")
    print(f"Pearson p-value at best lag: {best_p:.3g}")

    vol_ratio = joint["bond_return"].std() / joint["spx_return"].std()
    print(f"Daily volatility ratio (bond / SP500): {vol_ratio:.2f}")

    print(f"Saving chart to {args.output}...")
    plot_results(args.output, joint, corr_df, best_lag)
    print("Done.")


if __name__ == "__main__":
    main()
