# RuneScape Bond vs S&P 500 Correlation

This project creates a proof-of-concept analysis inspired by the Reddit post about RuneScape bond prices predicting S&P 500 performance.

It fetches:
- RuneScape bond daily price series from the RuneScape price API (`prices.runescape.wiki`)
- S&P 500 price history from `yfinance`

Then it computes:
- daily returns
- cross-correlation across lead/lag days
- the best lead/lag signal
- a chart showing normalized prices, lag correlation, and rolling volatility

## Setup

```bash
# From the repository root:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python bond_sp500_correlation.py
```

## Output

- `bond_sp500_analysis.png` — chart with normalized price series, lag correlation, and rolling volatility

## Customize

```bash
.venv/bin/python bond_sp500_correlation.py --item-id 13190 --timestep 24h --symbol ^GSPC --max-lag 70
```

### Notes

- OSRS tradeable bonds are item `13190`. The old default `12532` was a Dragon sq shield ornament kit, not a bond.
- The generated correlation is a simple in-sample analysis and should be treated as exploratory rather than a trading strategy.

## Historical archive study

**Longer history does exist.** The [Weird Gloop exchange archive](https://api.weirdgloop.org/exchange/history/osrs/all?id=13190) stores bond GE guide-price updates back to the instrument's 2015 launch. This is a different measurement from the recent transaction-average feed, so it is analyzed separately rather than spliced into the original series.

- **[Historical HTML report](rs_bond_sp500_historical_study.html)** — standalone, offline-capable report with four embedded figures, audit findings, statistical results, source comparison, and recommendations.
- **[Historical notebook](rs_bond_sp500_historical_investigation.ipynb)** — executed lab journal and reproducible computations.
- The original one-year notebook, HTML report, fetchers, and forecast ledger remain preserved as the earlier experiment.

```bash
.venv/bin/jupyter lab rs_bond_sp500_historical_investigation.ipynb
# After executing and saving the notebook, regenerate HTML without fetching/fitting:
.venv/bin/python render_historical_report.py
```

Run historical Sections 0–1 once per kernel. Subsequent sections compute their own prerequisites from cache. Section 5 freezes the historical forecast; full-range regime and feed-comparison sections defer until that report exists. Section 8 never opens an unevaluated final period.

The separate `cache/historical/` contains the raw archive JSON/parquet, market/control parquet files, snapshot date, research summary, and historical evaluation ledger. Pre-launch records are excluded and retained in the audit; multiple same-day updates use the final timestamp; missing dates are never filled. Guide prices use a conservative next-UTC-day availability convention, not an assumed RuneLite bucket definition. Historical first-publication times remain unverified.

Stationarity is diagnosed on contiguous training blocks rather than joining observations across gaps; local passes do not establish global stationarity. The full training lag search receives IID and 5-/20-session block-permutation sensitivity checks. Granger lag columns are constructed before incomplete rows are dropped. Breaks are descriptive BIC regression segmentation, not formal Bai–Perron significance.

**Historical evaluation is retrospective, not a pristine holdout:** some final-period dates were already exposed by the first study. A new fingerprint does not undo that exposure. Preserve both ledgers; genuinely confirmatory replication requires future outcomes.

The new computations live in `bond_historical_data.py`, `bond_historical_stats.py`, and `bond_source_comparison.py`, reusing the existing market fetcher, alignment, controls, regression, and walk-forward code. The HTML renderer takes headline metrics from metadata attached to the same saved notebook outputs, preventing a report from mixing unrelated cache snapshots.

```bash
.venv/bin/python -m unittest -v test_bond_historical test_bond_investigation
```

Historical integrity checks defend launch exclusions, final daily quotes, gap-aware Granger inference, contiguous stationarity coverage, cross-source lag signs, and zero-versus-real regression break detection.

## Original one-year HTML study report

Open [rs_bond_sp500_study.html](rs_bond_sp500_study.html) directly in a browser. This standalone report contains the frozen study results, three embedded notebook charts, methodological limitations, and six prioritized follow-up studies with explicit strengthening/weakening criteria.

The page works offline without a server or external assets, adapts to mobile screens, and includes print/PDF styling. It does not fetch data or rerun the holdout. Notebook and README links require the companion repository files; external source links require internet access.

## Investigation notebook

```bash
.venv/bin/jupyter lab rs_bond_sp500_investigation.ipynb
```

The notebook reuses `fetch_rs_timeseries` and `fetch_sp500` directly from the script. Its nine sections cover provenance and caching, ADF/KPSS stationarity, a ±90-trading-session lag search with 1,000 whole-search permutations, HAC-robust controlled regression, bidirectional Granger tests, regime diagnostics, expanding walk-forward validation, and an evidence-based summary. Saved outputs include the executed tables, plots, and observations.

Run Sections 0–1 once per kernel. Every later section can then run independently; its training prerequisites are recomputed as needed. Section 1 fetches and caches all four raw sources (bonds, S&P, VIX, UMCSENT) as parquet under ignored `cache/`. Existing caches are never automatically refreshed. A frozen acquisition timestamp prevents a possibly incomplete Yahoo daily row from being promoted to a completed close on later reruns.

### Research safeguards and limitations

- The existing RS endpoint supplies a rolling snapshot, not a 2013 archive. The notebook uses every returned observation and reports actual coverage. [OSRS bonds launched in March 2015](https://oldschool.runescape.wiki/w/Old_school_bond); the [price API](https://oldschool.runescape.wiki/w/RuneScape:Real-time_Prices) cannot provide the historical span assumed by the original proposal.
- RS daily-average buckets are usable only after completion. Returns are aligned on equity sessions before lagging; missing sessions are not forward-filled or compressed.
- A chronological 60/20/20 split is reserved before signal discovery. Both returns must pass the joint ADF/KPSS checkpoint; failed or infeasible prerequisites are reported instead of replaced with synthetic results.
- Lag selection, regression, and Granger diagnostics use training data only. VIX is lagged; UMCSENT is delayed two months, but its revised/latest-vintage values still make it a retrospective control, not a real-time predictor.
- `ruptures` provides BIC-selected multiple-break regression segmentation here, **not a formal Bai–Perron significance test**. Zero breaks is allowed and penalty sensitivity is reported. Out-of-range macro events are not invented.
- Section 6 shows training history until the holdout is evaluated. Section 7 persists its one-shot report before showing full-history regime plots; Section 8 never opens an unevaluated holdout.
- The positive lag stays fixed; validation chooses a prespecified signal gate. Expanding OLS refits every 20 sessions use only prior outcomes, including earlier test batches once observed. Accuracy is compared with a matched historical-majority baseline.
- Keep `cache/holdout_*.json` and its lock files. Identical reruns load the frozen report; changing the rule for the same dataset is rejected. A new cache directory can acquire newer data, but overlapping old test dates are **not** a fresh holdout. Do not delete the ledger to tune on spent outcomes.

Computations are separated into `bond_investigation_data.py`, `bond_investigation_stats.py`, and `bond_investigation_validation.py`; the notebook remains the readable lab journal. The CLI is the simpler exploratory plotter, not the notebook's leakage-safe forecast protocol.

### Integrity checks

```bash
.venv/bin/python -m unittest -v test_bond_investigation
```

These checks cover bucket availability, weekend macro alignment, numeric stationarity decisions, signed-lag permutation calibration, unseen-label invariance, holdout reuse protection, infeasible Granger orders, and zero-versus-real regime breaks.
