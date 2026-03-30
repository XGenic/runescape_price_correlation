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
cd /home/denisgurskiy/Documents/Coding/runescape_price_correlation
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
.venv/bin/python bond_sp500_correlation.py --item-id 12532 --timestep 24h --symbol ^GSPC --max-lag 70
```

### Notes

- This version uses the RuneScape OSRS bond item ID `12532` and the RuneScape price API for daily history.
- The generated correlation is a simple in-sample analysis and should be treated as exploratory rather than a trading strategy.
