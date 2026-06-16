# BTC-EUR Factor Strategy_V1

A systematic macro factor research framework for Bitcoin (BTC/EUR),
built from scratch using Python and public data sources.

---

## Motivation

Standard technical indicators (RSI, MACD, Bollinger Bands) are widely
used in retail crypto trading. This project tests whether they actually
have statistically significant predictive power — and if not, what does.

**Spoiler:** technical factors have no statistical significance on
post-2020 BTC/EUR daily data. Macro factors do.

---

## Methodology

### 1. Factor Selection via IC Analysis
Each candidate factor is evaluated using the **Information Coefficient
(IC)** — the Spearman rank correlation between the factor value today
and BTC returns over the next 3, 5, and 10 days.

Factors are selected based on:
- Statistical significance: p < 0.05
- IC magnitude: |IC| > 0.05
- Factor independence: |correlation| < 0.7

### 2. Factor Selection Period
Data from **2020-01-01 onwards** was used for factor selection.

Pre-2020 BTC was retail-dominated with low liquidity and minimal macro
correlation. Post-2020 data better reflects current market structure
(institutional participation, ETF discussions, macro sensitivity).

### 3. Signal Construction
Selected factors are normalised using a **rolling z-score** (252-day
window) to prevent look-ahead bias, then combined into a weighted
composite signal score.

### 4. Risk Controls
BTC return distribution has kurtosis = 9.39 (vs 3 for normal),
indicating fat tails. Two risk controls are added:
- **Hard stop loss**: maximum 5% loss per trade regardless of ATR
- **VIX filter**: no new entries when VIX > 35 (extreme fear periods)

---

## Key Findings

### Factor IC Analysis (2020-01-01 → 2026-06-05)

| Factor | IC (5d) | p-value | Selected |
|--------|---------|---------|----------|
| US 10Y yield 5d change | -0.100 | 0.000 | ✅ |
| VIX level | +0.071 | 0.001 | ✅ |
| Fear & Greed index | +0.042 | 0.043 | ❌ (borderline + unstable) |
| MA7 vs MA25 | +0.031 | 0.136 | ❌ (not significant) |
| Volume ratio | -0.003 | 0.869 | ❌ (no predictive power) |

**Main finding:** All technical factors fail the significance test on
post-2020 data. Only macro factors (interest rates, market fear) show
statistically significant predictive power.

### Backtest Results (2024-06-16 → 2026-06-15)

| Metric | Value |
|--------|-------|
| Strategy return | +14.82% |
| Buy & Hold return | -8.15% |
| Alpha | +22.97% |
| Sharpe ratio | 0.75 |
| Max drawdown | -7.61% |
| Win rate | 53.2% |
| Profit factor | 1.32 |
| Total trades | 47 |

---

## Project Structure

```
btc-eur-strategy/
├── data/
│   ├── fetch_data.py          # Fetch BTC/EUR + macro data
│   ├── process_data.py        # Merge and clean all data sources
│   ├── raw/                   # Raw CSV data
│   └── processed/             # Cleaned CSV data
│
├── src/
│   ├── indicators.py          # Technical indicator functions
│   ├── factors.py             # Factor construction + signal scoring
│   └── backtest.py            # Backtest engine + performance metrics
│
├── notebooks/
│   ├── 01_EDA.ipynb           # Exploratory data analysis
│   ├── 02_Factor_Analysis.ipynb  # IC analysis + factor selection
│   └── 03_Backtest.ipynb      # Strategy backtest + results
│
├── outputs/                   # Charts and backtest results
├── requirements.txt
└── README.md
```

---

## Data Sources

| Data | Source | Coverage |
|------|--------|----------|
| BTC/EUR OHLCV | Yahoo Finance | 2018-01-01 → present |
| VIX (market fear) | Yahoo Finance (^VIX) | 2018-01-01 → present |
| US 10Y Treasury Yield | Yahoo Finance (^TNX) | 2018-01-01 → present |
| Fear & Greed Index | alternative.me API | 2018-03-26 → present |

No API key required for any data source.

---

## Limitations

- Factor selection uses overlapping data with the backtest period
  (minor look-ahead bias, documented in `02_Factor_Analysis.ipynb`)
- Backtest covers only 2 years (730 days), limited statistical power
- No transaction costs modelled (Revolut charges 0% crypto fees)
- Past performance does not guarantee future results

---

## How to Run

```bash
pip install -r requirements.txt
```

Open notebooks in order:
1. `notebooks/01_EDA.ipynb`
2. `notebooks/02_Factor_Analysis.ipynb`
3. `notebooks/03_Backtest.ipynb`

Or run on Google Colab — no local setup required.

---

## Disclaimer

This project is for educational and research purposes only.
It is not investment advice.
