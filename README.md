# BTC-EUR Factor Strategy

A systematic macro factor research framework for Bitcoin (BTC/EUR),
built from scratch using Python and public data sources.
Includes factor selection, backtesting, and a comprehensive
risk analysis framework motivated by BTC's fat-tail return distribution.

---

## Motivation

Standard technical indicators (RSI, MACD, Bollinger Bands) are widely
used in retail crypto trading. This project tests whether they actually
have statistically significant predictive power — and if not, what does.

**Finding:** Technical factors have no statistical significance on
post-2020 BTC/EUR daily data. Macro factors (interest rates, market
fear) do.

A secondary motivation is risk management: BTC returns have kurtosis
of 9.39 (vs 3 for a normal distribution), meaning standard risk models
significantly underestimate tail risk. This project builds a risk
framework that accounts for this.

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

### 5. Risk Analysis Framework
Inspired by Basel III risk measurement standards:
- **VaR / CVaR**: tail risk quantification
- **Stress testing**: 6 macro scenarios
- **Early warning**: daily risk regime classification (LOW/MEDIUM/HIGH)

---

## Key Findings

### Factor IC Analysis (2020-01-01 → 2026-06-05)

| Factor | IC (5d) | p-value | Selected | Reason |
|--------|---------|---------|----------|--------|
| US 10Y yield 5d change | -0.100 | 0.000 | ✅ | Most significant, clear macro rationale |
| VIX level | +0.071 | 0.001 | ✅ | Stable rolling IC |
| Fear & Greed index | +0.042 | 0.043 | ❌ | Borderline + correlated with trend |
| MA7 vs MA25 | +0.031 | 0.136 | ❌ | Not significant post-2020 |
| Volume ratio | -0.003 | 0.869 | ❌ | No predictive power |

### Backtest Results (rolling 2-year window)

| Metric | Value |
|--------|-------|
| Strategy return | +14.82% |
| Buy & Hold return | -8.15% |
| Alpha | +22.97% |
| Sharpe ratio | 0.37 |
| Sortino ratio | 0.37 |
| Max drawdown | -7.61% |
| Win rate | 53.2% |
| Profit factor | 1.32 |
| Total trades | 47 |

### Risk Metrics

| Metric | Value |
|--------|-------|
| BTC 30d Volatility | 41.0% annualised |
| VaR 95% (daily) | -3.93% |
| VaR 99% (daily) | -6.46% |
| CVaR 95% (daily) | -5.87% |
| Beta vs Nasdaq | 0.109 |
| Beta vs S&P 500 | 0.140 |
| Beta vs BTC | 0.144 |

### Risk Regime Distribution

| Regime | Days | % of time |
|--------|------|-----------|
| 🟢 LOW | 174 | 78% |
| 🟡 MEDIUM | 42 | 19% |
| 🔴 HIGH | 6 | 3% |

### Stress Test (30% position)

| Scenario | Max Loss (EUR) | Max Loss (%) |
|----------|---------------|--------------|
| BTC crash -20% | -173 | -1.5% |
| Equity crash | -173 | -1.5% |
| USD strengthening | -173 | -1.5% |
| Rate shock +100bps | -173 | -1.5% |
| Volatility spike | -173 | -1.5% |
| Liquidity shock | -69 | -0.6% |

Hard stop loss limits single trade loss to 5% of position in all
scenarios except liquidity shock (slippage has no hard stop protection).

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
│   ├── backtest.py            # Backtest engine + performance metrics
│   └── risk_metrics.py        # VaR, CVaR, stress test, risk regime
│
├── notebooks/
│   ├── 01_EDA.ipynb           # Exploratory data analysis
│   ├── 02_Factor_Analysis.ipynb  # IC analysis + factor selection
│   ├── 03_Backtest.ipynb      # Strategy backtest + results
│   └── 04_Risk_Analysis.ipynb # Risk metrics + stress test + dashboard
│
├── outputs/                   # Charts and backtest results
│   ├── risk_dashboard.csv     # Daily risk regime snapshot
│   └── *.png                  # All charts
│
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
| Nasdaq | Yahoo Finance (^IXIC) | 2018-01-01 → present |
| S&P 500 | Yahoo Finance (^GSPC) | 2018-01-01 → present |
| Fear & Greed Index | alternative.me API | 2018-03-26 → present |

No API key required for any data source.

---

## How to Run

```bash
pip install -r requirements.txt
```

Open notebooks in order:
1. `notebooks/01_EDA.ipynb`
2. `notebooks/02_Factor_Analysis.ipynb`
3. `notebooks/03_Backtest.ipynb`
4. `notebooks/04_Risk_Analysis.ipynb`

Or run directly on Google Colab — no local setup required.

---

## Limitations

- Factor selection uses overlapping data with backtest period
  (minor look-ahead bias, documented in `02_Factor_Analysis.ipynb`)
- Backtest covers 2 years (730 days), limited statistical power
- No transaction costs modelled (Revolut charges 0% crypto fees)
- Stress test scenarios use estimated BTC sensitivities,
  not historically calibrated shocks
- Past performance does not guarantee future results

---

## Disclaimer

This project is for educational and research purposes only.
It is not investment advice.
