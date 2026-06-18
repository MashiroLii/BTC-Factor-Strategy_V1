"""
risk_metrics.py
---------------
Risk measurement, stress testing and early warning indicators
for BTC/EUR factor strategy.

Functions
---------
Rolling Risk Metrics:
    rolling_volatility    : Annualised rolling volatility
    var_historical        : Historical Value at Risk (VaR)
    expected_shortfall    : Expected Shortfall / CVaR
    rolling_drawdown      : Rolling drawdown series

Summary Metrics:
    max_drawdown          : Maximum drawdown
    downside_deviation    : Downside deviation (used in Sortino)
    sharpe_ratio          : Annualised Sharpe ratio
    sortino_ratio         : Annualised Sortino ratio
    beta                  : Beta vs benchmark

Stress Testing:
    stress_test           : 6 predefined stress scenarios

Early Warning:
    compute_risk_regime   : Daily risk regime (LOW/MEDIUM/HIGH)
"""

import pandas as pd
import numpy as np


# ── Rolling Risk Metrics ─────────────────────────────────────

def rolling_volatility(returns: pd.Series,
                       window: int = 30) -> pd.Series:
    """
    Annualised rolling volatility.

    Parameters
    ----------
    returns : Daily return series
    window  : Rolling window in days

    Returns
    -------
    Annualised volatility series (%)
    """
    return returns.rolling(window).std() * np.sqrt(365) * 100


def var_historical(returns: pd.Series,
                   confidence: float = 0.95,
                   window: int = 252) -> pd.Series:
    """
    Historical Value at Risk (VaR).
    Maximum loss not exceeded with given confidence level
    over a rolling window.

    Parameters
    ----------
    returns    : Daily return series
    confidence : Confidence level (0.95 or 0.99)
    window     : Rolling window in days

    Returns
    -------
    VaR series (negative = loss)
    """
    return returns.rolling(window).quantile(1 - confidence)


def expected_shortfall(returns: pd.Series,
                       confidence: float = 0.95,
                       window: int = 252) -> pd.Series:
    """
    Expected Shortfall (CVaR).
    Average loss beyond VaR — more conservative than VaR,
    captures tail risk.

    Parameters
    ----------
    returns    : Daily return series
    confidence : Confidence level
    window     : Rolling window in days

    Returns
    -------
    CVaR series (negative = expected loss in tail)
    """
    return returns.rolling(window).apply(
        lambda x: x[x <= np.quantile(x, 1 - confidence)].mean(),
        raw=True
    )


def rolling_drawdown(portfolio: pd.Series) -> pd.Series:
    """
    Rolling drawdown series.

    Parameters
    ----------
    portfolio : Portfolio value series

    Returns
    -------
    Drawdown series (%)
    """
    peak = portfolio.cummax()
    return (portfolio - peak) / peak * 100


# ── Summary Metrics ──────────────────────────────────────────

def max_drawdown(portfolio: pd.Series) -> float:
    """
    Maximum drawdown of a portfolio series.

    Parameters
    ----------
    portfolio : Portfolio value series

    Returns
    -------
    Maximum drawdown (%, negative)
    """
    return rolling_drawdown(portfolio).min()


def downside_deviation(returns: pd.Series,
                       mar: float = 0.0) -> float:
    """
    Downside deviation — std of returns below MAR.
    Used in Sortino ratio calculation.

    Parameters
    ----------
    returns : Daily return series
    mar     : Minimum acceptable return (default 0)

    Returns
    -------
    Annualised downside deviation
    """
    downside = returns[returns < mar]
    return downside.std() * np.sqrt(365)


def sharpe_ratio(returns: pd.Series,
                 risk_free: float = 0.04) -> float:
    """
    Annualised Sharpe ratio.

    Parameters
    ----------
    returns   : Daily return series
    risk_free : Annual risk-free rate (default 4%)

    Returns
    -------
    Sharpe ratio
    """
    excess = returns - risk_free / 365
    return (excess.mean() / returns.std()) * np.sqrt(365) \
           if returns.std() > 0 else 0


def sortino_ratio(returns: pd.Series,
                  risk_free: float = 0.04,
                  mar: float = 0.0) -> float:
    """
    Annualised Sortino ratio.
    Like Sharpe but only penalises downside volatility.

    Parameters
    ----------
    returns   : Daily return series
    risk_free : Annual risk-free rate
    mar       : Minimum acceptable return

    Returns
    -------
    Sortino ratio
    """
    excess = returns.mean() * 365 - risk_free
    dd     = downside_deviation(returns, mar)
    return excess / dd if dd > 0 else 0


def beta(strategy_returns: pd.Series,
         benchmark_returns: pd.Series) -> float:
    """
    Beta of strategy vs benchmark.

    Parameters
    ----------
    strategy_returns  : Strategy daily returns
    benchmark_returns : Benchmark daily returns

    Returns
    -------
    Beta coefficient
    """
    aligned = pd.concat([strategy_returns, benchmark_returns],
                        axis=1).dropna()
    if len(aligned) < 10:
        return 0
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0


# ── Stress Testing ───────────────────────────────────────────

def stress_test(portfolio_values: list,
                position_pct: float = 0.30) -> pd.DataFrame:
    """
    Apply 6 predefined stress scenarios.

    Scenarios
    ---------
    1. BTC crash -20%       : Hard stop limits loss to 5% of position
    2. Equity crash         : VIX filter blocks new entries
    3. USD strengthening    : FX translation loss
    4. Rate shock +100bps   : us10y factor triggers sell signal
    5. Volatility spike     : VIX filter prevents new entries
    6. Liquidity shock      : Slippage, no hard stop protection

    Parameters
    ----------
    portfolio_values : List of daily portfolio values
    position_pct     : Assumed position size (default 30%)

    Returns
    -------
    DataFrame with scenario results
    """
    current_value    = portfolio_values[-1]
    position_value   = current_value * position_pct

    scenarios = []
    for name, btc_shock, note in [
        ("BTC crash -20% (1 week)",         -0.20,
         "Hard stop limits loss to 5% of position"),
        ("Equity crash (Nasdaq -15%)",       -0.12,
         "VIX filter blocks new entries at VIX>35"),
        ("USD strengthening (EUR/USD -5%)",  -0.05,
         "BTC priced in USD, EUR weakening hurts"),
        ("Rate shock (+100bps US 10Y)",      -0.08,
         "us10y_5d factor triggers sell signal"),
        ("Volatility spike (VIX doubles)",   -0.10,
         "VIX filter prevents new entries"),
        ("Liquidity shock (volume -70%)",    -0.02,
         "No hard stop protection vs slippage"),
    ]:
        raw_loss = position_value * btc_shock
        # Hard stop caps loss at 5% of position (except liquidity)
        if "Liquidity" not in name:
            actual_loss = max(raw_loss, -position_value * 0.05)
        else:
            actual_loss = raw_loss

        scenarios.append({
            "scenario"          : name,
            "btc_impact"        : f"{btc_shock*100:.0f}%",
            "strategy_loss_eur" : round(actual_loss, 2),
            "strategy_loss_pct" : round(
                actual_loss / current_value * 100, 2),
            "portfolio_value"   : round(current_value + actual_loss, 2),
            "note"              : note
        })

    return pd.DataFrame(scenarios)


# ── Early Warning / Risk Regime ──────────────────────────────

def compute_risk_regime(df: pd.DataFrame,
                        portfolio_values: list) -> pd.DataFrame:
    """
    Compute daily risk indicators and classify risk regime.

    Risk Regime
    -----------
    LOW    : All indicators normal, safe to trade
    MEDIUM : One or more indicators elevated, trade with caution
    HIGH   : Multiple indicators elevated, reduce exposure

    Scoring
    -------
    btc_vol_30d > 80%  : +2  | > 50% : +1
    drawdown   < -10%  : +2  | < -5% : +1
    macro_pressure > 70: +2  | > 50  : +1
    momentum_reversal  : +1
    Score >= 4 → HIGH, >= 2 → MEDIUM, else LOW

    Parameters
    ----------
    df               : DataFrame with close, vix, us10y columns
    portfolio_values : List of portfolio values from backtest

    Returns
    -------
    DataFrame with daily risk indicators and regime
    """
    risk    = pd.DataFrame(index=df.index)
    returns = df["close"].pct_change()

    # Volatility
    risk["btc_vol_30d"] = \
        returns.rolling(30).std() * np.sqrt(365) * 100

    # Drawdown
    port_series      = pd.Series(
        portfolio_values, index=df.index[:len(portfolio_values)])
    risk["drawdown"] = rolling_drawdown(port_series).reindex(df.index)

    # Momentum reversal warning
    mom_5d  = df["close"].pct_change(5)
    mom_20d = df["close"].pct_change(20)
    risk["momentum_reversal"] = (
        (mom_5d < -0.03) & (mom_20d > 0.05)
    ).astype(int)
    risk["mom_5d"]  = mom_5d * 100
    risk["mom_20d"] = mom_20d * 100

    # Macro pressure score (0-100)
    vix_norm  = (df["vix"] - df["vix"].rolling(252).mean()) / \
                (df["vix"].rolling(252).std() + 1e-10)
    rate_norm = (df["us10y"].diff(5) -
                 df["us10y"].diff(5).rolling(252).mean()) / \
                (df["us10y"].diff(5).rolling(252).std() + 1e-10)
    macro_raw = (vix_norm * 0.5 + rate_norm * 0.5).clip(-3, 3)
    risk["macro_pressure"] = (
        (macro_raw - macro_raw.rolling(252).min()) /
        (macro_raw.rolling(252).max() -
         macro_raw.rolling(252).min() + 1e-10) * 100
    ).clip(0, 100)

    # Risk regime classification
    def classify(row):
        score = 0
        if row["btc_vol_30d"] > 80:    score += 2
        elif row["btc_vol_30d"] > 50:  score += 1
        if row["drawdown"] < -10:      score += 2
        elif row["drawdown"] < -5:     score += 1
        if row["macro_pressure"] > 70: score += 2
        elif row["macro_pressure"] > 50: score += 1
        if row["momentum_reversal"]:   score += 1
        return "HIGH" if score >= 4 else \
               "MEDIUM" if score >= 2 else "LOW"

    risk["risk_regime"] = risk.apply(classify, axis=1)
    risk["vix"]         = df["vix"]
    risk["us10y"]       = df["us10y"]
    risk["close"]       = df["close"]

    return risk.dropna()
