"""
backtest.py
-----------
Event-driven backtest engine for BTC/EUR factor strategy.

Entry/Exit logic
----------------
BUY  : signal_score > buy_threshold and no open position
       and VIX < vix_threshold (extreme fear filter)
SELL : signal_score < sell_threshold and open position
STOP : price drops below stop loss (ATR-based or hard stop)
TP   : price rises above entry + tp_mult * ATR
Trailing stop: stop loss moves up as price increases

Position sizing
---------------
Each trade uses position_pct of total portfolio value.
Fixed fraction sizing - not all-in, not risk-based.

Risk controls
-------------
Two stop loss mechanisms to handle BTC fat-tail distribution:
1. ATR-based stop  : entry - sl_mult * ATR (adapts to volatility)
2. Hard stop       : entry * (1 - hard_stop_pct) (absolute floor)
The more conservative (higher) of the two is used.

VIX filter: no new entries when VIX > vix_threshold.
Extreme fear periods have unpredictable tail risks.

Default parameters
------------------
buy_threshold  =  0.3
sell_threshold = -0.3
sl_mult        =  2.0
tp_mult        =  3.0
position_pct   =  0.3
initial_capital= 10000.0
hard_stop_pct  =  0.05   (max 5% loss per trade)
vix_threshold  = 35.0    (no entry when VIX > 35)
"""

import pandas as pd
import numpy as np


def run_backtest(df: pd.DataFrame,
                 buy_threshold:   float = 0.3,
                 sell_threshold:  float = -0.3,
                 sl_mult:         float = 2.0,
                 tp_mult:         float = 3.0,
                 position_pct:    float = 0.3,
                 initial_capital: float = 10000.0,
                 hard_stop_pct:   float = 0.05,
                 vix_threshold:   float = 35.0) -> tuple:
    """
    Event-driven backtest engine with fat-tail risk controls.

    Parameters
    ----------
    df              : DataFrame with signal_score, ATR, vix columns
    buy_threshold   : Score above this triggers BUY
    sell_threshold  : Score below this triggers SELL
    sl_mult         : ATR-based stop loss multiplier
    tp_mult         : ATR-based take profit multiplier
    position_pct    : Fraction of portfolio per trade
    initial_capital : Starting capital in EUR
    hard_stop_pct   : Hard stop loss as fraction of entry price
                      e.g. 0.05 = stop out if price drops 5% from entry
    vix_threshold   : No new entries when VIX exceeds this level
                      Protects against extreme fear / tail risk periods

    Returns
    -------
    trades_df : DataFrame of all trades with PnL
    portfolio : List of daily portfolio values
    """
    capital     = initial_capital
    position    = 0.0
    entry_price = 0.0
    stop_loss   = 0.0
    take_profit = 0.0
    cooldown    = 0

    trades    = []
    portfolio = []

    df_bt = df.dropna(subset=["signal_score", "ATR", "vix"]).copy()

    for i in range(len(df_bt)):
        row   = df_bt.iloc[i]
        price = row["close"]
        atr   = row["ATR"]
        score = row["signal_score"]
        vix   = row["vix"]
        date  = df_bt.index[i]

        total_value = capital + position * price

        if cooldown > 0:
            cooldown -= 1

        # ── Check stop loss / take profit ─────────────────────
        if position > 0:

            if price <= stop_loss:
                proceeds = position * price
                pnl      = proceeds - (position * entry_price)
                pnl_pct  = (price / entry_price - 1) * 100
                capital  += proceeds
                trades.append({
                    "date"    : date,
                    "action"  : "STOP_LOSS",
                    "price"   : price,
                    "pnl"     : pnl,
                    "pnl_pct" : pnl_pct
                })
                position = 0.0
                cooldown = 2

            elif price >= take_profit:
                proceeds = position * price
                pnl      = proceeds - (position * entry_price)
                pnl_pct  = (price / entry_price - 1) * 100
                capital  += proceeds
                trades.append({
                    "date"    : date,
                    "action"  : "TAKE_PROFIT",
                    "price"   : price,
                    "pnl"     : pnl,
                    "pnl_pct" : pnl_pct
                })
                position = 0.0

        # ── Execute signals ───────────────────────────────────
        # VIX filter: block new entries during extreme fear
        vix_ok = vix < vix_threshold

        if score > buy_threshold and position == 0 and cooldown == 0 and vix_ok:
            invest   = total_value * position_pct
            invest   = min(invest, capital)
            btc_buy  = invest / price

            position    = btc_buy
            entry_price = price
            capital    -= invest

            # Use more conservative of ATR stop and hard stop
            atr_stop    = price - sl_mult * atr
            hard_stop   = price * (1 - hard_stop_pct)
            stop_loss   = max(atr_stop, hard_stop)
            take_profit = price + tp_mult * atr

            trades.append({
                "date"       : date,
                "action"     : "BUY",
                "price"      : price,
                "size"       : position,
                "cost"       : invest,
                "sl"         : stop_loss,
                "tp"         : take_profit,
                "score"      : score,
                "vix"        : vix
            })

        elif score < sell_threshold and position > 0:
            proceeds = position * price
            pnl      = proceeds - (position * entry_price)
            pnl_pct  = (price / entry_price - 1) * 100
            capital  += proceeds
            trades.append({
                "date"    : date,
                "action"  : "SELL",
                "price"   : price,
                "pnl"     : pnl,
                "pnl_pct" : pnl_pct
            })
            position = 0.0

        # ── Trailing stop ─────────────────────────────────────
        if position > 0 and price > entry_price:
            atr_trail = price - sl_mult * atr
            if atr_trail > stop_loss:
                stop_loss = atr_trail

        portfolio.append(capital + position * price)

    trades_df = pd.DataFrame(trades)
    return trades_df, portfolio


def compute_metrics(trades_df: pd.DataFrame,
                    portfolio: list,
                    initial_capital: float,
                    df_bt: pd.DataFrame) -> dict:
    """
    Compute backtest performance metrics.

    Parameters
    ----------
    trades_df       : DataFrame of trades from run_backtest()
    portfolio       : List of daily portfolio values
    initial_capital : Starting capital in EUR
    df_bt           : Original DataFrame used in backtest

    Returns
    -------
    Dict of performance metrics
    """
    final_value  = portfolio[-1]
    total_return = (final_value / initial_capital - 1) * 100
    bh_return    = (df_bt["close"].iloc[-1] / df_bt["close"].iloc[0] - 1) * 100

    # Drawdown
    peak   = pd.Series(portfolio).cummax()
    dd     = (pd.Series(portfolio) - peak) / peak * 100
    max_dd = dd.min()

    # Sharpe ratio (annualised)
    daily_ret = pd.Series(portfolio).pct_change().dropna()
    sharpe    = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365) \
                if daily_ret.std() > 0 else 0

    # Win rate and profit factor
    exit_trades = trades_df[
        trades_df["action"].isin(["SELL", "STOP_LOSS", "TAKE_PROFIT"])
    ]
    if len(exit_trades) > 0:
        wins          = exit_trades[exit_trades["pnl"] > 0]
        losses        = exit_trades[exit_trades["pnl"] <= 0]
        win_rate      = len(wins) / len(exit_trades) * 100
        avg_win       = wins["pnl_pct"].mean()   if len(wins)   > 0 else 0
        avg_loss      = losses["pnl_pct"].mean() if len(losses) > 0 else 0
        gross_win     = wins["pnl"].sum()
        gross_loss    = abs(losses["pnl"].sum())
        profit_factor = gross_win / gross_loss   if gross_loss  > 0 else float("inf")
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0

    return {
        "final_value"   : final_value,
        "total_return"  : total_return,
        "bh_return"     : bh_return,
        "alpha"         : total_return - bh_return,
        "total_trades"  : len(exit_trades),
        "win_rate"      : win_rate,
        "avg_win"       : avg_win,
        "avg_loss"      : avg_loss,
        "profit_factor" : profit_factor,
        "max_drawdown"  : max_dd,
        "sharpe"        : sharpe,
    }


def print_report(metrics: dict, initial_capital: float) -> None:
    """Print a formatted backtest report."""
    print("\n" + "="*55)
    print("  BTC/EUR Factor Strategy — Backtest Report")
    print("="*55)
    print(f"  Initial capital   : EUR {initial_capital:>10,.2f}")
    print(f"  Final value       : EUR {metrics['final_value']:>10,.2f}")
    print(f"  Strategy return   : {metrics['total_return']:>+10.2f}%")
    print(f"  Buy & hold return : {metrics['bh_return']:>+10.2f}%")
    print(f"  Alpha             : {metrics['alpha']:>+10.2f}%")
    print(f"  ---")
    print(f"  Total trades      : {metrics['total_trades']:>10d}")
    print(f"  Win rate          : {metrics['win_rate']:>10.1f}%")
    print(f"  Avg win           : {metrics['avg_win']:>+10.2f}%")
    print(f"  Avg loss          : {metrics['avg_loss']:>+10.2f}%")
    print(f"  Profit factor     : {metrics['profit_factor']:>10.2f}")
    print(f"  Max drawdown      : {metrics['max_drawdown']:>10.2f}%")
    print(f"  Sharpe ratio      : {metrics['sharpe']:>10.2f}")
    print("="*55)
