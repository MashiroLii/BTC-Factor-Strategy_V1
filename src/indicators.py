"""
indicators.py
-------------
Technical indicator functions for BTC/EUR strategy.

All functions take a DataFrame with OHLCV columns as input
and return a new DataFrame with indicator columns added.

Indicators
----------
- Moving Averages : MA7, MA25, MA99
- ATR             : Average True Range (used for stop loss / take profit)
- RSI             : Relative Strength Index
- Bollinger Bands : BB_mid, BB_upper, BB_lower, BB_pctB
- Volume Ratio    : vol_ma20, vol_ratio
"""

import pandas as pd
import numpy as np


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple moving averages.

    Adds columns: MA7, MA25, MA99
    """
    df        = df.copy()
    df["MA7"]  = df["close"].rolling(7).mean()
    df["MA25"] = df["close"].rolling(25).mean()
    df["MA99"] = df["close"].rolling(99).mean()
    return df


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Add Average True Range (ATR).
    Used for stop loss and take profit calculation in backtest.

    Adds columns: ATR
    """
    df  = df.copy()
    tr  = pd.concat([
        df["high"] - df["low"],
        abs(df["high"] - df["close"].shift(1)),
        abs(df["low"]  - df["close"].shift(1))
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window).mean()
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Add Relative Strength Index (RSI).

    Adds columns: RSI
    """
    df    = df.copy()
    delta = df["close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(window).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Add Bollinger Bands.

    Adds columns: BB_mid, BB_upper, BB_lower, BB_pctB
    """
    df             = df.copy()
    df["BB_mid"]   = df["close"].rolling(window).mean()
    bb_std         = df["close"].rolling(window).std()
    df["BB_upper"] = df["BB_mid"] + 2 * bb_std
    df["BB_lower"] = df["BB_mid"] - 2 * bb_std
    df["BB_pctB"]  = (
        (df["close"] - df["BB_lower"]) /
        (df["BB_upper"] - df["BB_lower"])
    )
    return df


def add_volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Add volume ratio: current volume vs N-day average.

    Adds columns: vol_ma20, vol_ratio
    """
    df              = df.copy()
    df["vol_ma20"]  = df["volume"].rolling(window).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all indicators at once.

    Parameters
    ----------
    df : Raw OHLCV DataFrame with columns: open, high, low, close, volume

    Returns
    -------
    DataFrame with all indicator columns added
    """
    df = add_moving_averages(df)
    df = add_atr(df)
    df = add_rsi(df)
    df = add_bollinger_bands(df)
    df = add_volume_ratio(df)
    return df
