"""
process_data.py
---------------
Fetches all data sources, merges them, computes derived columns,
and saves the final dataset to data/processed/.

Output columns:
    OHLCV   : open, high, low, close, volume
    Macro   : vix, us10y, fear_greed
    Derived : vol_ma20, vol_ratio, MA7, MA25, MA7_vs_MA25, us10y_5d
"""

import yfinance as yf
import requests
import pandas as pd
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.fetch_data import fetch_btc_eur, fetch_macro, fetch_fear_greed


# ── Constants ────────────────────────────────────────────────
OUTPUT_PATH = "data/processed/btc_eur_clean.csv"


# ── Functions ────────────────────────────────────────────────
def build_dataset() -> pd.DataFrame:
    """
    Fetch all data sources, merge, and compute derived columns
    needed for factor construction.

    Returns
    -------
    DataFrame with columns:
        OHLCV   : open, high, low, close, volume
        Macro   : vix, us10y, fear_greed
        Derived : vol_ma20, vol_ratio, MA7, MA25, MA7_vs_MA25, us10y_5d
    """
    print("Fetching BTC-EUR...")
    btc = fetch_btc_eur()

    print("Fetching macro data...")
    macro = fetch_macro()

    print("Fetching Fear & Greed...")
    fng = fetch_fear_greed()

    # ── Merge all on date index ───────────────────────────────
    df = btc.join(macro, how="left")
    df = df.join(fng,   how="left")

    # Forward-fill macro data (weekends / holidays)
    df[["vix", "us10y", "fear_greed"]] = \
        df[["vix", "us10y", "fear_greed"]].ffill()

    # Drop rows where macro data not yet available
    df = df.dropna()

    # ── Derived columns needed for factors ───────────────────
    # vol_ratio: volume relative to 20-day average
    df["vol_ma20"]  = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]

    # MA7_vs_MA25: short vs medium term trend
    df["MA7"]         = df["close"].rolling(7).mean()
    df["MA25"]        = df["close"].rolling(25).mean()
    df["MA7_vs_MA25"] = (df["MA7"] - df["MA25"]) / df["MA25"]

    # us10y_5d: 5-day change in US 10Y yield
    df["us10y_5d"] = df["us10y"].diff(5)

    # Drop NaN from rolling calculations
    df = df.dropna()

    print(f"\nDataset ready:")
    print(f"  Rows    : {len(df)}")
    print(f"  Range   : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Columns : {list(df.columns)}")

    return df


def save_processed(df: pd.DataFrame, path: str) -> None:
    """
    Save cleaned DataFrame to CSV.

    Parameters
    ----------
    df   : Processed DataFrame
    path : Output file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    print(f"Saved {len(df)} rows → {path}")


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    df = build_dataset()
    save_processed(df, OUTPUT_PATH)
