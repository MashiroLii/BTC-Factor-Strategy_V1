"""
fetch_data.py
-------------
Fetches historical BTC/EUR daily OHLCV data from Yahoo Finance
and saves it as a CSV file in data/raw/.

Source  : Yahoo Finance (via yfinance)
Pair    : BTC-EUR
Range   : 2018-01-01 → present
Interval: daily
"""

import yfinance as yf
import pandas as pd
import os


# ── Constants ───────────────────────────────────────────────
TICKER      = "BTC-EUR"
START_DATE  = "2018-01-01"
OUTPUT_PATH = "data/raw/btc_eur_daily.csv"


# ── Functions ────────────────────────────────────────────────
def fetch_btc_eur(ticker: str, start: str) -> pd.DataFrame:
    """
    Download daily BTC/EUR OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker : Yahoo Finance ticker, e.g. 'BTC-EUR'
    start  : Start date string, e.g. '2018-01-01'

    Returns
    -------
    DataFrame with columns: open, high, low, close, volume
    Indexed by date (daily, UTC)
    """
    df = yf.download(ticker, start=start, interval="1d",
                     auto_adjust=True, progress=False)

    # Flatten MultiIndex columns
    df.columns = df.columns.get_level_values(0)

    # Lowercase
    df.columns    = [c.lower() for c in df.columns]
    df.index.name = "date"

    # Keep only OHLCV
    df = df[["open", "high", "low", "close", "volume"]]

    # Drop today's incomplete candle
    today = pd.Timestamp.now().normalize()
    df    = df[df.index < today]

    return df


def save_raw(df: pd.DataFrame, path: str) -> None:
    """
    Save DataFrame to CSV.

    Parameters
    ----------
    df   : DataFrame to save
    path : Output file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    print(f"Saved {len(df)} rows → {path}")


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Fetching {TICKER} daily data from Yahoo Finance...")

    df = fetch_btc_eur(TICKER, START_DATE)

    print(f"Rows  : {len(df)}")
    print(f"Range : {df.index[0].date()} → {df.index[-1].date()}")
    print(df.tail(3))

    save_raw(df, OUTPUT_PATH)
