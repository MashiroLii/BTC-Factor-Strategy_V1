"""
fetch_data.py
-------------
Fetches historical BTC/EUR price data and macro indicators
from Yahoo Finance and alternative.me API.

Data sources:
- BTC/EUR OHLCV : Yahoo Finance
- VIX           : Yahoo Finance (^VIX)
- US 10Y Yield  : Yahoo Finance (^TNX)
- Fear & Greed  : alternative.me API

No API key required.
"""

import yfinance as yf
import requests
import pandas as pd
import os


# ── Constants ────────────────────────────────────────────────
START_DATE = "2018-01-01"


# ── Functions ────────────────────────────────────────────────
def fetch_btc_eur(start: str = START_DATE) -> pd.DataFrame:
    """
    Download daily BTC/EUR OHLCV data from Yahoo Finance.

    Parameters
    ----------
    start : Start date string, e.g. '2018-01-01'

    Returns
    -------
    DataFrame with columns: open, high, low, close, volume
    Indexed by date (daily)
    """
    df = yf.download("BTC-EUR", start=start, interval="1d",
                     auto_adjust=True, progress=False)
    df.columns    = df.columns.get_level_values(0)
    df.columns    = [c.lower() for c in df.columns]
    df.index.name = "date"
    df = df[["open", "high", "low", "close", "volume"]]
    df = df[df.index < pd.Timestamp.now().normalize()]
    return df


def fetch_macro(start: str = START_DATE) -> pd.DataFrame:
    """
    Download daily macro data from Yahoo Finance.
    Includes: VIX, US 10Y Treasury Yield.

    Parameters
    ----------
    start : Start date string, e.g. '2018-01-01'

    Returns
    -------
    DataFrame with columns: vix, us10y
    Indexed by date (daily)
    """
    tickers = {
        "^VIX" : "vix",
        "^TNX" : "us10y",
    }

    frames = []
    for ticker, name in tickers.items():
        df = yf.download(ticker, start=start, interval="1d",
                         auto_adjust=True, progress=False)
        df.columns    = df.columns.get_level_values(0)
        df.index.name = "date"
        df = df[["Close"]].rename(columns={"Close": name})
        frames.append(df)

    macro = frames[0].join(frames[1:], how="outer")
    macro = macro[macro.index < pd.Timestamp.now().normalize()]
    return macro


def fetch_fear_greed(limit: int = 3000) -> pd.DataFrame:
    """
    Download Bitcoin Fear & Greed Index from alternative.me API.

    Parameters
    ----------
    limit : Number of days to fetch (max 3000)

    Returns
    -------
    DataFrame with column: fear_greed (0-100)
    Indexed by date (daily)
    """
    resp = requests.get(
        "https://api.alternative.me/fng/",
        params={"limit": limit, "format": "json"},
        timeout=10
    )
    resp.raise_for_status()

    data = resp.json()["data"]
    df   = pd.DataFrame(data)[["timestamp", "value"]]
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
    df.set_index("date", inplace=True)
    df = df[["value"]].rename(columns={"value": "fear_greed"})
    df["fear_greed"] = pd.to_numeric(df["fear_greed"])
    df = df.sort_index()
    df = df[df.index < pd.Timestamp.now().normalize()]
    return df


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Fetching BTC-EUR...")
    btc = fetch_btc_eur()
    print(f"  Rows: {len(btc)}, Range: {btc.index[0].date()} → {btc.index[-1].date()}")

    print("Fetching macro data...")
    macro = fetch_macro()
    print(f"  Rows: {len(macro)}, Range: {macro.index[0].date()} → {macro.index[-1].date()}")
    print(f"  Columns: {list(macro.columns)}")

    print("Fetching Fear & Greed...")
    fng = fetch_fear_greed()
    print(f"  Rows: {len(fng)}, Range: {fng.index[0].date()} → {fng.index[-1].date()}")

    print("\nAll done!")
