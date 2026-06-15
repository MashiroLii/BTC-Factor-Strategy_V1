"""
fetch_data.py
-------------
Fetches historical BTC/EUR daily OHLCV data from the Kraken public API
and saves it as a CSV file in data/raw/.

No API key required.
"""

import requests
import pandas as pd
import os
import time


# ── Constants ──────────────────────────────────────────────────────────────
KRAKEN_URL  = "https://api.kraken.com/0/public/OHLC"
PAIR        = "XBTEUR"
INTERVAL    = 1440                                      # daily candles (minutes)
OUTPUT_PATH = "data/raw/btc_eur_daily.csv"


# ── Functions ───────────────────────────────────────────────────────────────
def fetch_ohlcv(pair: str, interval: int, since: int = None) -> list:
    """
    Send one request to Kraken OHLC endpoint.

    Parameters
    ----------
    pair     : Kraken trading pair, e.g. 'XBTEUR'
    interval : Candle interval in minutes
    since    : Unix timestamp - if provided, fetch data starting from this time

    Returns
    -------
    List of raw OHLCV rows from the API
    """
    params = {"pair": pair, "interval": interval}
    if since:
        params["since"] = since

    response = requests.get(KRAKEN_URL, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    if data["error"]:
        raise ValueError(f"Kraken API error: {data['error']}")

    # Result key is the pair name (the other key is 'last')
    result_key = [k for k in data["result"].keys() if k != "last"][0]
    return data["result"][result_key]


def fetch_full_history(pair: str, interval: int, batches: int = 2) -> pd.DataFrame:
    """
    Fetch multiple batches to get a longer price history.
    Kraken returns max 720 candles per request.
    With daily candles: 720 candles x 2 batches = ~4 years.

    Parameters
    ----------
    pair     : Kraken trading pair
    interval : Candle interval in minutes
    batches  : Number of API requests to make (default 2)

    Returns
    -------
    DataFrame with columns: open, high, low, close, volume
    Indexed by date (UTC, daily)
    """
    all_rows = []

    # First batch: most recent 720 candles
    print(f"Fetching batch 1 / {batches} ...")
    rows = fetch_ohlcv(pair, interval)
    all_rows.extend(rows)

    # Subsequent batches: step back in time
    for i in range(1, batches):
        earliest_ts = all_rows[0][0]
        since       = earliest_ts - 720 * 86400    # go back 720 days
        print(f"Fetching batch {i + 1} / {batches} ...")
        time.sleep(1)                               # avoid rate limiting
        rows = fetch_ohlcv(pair, interval, since=since)
        all_rows.extend(rows)

    # ── Build DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows, columns=[
        "time", "open", "high", "low", "close", "vwap", "volume", "count"
    ])

    df["date"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("date", inplace=True)

    # Remove duplicates that can appear at batch boundaries
    df = df[~df.index.duplicated(keep="first")].sort_index()

    # Convert to numeric
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])

    # Keep only OHLCV
    df = df[["open", "high", "low", "close", "volume"]]

    # Drop today's candle (incomplete)
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


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Fetching BTC/EUR daily OHLCV from Kraken...")

    df = fetch_full_history(PAIR, INTERVAL, batches=2)

    print(f"Date range : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Total rows : {len(df)}")
    print(df.tail(3))

    save_raw(df, OUTPUT_PATH)
