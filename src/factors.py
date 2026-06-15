"""
factors.py
----------
Factor construction and signal scoring for BTC/EUR strategy.

Selected factors based on IC analysis (see notebooks/02_Factor_Analysis.ipynb):

    Factor       | IC (5d) | Direction | p-value
    -------------|---------|-----------|--------
    MA7_vs_MA25  | +0.078  | Positive  | 0.12
    vol_ratio    | +0.070  | Negative  | 0.18
    us10y_5d     | -0.073  | Negative  | 0.000
    vix_level    | +0.051  | Positive  | 0.005

Note: technical factors (MA, volume) are not statistically significant
on their own but are included as they provide trend/momentum context
independent from macro factors.
"""

import pandas as pd
import numpy as np


def add_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 4 selected factors.

    Factors
    -------
    f_trend  : (MA7 - MA25) / MA25
               Trend factor. Positive = short term above medium term.
               IC = +0.078, direction = positive.

    f_volume : -vol_ratio
               Volume factor. High volume = bearish signal (inverted).
               IC = +0.070, direction = negative.

    f_rates  : -(us10y 5-day change)
               Macro factor. Rising yields = bearish for BTC (inverted).
               IC = -0.073, direction = negative.

    f_vix    : vix level
               Fear factor. Higher VIX = slightly bullish for BTC.
               IC = +0.051, direction = positive.

    Parameters
    ----------
    df : DataFrame with indicator and macro columns already added
         Required columns: MA7, MA25, vol_ratio, us10y, vix

    Returns
    -------
    DataFrame with 4 factor columns added:
    f_trend, f_volume, f_rates, f_vix
    """
    df = df.copy()

    # Factor 1: Trend (MA7 vs MA25)
    df["f_trend"] = (df["MA7"] - df["MA25"]) / df["MA25"]

    # Factor 2: Volume (inverted - high volume is bearish)
    df["f_volume"] = -df["vol_ratio"]

    # Factor 3: Macro - US 10Y yield 5-day change (inverted)
    df["f_rates"] = -df["us10y"].diff(5)

    # Factor 4: Fear - VIX level
    df["f_vix"] = df["vix"]

    return df


def add_signal_score(df: pd.DataFrame,
                     weights: dict = None) -> pd.DataFrame:
    """
    Combine the 4 factors into a single signal score
    using rolling z-score normalisation and weighted sum.

    Rolling z-score (window=252 days) prevents look-ahead bias:
    each day's score is computed using only past data.

    Parameters
    ----------
    df      : DataFrame with factor columns added
    weights : Dict of factor weights, must sum to 1.0
              Default: equal weights (0.25 each)

    Returns
    -------
    DataFrame with normalised factor columns and signal_score added.
    Normalised columns: f_trend_z, f_volume_z, f_rates_z, f_vix_z
    Final score column: signal_score (range approximately -2 to +2)
    """
    df = df.copy()

    if weights is None:
        weights = {
            "f_trend"  : 0.25,
            "f_volume" : 0.25,
            "f_rates"  : 0.25,
            "f_vix"    : 0.25,
        }

    # Rolling z-score normalisation
    window = 252   # approximately 1 year of trading days
    score  = pd.Series(0.0, index=df.index)

    for factor, weight in weights.items():
        mean = df[factor].rolling(window, min_periods=60).mean()
        std  = df[factor].rolling(window, min_periods=60).std()
        z    = (df[factor] - mean) / (std + 1e-10)
        z    = z.clip(-2, 2)             # cap extreme values
        score             += z * weight
        df[f"{factor}_z"]  = z           # save normalised factor

    df["signal_score"] = score
    return df
