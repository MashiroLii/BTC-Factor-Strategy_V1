"""
factors.py
----------
Factor construction and signal scoring for BTC/EUR strategy.

Factor selection methodology
-----------------------------
Factors were selected using Information Coefficient (IC) analysis
on BTC/EUR daily data from 2020-2026.
See notebooks/02_Factor_Analysis.ipynb for full analysis.

Selected factors (2 macro factors):

    Factor      | IC (5d) | p-value | Direction
    ------------|---------|---------|----------
    us10y_5d    | -0.101  | 0.000   | Negative
    vix_level   | +0.067  | 0.001   | Positive

Rejected factors and reasons:
    MA7_vs_MA25 : IC dropped from 0.078 to 0.030 on 2020-2026 data
    vol_ratio   : IC collapsed to -0.004, no predictive power
    fear_greed  : p-value 0.051, just outside significance threshold

Design decision
---------------
Only 2020-2026 data was used for factor selection because:
- Pre-2020 BTC market was dominated by retail and had low liquidity
- Institutional participation and macro correlation increased post-2020
- BTC-ETF discussions and macro sensitivity emerged post-2020
- Earlier data introduces structural noise irrelevant to current market
"""

import pandas as pd
import numpy as np


def add_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 2 selected macro factors.

    Parameters
    ----------
    df : DataFrame with macro columns: us10y, vix
         (output of process_data.build_dataset())

    Returns
    -------
    DataFrame with 2 factor columns added:

    f_rates : -(us10y 5-day change)
              Rising yields = bearish for BTC, so we invert.
              IC = -0.101, p = 0.000

    f_vix   : VIX index level
              Higher VIX = slightly bullish for BTC.
              IC = +0.067, p = 0.001
    """
    df = df.copy()

    # Factor 1: Macro rates (inverted - rising yields bearish for BTC)
    df["f_rates"] = -df["us10y"].diff(5)

    # Factor 2: Fear - VIX level
    df["f_vix"] = df["vix"]

    return df


def add_signal_score(df: pd.DataFrame,
                     weights: dict = None) -> pd.DataFrame:
    """
    Combine the 2 macro factors into a single signal score
    using rolling z-score normalisation and weighted sum.

    Rolling z-score (window=252 days) prevents look-ahead bias:
    each day's score is computed using only past data.

    Weights are proportional to IC magnitude:
        us10y_5d  (IC=0.101) : 0.60
        vix_level (IC=0.067) : 0.40

    Parameters
    ----------
    df      : DataFrame with f_rates, f_vix columns
    weights : Dict of factor weights, must sum to 1.0
              Default: IC-proportional weights

    Returns
    -------
    DataFrame with normalised factor columns and signal_score added.
    Normalised columns : f_rates_z, f_vix_z
    Final score column : signal_score (range approximately -2 to +2)
    """
    df = df.copy()

    if weights is None:
        weights = {
            "f_rates" : 0.60,
            "f_vix"   : 0.40,
        }

    # Rolling z-score normalisation
    window = 252    # approximately 1 year of trading days
    score  = pd.Series(0.0, index=df.index)

    for factor, weight in weights.items():
        mean = df[factor].rolling(window, min_periods=60).mean()
        std  = df[factor].rolling(window, min_periods=60).std()
        z    = (df[factor] - mean) / (std + 1e-10)
        z    = z.clip(-2, 2)              # cap extreme values
        score            += z * weight
        df[f"{factor}_z"] = z             # save normalised factor

    df["signal_score"] = score
    return df
