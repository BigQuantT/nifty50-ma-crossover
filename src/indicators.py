"""
Technical indicator calculations: Simple Moving Average (SMA) and Exponential Moving Average (EMA).
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    Formula:
        SMA_t = (P_t + P_{t-1} + ... + P_{t-N+1}) / N
    """
    if window <= 0:
        raise ValueError(f"Window must be a positive integer, got {window}")
    return series.rolling(window=window, min_periods=window).mean()


def calculate_ema(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    Formula:
        multiplier = 2 / (window + 1)
        EMA_t = (Price_t * multiplier) + (EMA_{t-1} * (1 - multiplier))
    """
    if window <= 0:
        raise ValueError(f"Window must be a positive integer, got {window}")
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def add_moving_averages(
    df: pd.DataFrame,
    short_window: int = 10,
    long_window: int = 30,
    ma_type: str = "SMA",
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Append short-term and long-term moving averages to the dataframe.

    Parameters:
        df: Input DataFrame containing price data.
        short_window: Window size for short-term MA (default: 10).
        long_window: Window size for long-term MA (default: 30).
        ma_type: 'SMA' or 'EMA'.
        price_col: Name of the column to compute MA on (default: 'Close').

    Returns:
        pd.DataFrame: DataFrame with new columns:
            - Short_MA: short-period moving average
            - Long_MA: long-period moving average
            - MA_Spread: Short_MA - Long_MA
            - MA_Spread_Pct: Percentage distance between Short and Long MA
    """
    if short_window >= long_window:
        raise ValueError(
            f"short_window ({short_window}) must be strictly less than long_window ({long_window})"
        )

    if price_col not in df.columns:
        raise KeyError(f"Price column '{price_col}' not found in dataframe columns: {list(df.columns)}")

    result = df.copy()
    ma_type_upper = ma_type.upper()

    if ma_type_upper == "SMA":
        calc_fn = calculate_sma
    elif ma_type_upper == "EMA":
        calc_fn = calculate_ema
    else:
        raise ValueError(f"Unsupported ma_type '{ma_type}'. Must be 'SMA' or 'EMA'.")

    short_col_name = f"{ma_type_upper}_{short_window}"
    long_col_name = f"{ma_type_upper}_{long_window}"

    result[short_col_name] = calc_fn(result[price_col], short_window)
    result[long_col_name] = calc_fn(result[price_col], long_window)

    # Standardized alias columns for downstream strategy usage
    result["Short_MA"] = result[short_col_name]
    result["Long_MA"] = result[long_col_name]
    result["MA_Spread"] = result["Short_MA"] - result["Long_MA"]
    result["MA_Spread_Pct"] = (result["MA_Spread"] / result["Long_MA"]) * 100.0

    return result
