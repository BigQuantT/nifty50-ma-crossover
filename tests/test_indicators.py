"""
Unit tests for moving average indicator calculations.
"""

import pytest
import pandas as pd
import numpy as np

from src.indicators import calculate_sma, calculate_ema, add_moving_averages


def test_calculate_sma():
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    sma = calculate_sma(series, window=3)

    assert pd.isna(sma.iloc[0])
    assert pd.isna(sma.iloc[1])
    assert pytest.approx(sma.iloc[2]) == 20.0
    assert pytest.approx(sma.iloc[3]) == 30.0
    assert pytest.approx(sma.iloc[4]) == 40.0


def test_calculate_sma_invalid_window():
    series = pd.Series([1, 2, 3])
    with pytest.raises(ValueError):
        calculate_sma(series, window=0)
    with pytest.raises(ValueError):
        calculate_sma(series, window=-5)


def test_calculate_ema():
    series = pd.Series([10.0, 10.0, 20.0, 20.0, 30.0])
    ema = calculate_ema(series, window=2)

    assert pd.isna(ema.iloc[0])
    # Alpha for span 2 = 2 / (2 + 1) = 2/3
    assert not pd.isna(ema.iloc[1])
    assert len(ema) == 5


def test_add_moving_averages_success():
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    prices = np.linspace(100, 200, 40)
    df = pd.DataFrame({"Close": prices}, index=dates)

    res = add_moving_averages(df, short_window=10, long_window=30, ma_type="SMA")

    assert "SMA_10" in res.columns
    assert "SMA_30" in res.columns
    assert "Short_MA" in res.columns
    assert "Long_MA" in res.columns
    assert "MA_Spread" in res.columns
    assert "MA_Spread_Pct" in res.columns

    # Verify 30th value has valid long MA
    assert not pd.isna(res["Long_MA"].iloc[29])
    # Verify first 29 values of long MA are NaN
    assert pd.isna(res["Long_MA"].iloc[28])


def test_add_moving_averages_invalid_windows():
    df = pd.DataFrame({"Close": [100, 101, 102]})
    with pytest.raises(ValueError, match="strictly less"):
        add_moving_averages(df, short_window=30, long_window=10)

    with pytest.raises(ValueError, match="strictly less"):
        add_moving_averages(df, short_window=15, long_window=15)


def test_add_moving_averages_missing_column():
    df = pd.DataFrame({"Price": [100, 101, 102]})
    with pytest.raises(KeyError):
        add_moving_averages(df, price_col="Close")

