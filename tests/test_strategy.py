"""
Unit tests for moving average crossover strategy signals and trade extractions.
"""

import pytest
import pandas as pd
import numpy as np

from src.strategy import generate_crossover_signals, extract_trades


def test_crossover_signals_detection():
    # Construct synthetic Short and Long MA series with known crossovers
    # Day 0: Short 10 < Long 20 (no signal)
    # Day 1: Short 15 < Long 20 (no signal)
    # Day 2: Short 25 > Long 20 -> Golden Cross (BUY: +1)
    # Day 3: Short 30 > Long 20 -> Bullish continuation (HOLD: 0)
    # Day 4: Short 18 < Long 20 -> Death Cross (SELL: -1)
    # Day 5: Short 15 < Long 20 -> Bearish continuation (HOLD: 0)
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    df = pd.DataFrame(
        {
            "Close": [100.0, 102.0, 105.0, 108.0, 101.0, 99.0],
            "Short_MA": [10.0, 15.0, 25.0, 30.0, 18.0, 15.0],
            "Long_MA": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        },
        index=dates,
    )

    signals_df = generate_crossover_signals(df, short_col="Short_MA", long_col="Long_MA")

    expected_signals = [0, 0, 1, 0, -1, 0]
    expected_positions = [0, 0, 1, 1, 0, 0]
    expected_actions = ["HOLD", "HOLD", "BUY", "HOLD", "SELL", "HOLD"]

    assert list(signals_df["Signal"]) == expected_signals
    assert list(signals_df["Position"]) == expected_positions
    assert list(signals_df["Action"]) == expected_actions


def test_crossover_allow_short():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "Close": [100.0, 102.0, 101.0, 99.0],
            "Short_MA": [25.0, 25.0, 15.0, 15.0],
            "Long_MA": [20.0, 20.0, 20.0, 20.0],
        },
        index=dates,
    )

    signals_df = generate_crossover_signals(df, allow_short=True)
    # When Short > Long: Position = 1; when Short < Long: Position = -1
    assert list(signals_df["Position"]) == [1, 1, -1, -1]


def test_extract_trades_single_round_trip():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "Close": [100.0, 105.0, 110.0, 100.0],
            "Signal": [1, 0, -1, 0],
        },
        index=dates,
    )

    trades = extract_trades(df, price_col="Close")
    assert len(trades) == 1

    trade = trades.iloc[0]
    assert trade["Trade"] == 1
    assert trade["Entry Price"] == 100.0
    assert trade["Exit Price"] == 110.0
    assert trade["PnL (Pts)"] == 10.0
    assert pytest.approx(trade["Return (%)"]) == 10.0
    assert trade["Result"] == "WIN"
