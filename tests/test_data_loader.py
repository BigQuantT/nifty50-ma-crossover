"""
Unit tests for data loader and synthetic generator.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.data_loader import clean_dataframe, generate_realistic_nifty_data, load_cached_csv


def test_generate_realistic_nifty_data():
    df = generate_realistic_nifty_data(periods=50)

    assert len(df) == 50
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert (df["High"] >= df["Low"]).all()
    assert (df["High"] >= df["Close"]).all()
    assert (df["Low"] <= df["Close"]).all()
    assert (df["Volume"] > 0).all()


def test_clean_dataframe_timezone_and_types():
    dates = pd.date_range("2024-01-01", periods=5, tz="Asia/Kolkata")
    raw_df = pd.DataFrame(
        {
            "Open": ["20000", "20100", "20200", "20300", "20400"],
            "High": [20100, 20200, 20300, 20400, 20500],
            "Low": [19900, 20000, 20100, 20200, 20300],
            "Close": [20050, 20150, 20250, 20350, 20450],
            "Volume": [1000, 2000, 3000, 4000, 5000],
        },
        index=dates,
    )

    cleaned = clean_dataframe(raw_df)
    assert cleaned.index.tz is None  # Timezone removed
    assert pd.api.types.is_numeric_dtype(cleaned["Open"])
    assert pd.api.types.is_numeric_dtype(cleaned["Close"])


def test_clean_dataframe_empty_error():
    with pytest.raises(ValueError):
        clean_dataframe(pd.DataFrame())

