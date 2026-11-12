"""
Data loader module for fetching NIFTY 50 index historical data.
Supports yfinance downloads and local offline fallback caches.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TICKER = "^NSEI"  # Yahoo Finance symbol for NIFTY 50
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CACHE_FILE = DEFAULT_DATA_DIR / "nifty50_sample.csv"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize and clean raw DataFrame downloaded from yfinance or loaded from CSV.
    Handles MultiIndex columns, timezone-unaware conversion, and missing data.
    """
    if df.empty:
        raise ValueError("Provided DataFrame is empty.")

    cleaned = df.copy()

    # Handle MultiIndex columns if present
    if isinstance(cleaned.columns, pd.MultiIndex):
        cleaned.columns = [col[0] for col in cleaned.columns]

    # Normalize Index to DatetimeIndex
    if not isinstance(cleaned.index, pd.DatetimeIndex):
        cleaned.index = pd.to_datetime(cleaned.index, errors="coerce")

    # Drop rows where index could not be parsed to datetime
    cleaned = cleaned[cleaned.index.notnull()]

    # Ensure required columns exist
    required_cols = ["Open", "High", "Low", "Close"]
    missing = [c for c in required_cols if c not in cleaned.columns]
    if missing:
        if "Adj Close" in cleaned.columns and "Close" in missing:
            cleaned["Close"] = cleaned["Adj Close"]
        else:
            raise KeyError(f"DataFrame is missing required price columns: {missing}")

    # Standardize column numeric types
    for col in ["Open", "High", "Low", "Close"]:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    if "Volume" in cleaned.columns:
        cleaned["Volume"] = pd.to_numeric(cleaned["Volume"], errors="coerce").fillna(0)
    else:
        cleaned["Volume"] = 0

    # Remove timezone info for consistent date matching
    if cleaned.index.tz is not None:
        cleaned.index = cleaned.index.tz_localize(None)

    cleaned = cleaned.dropna(subset=["Close"]).sort_index()
    # Reorder columns neatly
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in cleaned.columns]
    cleaned = cleaned[cols]
    cleaned.index.name = "Date"
    return cleaned


def load_cached_csv(cache_path: Path | str) -> pd.DataFrame:
    """
    Load and parse a cached CSV file safely, supporting single-level or multi-level headers.
    """
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found at {path}")

    # Check for multi-index header
    with open(path, "r", encoding="utf-8") as f:
        first_lines = [f.readline() for _ in range(3)]

    if len(first_lines) > 1 and "Ticker" in first_lines[1]:
        df = pd.read_csv(path, header=[0, 1], index_col=0)
        df.columns = [col[0] for col in df.columns]
        df = df.iloc[1:]
    else:
        df = pd.read_csv(path, index_col=0)

    return clean_dataframe(df)


def fetch_nifty50_data(
    ticker: str = DEFAULT_TICKER,
    period: str = "2y",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache_fallback: bool = True,
    save_cache: bool = True,
    cache_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Fetch historical price data for Nifty 50 (^NSEI).
    
    Parameters:
        ticker: Ticker symbol (default '^NSEI')
        period: Period string if dates not specified (e.g. '1y', '2y', '5y')
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        use_cache_fallback: Fall back to local CSV if download fails
        save_cache: Save downloaded data to cache if successful
        cache_path: Path to CSV cache file

    Returns:
        pd.DataFrame: Cleaned OHLCV dataframe indexed by date
    """
    cache_file = Path(cache_path) if cache_path else DEFAULT_CACHE_FILE

    try:
        import yfinance as yf

        logger.info("Attempting to fetch '%s' data from Yahoo Finance...", ticker)
        if start_date and end_date:
            raw_df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        else:
            raw_df = yf.download(ticker, period=period, progress=False)

        if raw_df is not None and not raw_df.empty:
            df = clean_dataframe(raw_df)
            logger.info("Successfully fetched %d records for %s from Yahoo Finance.", len(df), ticker)

            if save_cache:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_file)
                logger.info("Saved data cache to %s", cache_file)

            return df
        else:
            logger.warning("Empty data returned by Yahoo Finance.")
    except Exception as e:
        logger.warning("Failed to fetch online data (%s).", e)

    if use_cache_fallback and cache_file.exists():
        logger.info("Loading fallback data from %s...", cache_file)
        return load_cached_csv(cache_file)

    logger.info("Generating realistic synthetic Nifty 50 data as fallback...")
    return generate_realistic_nifty_data()


def generate_realistic_nifty_data(
    start_date: str = "2023-01-01",
    periods: int = 500,
    base_price: float = 21500.0,
    annual_vol: float = 0.14,
    annual_drift: float = 0.12,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic geometric Brownian motion synthetic Nifty 50 data.
    Useful for unit testing and offline demonstrations.
    """
    np.random.seed(seed)
    dates = pd.bdate_range(start=start_date, periods=periods)
    n = len(dates)

    dt = 1.0 / 252.0
    daily_drift = (annual_drift - 0.5 * annual_vol**2) * dt
    daily_vol = annual_vol * np.sqrt(dt)

    random_shocks = np.random.normal(0, 1, n)
    log_returns = daily_drift + daily_vol * random_shocks
    log_returns[0] = 0

    price_series = base_price * np.exp(np.cumsum(log_returns))

    daily_spread = np.random.uniform(0.003, 0.012, n)
    high = price_series * (1 + daily_spread * np.random.uniform(0.4, 0.9, n))
    low = price_series * (1 - daily_spread * np.random.uniform(0.4, 0.9, n))
    open_p = price_series * (1 + daily_spread * np.random.uniform(-0.5, 0.5, n))
    volume = np.random.randint(150_000_000, 350_000_000, n)

    df = pd.DataFrame(
        {
            "Open": np.round(open_p, 2),
            "High": np.round(np.maximum.reduce([high, open_p, price_series]), 2),
            "Low": np.round(np.minimum.reduce([low, open_p, price_series]), 2),
            "Close": np.round(price_series, 2),
            "Volume": volume,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df
