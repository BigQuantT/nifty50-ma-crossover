"""
Moving Average Crossover Trading Strategy.
Generates Buy and Sell signals based on 10-day short-term and 30-day long-term MA crossovers.
"""

from __future__ import annotations

from typing import Dict, List
import pandas as pd
import numpy as np


def generate_crossover_signals(
    df: pd.DataFrame,
    short_col: str = "Short_MA",
    long_col: str = "Long_MA",
    allow_short: bool = False,
) -> pd.DataFrame:
    """
    Generate Buy and Sell signals for a moving average crossover.

    Signal definitions:
        - Signal = +1 (BUY): Short MA crosses above Long MA (Golden Cross)
        - Signal = -1 (SELL): Short MA crosses below Long MA (Death Cross)
        - Signal = 0: No change / Hold previous state

    Position tracking:
        - If allow_short=False (Long-Only):
            Holding position is 1 (in market) or 0 (cash).
        - If allow_short=True (Long/Short):
            Holding position is 1 (long) or -1 (short).

    Parameters:
        df: DataFrame with short and long MA columns.
        short_col: Column name of the short-term moving average.
        long_col: Column name of the long-term moving average.
        allow_short: Whether to enter short positions on sell signal.

    Returns:
        pd.DataFrame: Augmented with:
            - Signal: +1 (Buy), -1 (Sell), 0 (None)
            - Position: 1 (Long), -1 (Short) or 0 (Flat)
            - Action: String representation ('BUY', 'SELL', 'HOLD')
    """
    if short_col not in df.columns or long_col not in df.columns:
        raise KeyError(f"Required MA columns '{short_col}' and '{long_col}' not found.")

    res = df.copy()

    # Mask of rows where both MAs are defined
    valid_mask = res[short_col].notnull() & res[long_col].notnull()

    # Determine bullish status: Short MA > Long MA
    bullish = (res[short_col] > res[long_col]) & valid_mask
    bullish_bool = bullish.astype(bool)

    # Shift yesterday's bullish state, explicitly casting to boolean
    prev_bullish = bullish_bool.shift(1).fillna(False).astype(bool)

    # Crossover conditions:
    # 1. Golden Cross (BUY): yesterday not bullish, today bullish
    # 2. Death Cross (SELL): yesterday bullish, today not bullish
    buy_condition = valid_mask & bullish_bool & (~prev_bullish)
    sell_condition = valid_mask & (~bullish_bool) & prev_bullish

    res["Signal"] = 0
    res.loc[buy_condition, "Signal"] = 1
    res.loc[sell_condition, "Signal"] = -1

    # Position tracking
    if allow_short:
        raw_position = np.where(bullish, 1, -1)
        res["Position"] = np.where(valid_mask, raw_position, 0)
    else:
        raw_position = np.where(bullish, 1, 0)
        res["Position"] = np.where(valid_mask, raw_position, 0)

    # Human-readable Action column
    res["Action"] = "HOLD"
    res.loc[res["Signal"] == 1, "Action"] = "BUY"
    res.loc[res["Signal"] == -1, "Action"] = "SELL"

    return res


def extract_trades(
    df: pd.DataFrame,
    price_col: str = "Close",
    signal_col: str = "Signal",
) -> pd.DataFrame:
    """
    Extract discrete round-trip trade details from the signal series.

    Returns:
        pd.DataFrame containing:
            - Trade #: Sequential index
            - Entry Date, Entry Price
            - Exit Date, Exit Price
            - Days Held
            - Return (%)
            - PnL (Points)
            - Result ('WIN' or 'LOSS')
    """
    trades: List[Dict] = []
    in_trade = False
    entry_date = None
    entry_price = 0.0

    for date, row in df.iterrows():
        sig = row.get(signal_col, 0)
        price = row[price_col]

        if sig == 1:  # Buy signal
            if not in_trade:
                in_trade = True
                entry_date = date
                entry_price = price
        elif sig == -1:  # Sell signal
            if in_trade:
                exit_date = date
                exit_price = price
                pnl = exit_price - entry_price
                ret_pct = (pnl / entry_price) * 100.0
                days_held = (exit_date - entry_date).days if hasattr(exit_date - entry_date, "days") else 0

                trades.append(
                    {
                        "Trade": len(trades) + 1,
                        "Entry Date": entry_date.strftime("%Y-%m-%d") if hasattr(entry_date, "strftime") else str(entry_date),
                        "Entry Price": round(entry_price, 2),
                        "Exit Date": exit_date.strftime("%Y-%m-%d") if hasattr(exit_date, "strftime") else str(exit_date),
                        "Exit Price": round(exit_price, 2),
                        "Days Held": days_held,
                        "PnL (Pts)": round(pnl, 2),
                        "Return (%)": round(ret_pct, 2),
                        "Result": "WIN" if pnl > 0 else "LOSS",
                    }
                )
                in_trade = False

    return pd.DataFrame(trades)
