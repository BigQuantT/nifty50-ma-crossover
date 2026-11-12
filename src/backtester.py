"""
Backtester module: Simulates strategy performance against benchmark (Buy & Hold).
Calculates cumulative returns, Sharpe Ratio, Maximum Drawdown, CAGR, Win Rate, and Calmar Ratio.
"""

from __future__ import annotations

from typing import Any, Dict
import numpy as np
import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    price_col: str = "Close",
    position_col: str = "Position",
    initial_capital: float = 100_000.0,
    transaction_cost_pct: float = 0.0005,  # 0.05% brokerage + STT + slippage
    risk_free_rate: float = 0.065,         # 6.5% typical Indian risk-free rate
    trading_days_per_year: int = 252,
) -> Dict[str, Any]:
    """
    Simulate trading strategy execution and evaluate quantitative metrics.

    Parameters:
        df: DataFrame containing price and position.
        price_col: Name of column with execution prices (default: 'Close').
        position_col: Name of column with trade positions (1, 0, -1).
        initial_capital: Starting portfolio value in currency units (INR).
        transaction_cost_pct: Brokerage and slippage cost per trade turn.
        risk_free_rate: Annualized risk-free rate for Sharpe Ratio.
        trading_days_per_year: Normalization factor for annualized metrics.

    Returns:
        Dict with keys:
            - 'results_df': DataFrame with daily returns, equity curves, drawdowns.
            - 'metrics': Dictionary of quantitative performance statistics.
    """
    res = df.copy()

    # Asset daily return
    res["Market_Return"] = res[price_col].pct_change().fillna(0.0)

    # Strategy daily return: use yesterday's position to avoid lookahead bias
    res["Lagged_Position"] = res[position_col].shift(1).fillna(0.0)

    # Transaction costs triggered on position changes
    position_changes = res["Lagged_Position"].diff().abs().fillna(0.0)
    tx_costs = position_changes * transaction_cost_pct

    # Net strategy return
    res["Strategy_Return"] = (res["Lagged_Position"] * res["Market_Return"]) - tx_costs

    # Cumulative growth curves
    res["Benchmark_Equity"] = initial_capital * (1.0 + res["Market_Return"]).cumprod()
    res["Strategy_Equity"] = initial_capital * (1.0 + res["Strategy_Return"]).cumprod()

    # Drawdown calculations
    res["Strategy_Peak"] = res["Strategy_Equity"].cummax()
    res["Strategy_Drawdown"] = (res["Strategy_Equity"] - res["Strategy_Peak"]) / res["Strategy_Peak"]

    res["Benchmark_Peak"] = res["Benchmark_Equity"].cummax()
    res["Benchmark_Drawdown"] = (res["Benchmark_Equity"] - res["Benchmark_Peak"]) / res["Benchmark_Peak"]

    # Compute high-level metrics
    n_days = len(res)
    years = max(n_days / trading_days_per_year, 1.0 / trading_days_per_year)

    strat_total_return = (res["Strategy_Equity"].iloc[-1] / initial_capital) - 1.0
    bench_total_return = (res["Benchmark_Equity"].iloc[-1] / initial_capital) - 1.0

    strat_cagr = (res["Strategy_Equity"].iloc[-1] / initial_capital) ** (1.0 / years) - 1.0
    bench_cagr = (res["Benchmark_Equity"].iloc[-1] / initial_capital) ** (1.0 / years) - 1.0

    strat_vol = res["Strategy_Return"].std() * np.sqrt(trading_days_per_year)
    bench_vol = res["Market_Return"].std() * np.sqrt(trading_days_per_year)

    # Sharpe ratio
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / trading_days_per_year) - 1.0
    strat_excess_returns = res["Strategy_Return"] - rf_daily
    strat_sharpe = (
        (strat_excess_returns.mean() / (res["Strategy_Return"].std() + 1e-9)) * np.sqrt(trading_days_per_year)
        if res["Strategy_Return"].std() > 0
        else 0.0
    )

    bench_excess_returns = res["Market_Return"] - rf_daily
    bench_sharpe = (
        (bench_excess_returns.mean() / (res["Market_Return"].std() + 1e-9)) * np.sqrt(trading_days_per_year)
        if res["Market_Return"].std() > 0
        else 0.0
    )

    strat_max_dd = abs(float(res["Strategy_Drawdown"].min()))
    bench_max_dd = abs(float(res["Benchmark_Drawdown"].min()))

    calmar_ratio = strat_cagr / (strat_max_dd + 1e-9)

    # Trade counts and win rate from signals
    from .strategy import extract_trades

    trades_df = extract_trades(res, price_col=price_col)
    total_trades = len(trades_df)

    if total_trades > 0:
        win_trades = (trades_df["PnL (Pts)"] > 0).sum()
        loss_trades = (trades_df["PnL (Pts)"] <= 0).sum()
        win_rate = (win_trades / total_trades) * 100.0

        gross_profit = trades_df.loc[trades_df["PnL (Pts)"] > 0, "PnL (Pts)"].sum()
        gross_loss = abs(trades_df.loc[trades_df["PnL (Pts)"] < 0, "PnL (Pts)"].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan

        avg_return = trades_df["Return (%)"].mean()
        max_win = trades_df["Return (%)"].max()
        max_loss = trades_df["Return (%)"].min()
        avg_holding = trades_df["Days Held"].mean()
    else:
        win_trades = 0
        loss_trades = 0
        win_rate = 0.0
        profit_factor = np.nan
        avg_return = 0.0
        max_win = 0.0
        max_loss = 0.0
        avg_holding = 0.0

    exposure_pct = (res["Lagged_Position"] != 0).mean() * 100.0

    metrics = {
        "Initial Capital (₹)": initial_capital,
        "Ending Portfolio Value (₹)": round(float(res["Strategy_Equity"].iloc[-1]), 2),
        "Strategy Total Return (%)": round(strat_total_return * 100.0, 2),
        "Benchmark Total Return (%)": round(bench_total_return * 100.0, 2),
        "Strategy CAGR (%)": round(strat_cagr * 100.0, 2),
        "Benchmark CAGR (%)": round(bench_cagr * 100.0, 2),
        "Annualized Volatility (%)": round(strat_vol * 100.0, 2),
        "Benchmark Volatility (%)": round(bench_vol * 100.0, 2),
        "Sharpe Ratio": round(strat_sharpe, 2),
        "Benchmark Sharpe Ratio": round(bench_sharpe, 2),
        "Max Drawdown (%)": round(strat_max_dd * 100.0, 2),
        "Benchmark Max Drawdown (%)": round(bench_max_dd * 100.0, 2),
        "Calmar Ratio": round(calmar_ratio, 2),
        "Total Round-Trip Trades": total_trades,
        "Winning Trades": int(win_trades),
        "Losing Trades": int(loss_trades),
        "Win Rate (%)": round(win_rate, 2),
        "Profit Factor": round(profit_factor, 2) if not np.isnan(profit_factor) else "N/A",
        "Average Trade Return (%)": round(avg_return, 2),
        "Best Trade Return (%)": round(max_win, 2),
        "Worst Trade Return (%)": round(max_loss, 2),
        "Average Holding Period (Days)": round(avg_holding, 1),
        "Market Exposure (%)": round(exposure_pct, 2),
    }

    return {
        "results_df": res,
        "trades_df": trades_df,
        "metrics": metrics,
    }

