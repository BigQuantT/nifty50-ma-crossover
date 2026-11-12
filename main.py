#!/usr/bin/env python3
"""
Nifty 50 Moving Average Crossover Strategy Runner.
Calculates 10-day (Short-Term) and 30-day (Long-Term) Moving Averages,
generates Buy/Sell signals, executes backtesting, and generates performance charts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tabulate import tabulate

from src.data_loader import fetch_nifty50_data
from src.indicators import add_moving_averages
from src.strategy import generate_crossover_signals, extract_trades
from src.backtester import run_backtest
from src.visualizer import plot_strategy


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="NIFTY 50 Moving Average (10-Day vs 30-Day) Crossover Strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--short-window",
        type=int,
        default=10,
        help="Short-term moving average window in days",
    )
    parser.add_argument(
        "--long-window",
        type=int,
        default=30,
        help="Long-term moving average window in days",
    )
    parser.add_argument(
        "--ma-type",
        type=str,
        default="SMA",
        choices=["SMA", "EMA"],
        help="Moving average type (SMA: Simple, EMA: Exponential)",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="2y",
        help="Data period to download from Yahoo Finance (e.g. 1y, 2y, 5y)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Custom start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Custom end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        help="Initial capital in INR (₹)",
    )
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="Allow short positions on sell signals (default: Long-only cash exit)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force using cached/sample offline dataset without network request",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="output/nifty50_ma_strategy.png",
        help="Output file path for the visualization plot",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable chart generation",
    )
    return parser.parse_args()


def print_banner():
    banner = """
================================================================================
          NIFTY 50 MOVING AVERAGE CROSSOVER STRATEGY SYSTEM
    Short-Term: 10 Days  |  Long-Term: 30 Days  |  Index: ^NSEI
================================================================================
    Buy Signal : Short MA crosses ABOVE Long MA (Bullish Golden Cross)
    Sell Signal: Short MA crosses BELOW Long MA (Bearish Death Cross)
================================================================================
"""
    print(banner)


def main():
    args = parse_arguments()
    print_banner()

    # Step 1: Fetch Data
    print(f"[*] Step 1: Fetching NIFTY 50 data (Offline mode: {args.offline})...")
    df = fetch_nifty50_data(
        ticker="^NSEI",
        period=args.period,
        start_date=args.start_date,
        end_date=args.end_date,
        use_cache_fallback=True,
    )

    if df.empty or len(df) < args.long_window:
        print(f"[!] Error: Insufficient data rows ({len(df)}) for window {args.long_window}.")
        sys.exit(1)

    start_dt = df.index[0].strftime("%Y-%m-%d")
    end_dt = df.index[-1].strftime("%Y-%m-%d")
    latest_close = df["Close"].iloc[-1]
    print(f"    Loaded {len(df)} trading days from {start_dt} to {end_dt}.")
    print(f"    Latest Nifty 50 Close: ₹{latest_close:,.2f}")

    # Step 2: Compute Moving Averages
    print(f"[*] Step 2: Calculating {args.ma_type} indicators ({args.short_window}d vs {args.long_window}d)...")
    df_ma = add_moving_averages(
        df,
        short_window=args.short_window,
        long_window=args.long_window,
        ma_type=args.ma_type,
    )

    latest_short = df_ma["Short_MA"].iloc[-1]
    latest_long = df_ma["Long_MA"].iloc[-1]
    spread = df_ma["MA_Spread"].iloc[-1]
    spread_pct = df_ma["MA_Spread_Pct"].iloc[-1]

    print(f"    Latest {args.short_window}-Day MA: ₹{latest_short:,.2f}")
    print(f"    Latest {args.long_window}-Day MA: ₹{latest_long:,.2f}")
    print(f"    Current MA Spread: ₹{spread:,.2f} ({spread_pct:+.2f}%)")

    # Step 3: Generate Signals
    print("[*] Step 3: Generating Buy and Sell crossover signals...")
    df_signals = generate_crossover_signals(
        df_ma,
        short_col="Short_MA",
        long_col="Long_MA",
        allow_short=args.allow_short,
    )

    buy_count = (df_signals["Signal"] == 1).sum()
    sell_count = (df_signals["Signal"] == -1).sum()
    print(f"    Total Buy Signals Generated : {buy_count}")
    print(f"    Total Sell Signals Generated: {sell_count}")

    # Current Market Regime
    current_regime = "BULLISH (LONG)" if df_signals["Position"].iloc[-1] == 1 else "BEARISH (FLAT/CASH)"
    last_signal_row = df_signals[df_signals["Signal"] != 0].tail(1)
    if not last_signal_row.empty:
        last_sig_date = last_signal_row.index[0].strftime("%Y-%m-%d")
        last_sig_action = last_signal_row["Action"].values[0]
        last_sig_price = last_signal_row["Close"].values[0]
        print(f"    Current Status: {current_regime}")
        print(f"    Most Recent Signal: {last_sig_action} on {last_sig_date} @ ₹{last_sig_price:,.2f}")

    # Step 4: Run Backtester
    print("\n[*] Step 4: Running Portfolio Backtest & Performance Analysis...")
    backtest_out = run_backtest(
        df_signals,
        initial_capital=args.capital,
    )
    metrics = backtest_out["metrics"]
    trades_df = backtest_out["trades_df"]

    # Step 5: Display Performance Table
    metrics_table = [[k, v] for k, v in metrics.items()]
    print("\n" + tabulate(metrics_table, headers=["Performance Metric", "Value"], tablefmt="fancy_grid"))

    # Step 6: Display Trades Log
    if not trades_df.empty:
        print("\n[*] Discrete Executed Trades Log (Round-Trips):")
        print(tabulate(trades_df.tail(10), headers="keys", tablefmt="simple", showindex=False))
        if len(trades_df) > 10:
            print(f"    (... showing last 10 of {len(trades_df)} total trades)")
    else:
        print("\n[*] No round-trip trades completed in the selected window.")

    # Step 7: Plot Strategy Visualization
    if not args.no_plot:
        plot_path = Path(args.save_plot)
        print(f"\n[*] Step 5: Generating visual chart to '{plot_path}'...")
        plot_strategy(
            backtest_out["results_df"],
            save_path=plot_path,
            title=f"Nifty 50 ({args.short_window}-Day vs {args.long_window}-Day {args.ma_type} Crossover)",
        )
        print(f"    [OK] Chart successfully saved to: {plot_path.resolve()}")

    print("\n[OK] Pipeline completed successfully!\n")


if __name__ == "__main__":
    main()
