"""
Visualizer module: Generates multi-panel financial charts for the MA Crossover strategy.
Includes price action with moving averages, crossover markers, spread bars, and equity curve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_strategy(
    df: pd.DataFrame,
    save_path: Optional[Path | str] = None,
    show_plot: bool = False,
    title: str = "Nifty 50 (10-Day vs 30-Day Moving Average Crossover Strategy)",
) -> plt.Figure:
    """
    Generate publication-ready 3-panel visualization:
      - Subplot 1: Price, 10-day MA, 30-day MA, Buy/Sell signal markers
      - Subplot 2: MA Spread (Short MA - Long MA) with zero-line shading
      - Subplot 3: Equity Curve (Strategy vs Benchmark Buy & Hold)

    Parameters:
        df: Backtested results DataFrame.
        save_path: Filepath where the plot image should be saved.
        show_plot: Whether to call plt.show() interactively.
        title: Chart super-title.

    Returns:
        plt.Figure: The rendered matplotlib figure object.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(14, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2, 2], "hspace": 0.12},
    )

    dates = df.index

    # --- Subplot 1: Price and Moving Averages ---
    ax1.plot(dates, df["Close"], label="Nifty 50 Close", color="#2c3e50", alpha=0.6, linewidth=1.2)

    short_col = "Short_MA" if "Short_MA" in df.columns else "SMA_10"
    long_col = "Long_MA" if "Long_MA" in df.columns else "SMA_30"

    ax1.plot(dates, df[short_col], label="10-Day Short MA", color="#2980b9", linewidth=1.8)
    ax1.plot(dates, df[long_col], label="30-Day Long MA", color="#e67e22", linewidth=1.8)

    # Plot Buy and Sell markers
    buy_signals = df[df["Signal"] == 1]
    sell_signals = df[df["Signal"] == -1]

    ax1.scatter(
        buy_signals.index,
        buy_signals["Close"],
        marker="^",
        color="#27ae60",
        s=120,
        label="Buy Signal (Golden Cross)",
        zorder=5,
        edgecolors="black",
        linewidth=0.8,
    )

    ax1.scatter(
        sell_signals.index,
        sell_signals["Close"],
        marker="v",
        color="#e74c3c",
        s=120,
        label="Sell Signal (Death Cross)",
        zorder=5,
        edgecolors="black",
        linewidth=0.8,
    )

    ax1.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Price (INR)", fontsize=11, fontweight="semibold")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # --- Subplot 2: MA Spread (Short MA - Long MA) ---
    spread = df[short_col] - df[long_col]
    ax2.plot(dates, spread, color="#34495e", linewidth=1.0, label="MA Spread (10d - 30d)")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax2.fill_between(
        dates,
        spread,
        0,
        where=(spread >= 0),
        color="#27ae60",
        alpha=0.3,
        interpolate=True,
        label="Bullish Regime",
    )
    ax2.fill_between(
        dates,
        spread,
        0,
        where=(spread < 0),
        color="#e74c3c",
        alpha=0.3,
        interpolate=True,
        label="Bearish Regime",
    )
    ax2.set_ylabel("Spread (Pts)", fontsize=11, fontweight="semibold")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # --- Subplot 3: Equity Curve ---
    if "Strategy_Equity" in df.columns and "Benchmark_Equity" in df.columns:
        ax3.plot(
            dates,
            df["Strategy_Equity"],
            label="Strategy (10/30 MA Crossover)",
            color="#2ecc71",
            linewidth=2.0,
        )
        ax3.plot(
            dates,
            df["Benchmark_Equity"],
            label="Benchmark (Buy & Hold Nifty 50)",
            color="#7f8c8d",
            linestyle="--",
            linewidth=1.6,
        )
        ax3.set_ylabel("Portfolio Value (INR)", fontsize=11, fontweight="semibold")
        ax3.legend(loc="upper left", frameon=True, framealpha=0.9)
        ax3.grid(True, linestyle="--", alpha=0.5)

    ax3.set_xlabel("Date", fontsize=11, fontweight="semibold")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()

    if save_path:
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_file, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()

    return fig
