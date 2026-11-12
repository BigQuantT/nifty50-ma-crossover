"""
Streamlit Web Dashboard for Nifty 50 Moving Average Crossover Strategy.
Run with: streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Ensure matplotlib uses a writable temp cache directory
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from src.data_loader import fetch_nifty50_data
from src.indicators import add_moving_averages
from src.strategy import generate_crossover_signals
from src.backtester import run_backtest
from src.visualizer import plot_strategy

# Streamlit Page Configuration
st.set_page_config(
    page_title="Nifty 50 MA Crossover Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e9ecef;
    }
    .badge-bullish {
        background-color: #27ae60;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
    }
    .badge-bearish {
        background-color: #e74c3c;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar Controls ---
st.sidebar.title("⚙️ Strategy Parameters")

st.sidebar.subheader("Moving Averages")
short_window = st.sidebar.slider("Short-Term MA Window (Days)", min_value=3, max_value=50, value=10, step=1)
long_window = st.sidebar.slider("Long-Term MA Window (Days)", min_value=15, max_value=150, value=30, step=1)

if short_window >= long_window:
    st.sidebar.error("Error: Short window must be strictly less than Long window!")

ma_type = st.sidebar.selectbox("MA Type", options=["SMA", "EMA"], index=0)

st.sidebar.subheader("Backtest & Capital")
initial_capital = st.sidebar.number_input("Initial Capital (₹)", min_value=10_000.0, value=100_000.0, step=10_000.0)
tx_cost_bps = st.sidebar.slider("Transaction Cost / Slippage (%)", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
allow_short = st.sidebar.checkbox("Allow Short Selling", value=False, help="If unchecked, sell signals exit to cash.")

st.sidebar.subheader("Data Settings")
period = st.sidebar.selectbox("Historical Period", options=["1y", "2y", "5y"], index=1)
use_offline_cache = st.sidebar.checkbox("Force Offline Cache Fallback", value=False)

# --- Data Loading and Execution ---
@st.cache_data(ttl=3600)
def load_data(period_str: str, offline: bool):
    return fetch_nifty50_data(ticker="^NSEI", period=period_str, use_cache_fallback=True)

try:
    raw_df = load_data(period, use_offline_cache)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Compute Indicators
df_ma = add_moving_averages(raw_df, short_window=short_window, long_window=long_window, ma_type=ma_type)

# Generate Signals
df_signals = generate_crossover_signals(df_ma, short_col="Short_MA", long_col="Long_MA", allow_short=allow_short)

# Run Backtest
backtest_out = run_backtest(
    df_signals,
    initial_capital=initial_capital,
    transaction_cost_pct=tx_cost_bps / 100.0,
)
results_df = backtest_out["results_df"]
trades_df = backtest_out["trades_df"]
metrics = backtest_out["metrics"]

# --- Main Dashboard ---
st.title("📊 NIFTY 50 Moving Average Crossover Strategy")
st.caption(f"10-Day Short vs 30-Day Long Moving Average Strategy on recent NIFTY 50 (^NSEI) data.")

# Current Market Regime Badge
latest_pos = df_signals["Position"].iloc[-1]
latest_close = df_signals["Close"].iloc[-1]
latest_short = df_signals["Short_MA"].iloc[-1]
latest_long = df_signals["Long_MA"].iloc[-1]
spread = df_signals["MA_Spread"].iloc[-1]
spread_pct = df_signals["MA_Spread_Pct"].iloc[-1]

regime_col1, regime_col2 = st.columns([2, 1])
with regime_col1:
    if latest_pos == 1:
        st.markdown(
            f"### Current Regime: <span class='badge-bullish'>BULLISH (LONG)</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"### Current Regime: <span class='badge-bearish'>BEARISH (CASH)</span>",
            unsafe_allow_html=True,
        )
with regime_col2:
    st.markdown(
        f"**Latest Close:** ₹{latest_close:,.2f}  \n"
        f"**10d MA:** ₹{latest_short:,.2f} | **30d MA:** ₹{latest_long:,.2f}  \n"
        f"**Spread:** ₹{spread:,.2f} ({spread_pct:+.2f}%)"
    )

st.divider()

# High Level KPI Metric Cards
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric(
    "Strategy Return",
    f"{metrics['Strategy Total Return (%)']}%",
    delta=f"{round(metrics['Strategy Total Return (%)'] - metrics['Benchmark Total Return (%)'], 2)}% vs B&H",
)
kpi2.metric(
    "Max Drawdown",
    f"{metrics['Max Drawdown (%)']}%",
    delta=f"{round(metrics['Benchmark Max Drawdown (%)'] - metrics['Max Drawdown (%)'], 2)}% Lower Risk",
    delta_color="normal",
)
kpi3.metric("Win Rate", f"{metrics['Win Rate (%)']}%", f"{metrics['Winning Trades']}/{metrics['Total Round-Trip Trades']} Trades")
kpi4.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']}", f"Benchmark: {metrics['Benchmark Sharpe Ratio']}")
kpi5.metric("Ending Value", f"₹{metrics['Ending Portfolio Value (₹)']:,.0f}")

st.divider()

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Strategy Visualizer", "📋 Performance Metrics", "📑 Trade Log", "💾 Raw Data"])

with tab1:
    fig = plot_strategy(
        results_df,
        title=f"NIFTY 50 ({short_window}-Day vs {long_window}-Day {ma_type} Crossover)",
    )
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    st.subheader("Comprehensive Quantitative Performance Statistics")
    col_a, col_b = st.columns(2)
    items = list(metrics.items())
    half = len(items) // 2
    with col_a:
        for k, v in items[:half]:
            st.write(f"**{k}:** {v}")
    with col_b:
        for k, v in items[half:]:
            st.write(f"**{k}:** {v}")

with tab3:
    st.subheader(f"Closed Round-Trip Trades ({len(trades_df)} Trades)")
    if not trades_df.empty:
        st.dataframe(trades_df, use_container_width=True)
        csv_trades = trades_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Trades CSV", data=csv_trades, file_name="nifty50_trades.csv", mime="text/csv")
    else:
        st.info("No round-trip trades completed in the selected historical period.")

with tab4:
    st.subheader("Historical NIFTY 50 OHLCV + Indicators Dataset")
    st.dataframe(results_df.tail(100), use_container_width=True)
    csv_data = results_df.to_csv().encode("utf-8")
    st.download_button("📥 Download Full Backtest Dataset", data=csv_data, file_name="nifty50_backtest_data.csv", mime="text/csv")

