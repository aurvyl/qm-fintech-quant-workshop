"""Trading dashboard: your web app. ✏️ Fill in the TODOs.

Before you start: copy your functions from the take-home notebook into analytics.py (next to this file).
Each TODO replaces one st.info(...) placeholder. Stuck? See solution/app.py.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # repo root, for `backtester`
sys.path.insert(0, str(HERE))  # analytics.py next to this file

import analytics
import pandas as pd
import streamlit as st

from backtester import available_tickers, load_prices

st.set_page_config(page_title="Trading Dashboard", page_icon="📊", layout="wide")
st.title("📊 Trading dashboard")  # ✏️ give it your own name

# The sidebar: what to analyse.
tickers = available_tickers()
with st.sidebar:
    st.header("Settings")
    ticker = st.selectbox("Ticker to analyse", tickers, index=tickers.index("AAPL"))
    benchmark_ticker = st.selectbox("Benchmark (the market)", tickers, index=tickers.index("SPY"))

# The numbers, from your analytics.py.
prices = load_prices(ticker)["Close"]
returns = analytics.to_returns(prices)
equity = analytics.equity_curve(returns)
stats = analytics.tearsheet(prices)

# Six headline numbers in two rows of three. The first one is done for you.
c1, c2, c3 = st.columns(3)
c1.metric("Annual return", f"{stats['Annual return']:.1%}")
# ✏️ TODO 1: show "Annual volatility" in c2 and "Hit rate" in c3 the same way.
c2.info("TODO 1")

c4, c5, c6 = st.columns(3)
# ✏️ TODO 2: show "Sharpe ratio" and "Sortino ratio" in c4 and c5 with two decimals (f"{value:.2f}"),
#            and "Max drawdown" in c6 as a percentage.
c4.info("TODO 2")

performance, risk, capm_tab, data_tab = st.tabs(["Performance", "Risk", "CAPM", "Data"])

with performance:
    st.subheader("Equity curve: growth of $1")
    st.line_chart(equity)
    st.subheader("Rolling Sharpe ratio (63 days)")
    # ✏️ TODO 3: draw analytics.rolling_sharpe(returns) with st.line_chart, like the chart above.
    st.info("TODO 3")

with risk:
    # ✏️ TODO 4: a "Drawdown" chart of analytics.drawdown(equity) (try st.area_chart),
    #            and a "Rolling volatility (30 days)" chart. Give each one an st.subheader.
    st.info("TODO 4")

with capm_tab:
    # ✏️ TODO 5: load the benchmark's prices with load_prices(benchmark_ticker)["Close"],
    #            turn them into returns, call analytics.capm(returns, benchmark_returns)
    #            and show Alpha, Beta and R squared with st.metric in three columns.
    st.info("TODO 5")

with data_tab:
    st.subheader("Summary")
    st.dataframe(pd.Series(stats, name="Value").to_frame())
    st.subheader("Prices")
    st.dataframe(prices)
