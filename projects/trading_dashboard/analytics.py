"""Your dashboard's calculations. ✏️ Copy each finished function from the take-home notebook into this file.

Replace each `raise NotImplementedError(...)` line with your code from the notebook.
Stuck? The reference answers are in solution/analytics.py.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def to_returns(prices):
    """Daily % changes of a price series, without the empty first day."""
    raise NotImplementedError("Copy your to_returns from the take-home notebook into analytics.py")


def equity_curve(returns):
    """Growth of $1: compound the daily returns."""
    raise NotImplementedError("Copy your equity_curve from the take-home notebook into analytics.py")


def drawdown(equity):
    """How far below its highest value so far the equity is each day (0 = at a new high)."""
    raise NotImplementedError("Copy your drawdown from the take-home notebook into analytics.py")


def annual_return(returns):
    """CAGR: the steady yearly return that gives the same end result."""
    raise NotImplementedError("Copy your annual_return from the take-home notebook into analytics.py")


def annual_volatility(returns):
    """How much daily returns swing, scaled to a year."""
    raise NotImplementedError("Copy your annual_volatility from the take-home notebook into analytics.py")


def sharpe(returns):
    """Average daily return per unit of daily volatility, scaled to a year."""
    raise NotImplementedError("Copy your sharpe from the take-home notebook into analytics.py")


def sortino(returns):
    """Like Sharpe, but only the down days count as risk."""
    raise NotImplementedError("Copy your sortino from the take-home notebook into analytics.py")


def hit_rate(returns):
    """Share of days with a positive return."""
    raise NotImplementedError("Copy your hit_rate from the take-home notebook into analytics.py")


def rolling_sharpe(returns, window=63):
    """Sharpe ratio over the last `window` days, recalculated every day."""
    raise NotImplementedError("Copy your rolling_sharpe from the take-home notebook into analytics.py")


def rolling_volatility(returns, window=30):
    """Annual volatility over the last `window` days, recalculated every day."""
    raise NotImplementedError("Copy your rolling_volatility from the take-home notebook into analytics.py")


def capm(returns, benchmark_returns):
    """Alpha (annualised), beta and R squared of returns against a benchmark."""
    raise NotImplementedError("Copy your capm from the take-home notebook into analytics.py")


def tearsheet(prices):
    """The dashboard's six headline numbers for one price series."""
    raise NotImplementedError("Copy your tearsheet from the take-home notebook into analytics.py")
