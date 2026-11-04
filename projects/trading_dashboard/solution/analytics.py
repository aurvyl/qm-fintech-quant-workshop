"""Reference solution: every number and chart line on the trading dashboard.

Stuck on a step in the take-home notebook? Compare your function with the one here.
Daily data is assumed (one year = 252 trading days) and the risk-free rate is taken as 0,
the same conventions as the workshop's backtester.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def to_returns(prices):
    """Daily % changes of a price series, without the empty first day."""
    return prices.pct_change().dropna()


def equity_curve(returns):
    """Growth of $1: compound the daily returns."""
    return (1 + returns).cumprod()


def drawdown(equity):
    """How far below its highest value so far the equity is each day (0 = at a new high)."""
    return equity / equity.cummax() - 1


def annual_return(returns):
    """CAGR: the steady yearly return that gives the same end result."""
    return (1 + returns).prod() ** (TRADING_DAYS / len(returns)) - 1


def annual_volatility(returns):
    """How much daily returns swing, scaled to a year."""
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe(returns):
    """Average daily return per unit of daily volatility, scaled to a year."""
    return returns.mean() / returns.std() * np.sqrt(TRADING_DAYS)


def sortino(returns):
    """Like Sharpe, but only the down days count as risk."""
    downside = returns[returns < 0].std()
    return returns.mean() / downside * np.sqrt(TRADING_DAYS)


def hit_rate(returns):
    """Share of days with a positive return."""
    return (returns > 0).mean()


def rolling_sharpe(returns, window=63):
    """Sharpe ratio over the last `window` days, recalculated every day."""
    return returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def rolling_volatility(returns, window=30):
    """Annual volatility over the last `window` days, recalculated every day."""
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def capm(returns, benchmark_returns):
    """Alpha (annualised), beta and R squared of returns against a benchmark.

    This is the straight line fitted through "benchmark return (x) vs our return (y)", one dot per day.
    """
    both = pd.concat([returns, benchmark_returns], axis=1, keys=["asset", "bench"]).dropna()
    beta = both["asset"].cov(both["bench"]) / both["bench"].var()
    alpha_daily = both["asset"].mean() - beta * both["bench"].mean()
    r_squared = both["asset"].corr(both["bench"]) ** 2
    return {"Alpha": alpha_daily * TRADING_DAYS, "Beta": beta, "R squared": r_squared}


def tearsheet(prices):
    """The dashboard's six headline numbers for one price series."""
    returns = to_returns(prices)
    return {
        "Annual return": annual_return(returns),
        "Annual volatility": annual_volatility(returns),
        "Sharpe ratio": sharpe(returns),
        "Sortino ratio": sortino(returns),
        "Max drawdown": drawdown(equity_curve(returns)).min(),
        "Hit rate": hit_rate(returns),
    }
