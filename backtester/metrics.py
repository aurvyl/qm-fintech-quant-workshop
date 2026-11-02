"""Performance metrics: how good, and how risky, was a strategy?

Every function takes plain pandas Series, so you can use them on anything.
Daily data is assumed: one year = 252 trading days.
"""

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(equity: pd.Series) -> float:
    """How much the portfolio grew overall. 0.25 means +25%."""
    return equity.iloc[-1] / equity.iloc[0] - 1


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate: the steady yearly return that gives the same end result."""
    n_days = len(equity) - 1
    if n_days <= 0:
        return float("nan")
    return (equity.iloc[-1] / equity.iloc[0]) ** (TRADING_DAYS / n_days) - 1


def annual_volatility(returns: pd.Series) -> float:
    """How much daily returns bounce around, scaled to a year. Higher = bumpier ride."""
    return returns.std() * math.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Return earned per unit of risk, above what cash would have paid.

    risk_free_rate is the yearly cash rate (0.04 = 4%); the workshop default is 0.
    Above 1 is often treated as good. NaN when returns never vary
    (for example a strategy that never trades), because you can't divide by zero risk.
    """
    std = returns.std()
    if not np.isfinite(std) or std < 1e-12:
        return float("nan")
    excess = returns - ((1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1)
    return excess.mean() / std * math.sqrt(TRADING_DAYS)


def drawdown_series(equity: pd.Series) -> pd.Series:
    """How far below its previous peak the portfolio is each day (0 = at a new high)."""
    return equity / equity.cummax() - 1


def max_drawdown(equity: pd.Series) -> float:
    """The worst peak-to-trough fall. -0.3 means at some point you were 30% below a previous high."""
    return drawdown_series(equity).min()


def win_rate(trades: pd.DataFrame | None) -> float:
    """Share of round-trip trades that made money. NaN if there were no trades."""
    if trades is None or len(trades) == 0:
        return float("nan")
    return float((trades["return"] > 0).mean())


def compute_metrics(
    equity: pd.Series, returns: pd.Series, trades: pd.DataFrame | None = None, risk_free_rate: float = 0.0
) -> dict:
    """All metrics in one dict. Pass trades=None for benchmarks that don't trade."""
    return {
        "Total return": total_return(equity),
        "Annual return (CAGR)": cagr(equity),
        "Annual volatility": annual_volatility(returns),
        "Sharpe ratio": sharpe_ratio(returns, risk_free_rate),
        "Max drawdown": max_drawdown(equity),
        "Number of trades": float("nan") if trades is None else len(trades),
        "Win rate": win_rate(trades),
    }


def recent_period(result, share: float = 0.2, min_days: int = 60) -> pd.DataFrame | None:
    """Score the last `share` of a backtest again, on its own, next to the benchmark.

    A result that only looks good over the whole history, and not over its most recent stretch,
    deserves suspicion. Returns None when that stretch is shorter than `min_days` trading days.
    Works with any result that has `equity` and `benchmark_equity` (run_backtest or run_portfolio_backtest).
    """
    span = round((len(result.equity) - 1) * share)
    if span < min_days:
        return None
    strategy = result.equity.iloc[-span - 1 :]
    benchmark = result.benchmark_equity.iloc[-span - 1 :]
    rows = {}
    for name, equity in (("Strategy", strategy), ("Benchmark", benchmark)):
        returns = equity.pct_change().iloc[1:]
        rows[name] = format_metrics(
            {
                "Total return": total_return(equity),
                "Annual return (CAGR)": cagr(equity),
                "Sharpe ratio": sharpe_ratio(returns),
                "Max drawdown": max_drawdown(equity),
            }
        )
    table = pd.DataFrame(rows)
    table.columns.name = f"From {strategy.index[0]:%Y-%m-%d}"
    return table


_PERCENT_METRICS = {"Total return", "Annual return (CAGR)", "Annual volatility", "Max drawdown", "Win rate"}


def format_metrics(metrics: dict) -> dict:
    """Turn raw numbers into readable strings: 0.1234 -> '12.3%', NaN -> '—'."""
    formatted = {}
    for name, value in metrics.items():
        if value is None or (isinstance(value, float) and math.isnan(value)):
            formatted[name] = "—"
        elif name in _PERCENT_METRICS:
            formatted[name] = f"{value:.1%}"
        elif name == "Number of trades":
            formatted[name] = f"{int(value)}"
        else:
            formatted[name] = f"{value:.2f}"
    return formatted
