"""Backtest a portfolio of several assets held in chosen weights.

run_backtest() trades one asset, fully in or out. run_portfolio_backtest() holds
many assets at once. A weight is the share of the portfolio in each asset:
0.25 means a quarter of the money. Whatever the weights leave over sits in cash.

The rules are the same as the one-asset engine:
  1. Weights decided at day t's close are held from day t+1 (no look-ahead).
  2. Every change of weight pays a trading cost on the size of the change (turnover).
  3. Between rebalances the holdings drift with prices, as a real portfolio does.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS, compute_metrics, format_metrics

WEIGHT_TOLERANCE = 1e-9


@dataclass
class PortfolioResult:
    """Everything a portfolio backtest produces."""

    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    benchmark_equity: pd.Series
    metrics: dict
    benchmark_metrics: dict
    benchmark_name: str

    def summary(self) -> pd.DataFrame:
        """Side-by-side table: the portfolio vs the benchmark, formatted for reading."""
        return pd.DataFrame(
            {"Portfolio": format_metrics(self.metrics), self.benchmark_name: format_metrics(self.benchmark_metrics)}
        ).drop(index=["Number of trades", "Win rate"])

    def plot(self):
        """Portfolio value and drawdown charts. Returns a matplotlib Figure."""
        from .plotting import plot_result

        return plot_result(self)


def daily_cash_return(cash_rate: float) -> float:
    """Turn a yearly cash interest rate (0.04 = 4% a year) into the rate earned each trading day."""
    return (1 + cash_rate) ** (1 / TRADING_DAYS) - 1


def _clean_weights(weights: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("weights must be a DataFrame: one row per day, one column per ticker.")
    unknown = sorted(set(weights.columns) - set(closes.columns))
    if unknown:
        raise ValueError(f"No prices for {unknown}. Every weight column needs a price column.")
    if not weights.index.isin(closes.index).any():
        raise ValueError("The weights' dates don't match the price dates. Build weights from the same prices.")
    aligned = weights.reindex(index=closes.index, columns=closes.columns).astype(float).fillna(0.0)
    if (aligned < -WEIGHT_TOLERANCE).any().any():
        raise ValueError("Weights must be 0 or more: this engine is long only.")
    if (aligned.sum(axis=1) > 1 + 1e-6).any():
        worst = aligned.sum(axis=1).max()
        raise ValueError(f"Weights on a day must add up to at most 1 (100%); found {worst:.3f}.")
    return aligned.clip(lower=0.0)


def run_portfolio_backtest(
    closes: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float = 5.0,
    cash_rate: float = 0.0,
    start_cash: float = 10_000.0,
    rebalance: pd.Series | None = None,
    benchmark: str | None = None,
) -> PortfolioResult:
    """Simulate holding `weights` of each asset in `closes`.

    closes:    DataFrame of closing prices, one column per ticker (e.g. from load_closes).
    weights:   DataFrame of target weights on the same dates and tickers, decided at each close.
               NaN counts as 0. Each row must add up to at most 1; the rest is cash.
    cost_bps:  trading cost in basis points, charged on the total size of every rebalance.
    cash_rate: yearly interest earned on cash (0.04 = 4%). The workshop default is 0.
    rebalance: optional True/False Series of days to trade. By default the portfolio trades back to
               its target weights whenever the target changes, and otherwise lets the holdings drift.
    benchmark: optional ticker to compare with. By default: equal parts of every ticker, bought on day 1 and held.
    """
    closes = closes.sort_index()
    if len(closes) < 2:
        raise ValueError("Need at least 2 days of prices to run a backtest. Check your date range.")
    if closes.iloc[0].isna().any():
        missing = closes.columns[closes.iloc[0].isna()].tolist()
        raise ValueError(f"No price on the first day for {missing}. Start later or drop those tickers.")

    target = _clean_weights(weights, closes)
    asset_returns = closes.ffill().pct_change().fillna(0.0)
    if rebalance is None:
        trade_days = target.diff().abs().sum(axis=1).gt(WEIGHT_TOLERANCE)
        trade_days.iloc[0] = target.iloc[0].sum() > 0
    else:
        trade_days = rebalance.reindex(closes.index).fillna(False).astype(bool)
    cash_daily = daily_cash_return(cash_rate)

    tickers = closes.columns
    held = np.zeros(len(tickers))
    held_rows, turnover_rows, daily_rows = [], [], []
    pending = None
    for i, day in enumerate(closes.index):
        cost = 0.0
        turnover = 0.0
        if pending is not None:
            turnover = float(np.abs(pending - held).sum())
            cost = turnover * cost_bps / 10_000
            held = pending
            pending = None
        held_rows.append(held.copy())
        turnover_rows.append(turnover)
        if i == 0:
            daily_rows.append(0.0)
        else:
            r = asset_returns.iloc[i].to_numpy()
            cash = 1.0 - held.sum()
            gross = float((held * r).sum() + cash * cash_daily)
            daily = gross - cost
            daily_rows.append(daily)
            growth = held * (1 + r)
            total = growth.sum() + cash * (1 + cash_daily)
            held = growth / total if total > 0 else np.zeros(len(tickers))
        if trade_days.iloc[i]:
            pending = target.iloc[i].to_numpy()

    daily = pd.Series(daily_rows, index=closes.index)
    equity = start_cash * (1 + daily).cumprod()
    held_weights = pd.DataFrame(held_rows, index=closes.index, columns=tickers)

    if benchmark is None:
        benchmark_equity = start_cash * (closes.ffill() / closes.iloc[0]).mean(axis=1)
        benchmark_name = "Equal-weight buy & hold"
    else:
        if benchmark not in closes.columns:
            raise ValueError(f"Benchmark {benchmark!r} must be one of the price columns.")
        benchmark_equity = start_cash * closes[benchmark].ffill() / closes[benchmark].iloc[0]
        benchmark_name = f"{benchmark} buy & hold"

    returns = daily.iloc[1:]
    benchmark_returns = benchmark_equity.pct_change().iloc[1:]
    return PortfolioResult(
        equity=equity,
        returns=returns,
        weights=held_weights,
        turnover=pd.Series(turnover_rows, index=closes.index),
        benchmark_equity=benchmark_equity,
        metrics=compute_metrics(equity, returns, risk_free_rate=cash_rate),
        benchmark_metrics=compute_metrics(benchmark_equity, benchmark_returns, risk_free_rate=cash_rate),
        benchmark_name=benchmark_name,
    )


def month_ends(index: pd.DatetimeIndex) -> pd.Series:
    """True on the last trading day of each month: a common day to rebalance."""
    dates = pd.Series(index, index=index)
    return dates.groupby(dates.dt.to_period("M")).transform("max").eq(dates)


def top_n_weights(scores: pd.DataFrame, n: int, on: pd.Series | None = None) -> pd.DataFrame:
    """Equal weights in the `n` assets with the highest score each day (ties broken A to Z).

    scores: DataFrame of any score, higher is better (e.g. 6-month momentum). NaN never gets picked.
    on:     optional True/False Series of decision days (e.g. month_ends). Between them the choice is kept.
    """
    ranks = scores.rank(axis=1, ascending=False, method="first")
    picked = ranks.le(n) & scores.notna()
    counts = picked.sum(axis=1).replace(0, np.nan)
    weights = picked.div(counts, axis=0).fillna(0.0)
    if on is not None:
        keep = on.reindex(scores.index).fillna(False).astype(bool)
        weights = weights.where(keep).ffill().fillna(0.0)
    return weights


def volatility_target(weights: pd.DataFrame, closes: pd.DataFrame, target: float = 0.10, lookback: int = 20) -> pd.DataFrame:
    """Scale weights down when the portfolio has been bumpy, so its risk stays near `target` a year.

    exposure = min(1, target / recent volatility), measured on the last `lookback` days of the
    portfolio the weights describe. Never leverages up: in calm markets exposure stays at 1.
    """
    returns = closes.pct_change()
    portfolio_returns = (weights.shift(1).fillna(0.0) * returns.fillna(0.0)).sum(axis=1)
    recent_vol = portfolio_returns.rolling(lookback).std() * np.sqrt(TRADING_DAYS)
    exposure = (target / recent_vol).clip(upper=1.0).fillna(1.0)
    exposure = exposure.where(recent_vol > 0, 1.0)
    return weights.mul(exposure, axis=0)
