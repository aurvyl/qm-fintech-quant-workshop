"""Tools for testing a backtest honestly: toy strategies with a known answer, and fair ways to test.

The toy strategies have no skill at all, so whatever a backtest says about them is luck, costs or a bug:
  coin_flip_positions   hold or sit in cash on a coin flip, every day
  random_strategies     many coin-flip strategies that change their mind rarely
  peek_positions        hold on the days the price rose: only knowable after the close (a cheat)

The fair tests:
  period_sharpe         the Sharpe ratio of a position series over one slice of time
  best_of_many          pick the best of many strategies on one period, then score that pick on a later one
  walk_forward          pick settings on a training window, test on the next, then slide forward and repeat
"""

import numpy as np
import pandas as pd

from .engine import run_backtest


def coin_flip_positions(prices: pd.DataFrame, hold_prob: float = 0.5, seed: int = 0) -> pd.Series:
    """Each day, flip a coin: 1 (hold) with probability hold_prob, otherwise 0 (cash)."""
    rng = np.random.default_rng(seed)
    flips = (rng.random(len(prices)) < hold_prob).astype(int)
    return pd.Series(flips, index=prices.index, name="position")


def random_strategies(prices: pd.DataFrame, n: int = 200, switch_prob: float = 0.02, seed: int = 0) -> pd.DataFrame:
    """n strategies with no skill. Each day, each one switches between hold and cash with probability switch_prob."""
    if n < 1:
        raise ValueError("Need at least one strategy.")
    rng = np.random.default_rng(seed)
    switches = rng.random((len(prices), n)) < switch_prob
    start = rng.random(n) < 0.5
    state = (np.cumsum(switches, axis=0) + start.astype(int)) % 2
    return pd.DataFrame(state.astype(int), index=prices.index, columns=[f"random_{i:03d}" for i in range(n)])


def peek_positions(prices: pd.DataFrame) -> pd.Series:
    """A cheat: hold on the days the price rose. You only know that after the close, so this cannot be traded."""
    return (prices["Close"].pct_change() > 0).astype(int)


def period_sharpe(prices: pd.DataFrame, positions: pd.Series, start=None, end=None, cost_bps: float = 5.0,
                  cash_rate: float = 0.0) -> float:
    """Sharpe ratio of the positions over one slice of time. Positions are built on the full history first,
    which is safe for indicators that only look backwards."""
    window = slice(start, end)
    return run_backtest(prices.loc[window], positions.loc[window], cost_bps=cost_bps, cash_rate=cash_rate).metrics["Sharpe ratio"]


def best_of_many(prices: pd.DataFrame, positions: pd.DataFrame, split: str = "2021-12-31",
                 cost_bps: float = 5.0) -> pd.DataFrame:
    """Score every strategy before and after `split`, best first by the earlier period.

    The honest question: does the winner of the first period still win in the second?
    """
    rows = []
    for name in positions.columns:
        rows.append({
            "strategy": name,
            "before Sharpe": period_sharpe(prices, positions[name], None, split, cost_bps),
            "after Sharpe": period_sharpe(prices, positions[name], pd.Timestamp(split) + pd.Timedelta(days=1), None, cost_bps),
        })
    return pd.DataFrame(rows).sort_values("before Sharpe", ascending=False).reset_index(drop=True)


def walk_forward(prices: pd.DataFrame, make_positions, settings: list[dict], train_days: int = 756,
                 test_days: int = 252, cost_bps: float = 5.0) -> pd.DataFrame:
    """Walk forward: tune on the past, test on what comes next, slide on, repeat.

    make_positions(prices, **setting) must return a position series built from the prices alone.
    settings is a list of dicts such as [{"fast": 20, "slow": 100}, {"fast": 50, "slow": 200}].
    Each row is one fold: the setting that did best on the training window, and how it did on the
    test window that follows, next to buy and hold on the same test window. Test windows never overlap.
    """
    if not settings:
        raise ValueError("Need at least one setting to choose from.")
    candidates = [(setting, make_positions(prices, **setting)) for setting in settings]
    rows = []
    start = 0
    while start + train_days + test_days <= len(prices):
        train = prices.index[start : start + train_days]
        test = prices.index[start + train_days : start + train_days + test_days]
        scored = [(period_sharpe(prices, pos, train[0], train[-1], cost_bps), setting, pos) for setting, pos in candidates]
        best_sharpe, best_setting, best_pos = max(scored, key=lambda item: (np.nan_to_num(item[0], nan=-np.inf)))
        result = run_backtest(prices.loc[test[0] : test[-1]], best_pos.loc[test[0] : test[-1]], cost_bps=cost_bps)
        rows.append({
            "train from": train[0], "test from": test[0], "test to": test[-1],
            "chosen": ", ".join(f"{k}={v}" for k, v in best_setting.items()),
            "train Sharpe": best_sharpe,
            "test Sharpe": result.metrics["Sharpe ratio"],
            "test return": result.metrics["Total return"],
            "buy and hold test return": result.benchmark_metrics["Total return"],
        })
        start += test_days
    return pd.DataFrame(rows)
