import numpy as np
import pandas as pd
import pytest

from backtester import run_backtest
from backtester.honesty import (
    best_of_many,
    coin_flip_positions,
    peek_positions,
    period_sharpe,
    random_strategies,
    walk_forward,
)
from strategies import ma_crossover


@pytest.fixture
def prices():
    rng = np.random.default_rng(1)
    index = pd.bdate_range("2016-01-01", periods=2000, name="Date")
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, len(index)))
    return pd.DataFrame({"Close": close}, index=index)


def test_coin_flip_holds_about_half_the_days_and_is_repeatable(prices):
    first = coin_flip_positions(prices, seed=3)
    again = coin_flip_positions(prices, seed=3)
    assert first.equals(again)
    assert set(first.unique()) <= {0, 1}
    assert 0.45 < first.mean() < 0.55


def test_costs_eat_a_coin_flip_strategy(prices):
    flips = coin_flip_positions(prices, seed=0)
    free = run_backtest(prices, flips, cost_bps=0).metrics["Total return"]
    costly = run_backtest(prices, flips, cost_bps=20).metrics["Total return"]
    assert costly < free


def test_random_strategies_have_the_requested_shape_and_switch_rarely(prices):
    table = random_strategies(prices, n=20, switch_prob=0.02, seed=0)
    assert table.shape == (len(prices), 20)
    changes = table.diff().abs().iloc[1:].sum().mean()
    assert 15 < changes < 70
    with pytest.raises(ValueError):
        random_strategies(prices, n=0)


def test_peek_positions_look_great_only_without_the_one_day_shift(prices):
    peek = peek_positions(prices)
    daily = prices["Close"].pct_change()
    cheating = ((1 + peek * daily.fillna(0)).prod()) - 1
    honest = run_backtest(prices, peek, cost_bps=0).metrics["Total return"]
    assert cheating > 10 * max(honest, 0.01)


def test_best_of_many_sorts_by_the_earlier_period(prices):
    table = best_of_many(prices, random_strategies(prices, n=12, seed=2), split="2019-06-30")
    assert table["before Sharpe"].is_monotonic_decreasing
    assert list(table.columns) == ["strategy", "before Sharpe", "after Sharpe"]


def test_period_sharpe_uses_only_the_slice(prices):
    positions = pd.Series(1, index=prices.index)
    early = period_sharpe(prices, positions, None, "2017-12-31")
    late = period_sharpe(prices, positions, "2019-01-01", None)
    assert early != late


def test_walk_forward_test_windows_never_overlap_and_follow_training(prices):
    settings = [{"fast": 10, "slow": 50}, {"fast": 20, "slow": 100}]
    table = walk_forward(prices, lambda p, fast, slow: ma_crossover(p, fast=fast, slow=slow), settings,
                         train_days=500, test_days=250)
    assert len(table) >= 3
    assert (table["test from"] > table["train from"]).all()
    assert (table["test from"].iloc[1:].to_numpy() > table["test to"].iloc[:-1].to_numpy()).all()
    assert table["chosen"].isin(["fast=10, slow=50", "fast=20, slow=100"]).all()


def test_walk_forward_needs_a_setting(prices):
    with pytest.raises(ValueError):
        walk_forward(prices, lambda p: p["Close"] * 0, [])
