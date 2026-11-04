import math

import numpy as np
import pandas as pd
import pytest

from backtester import run_backtest
from backtester.metrics import recent_period, sharpe_ratio
from backtester.portfolio import (
    daily_cash_return,
    month_ends,
    run_portfolio_backtest,
    top_n_weights,
    volatility_target,
)


def closes_of(**columns):
    length = len(next(iter(columns.values())))
    index = pd.bdate_range("2024-01-01", periods=length, name="Date")
    return pd.DataFrame({k: [float(x) for x in v] for k, v in columns.items()}, index=index)


def constant(closes, **weights):
    return pd.DataFrame({t: weights.get(t, 0.0) for t in closes.columns}, index=closes.index)


def test_fully_in_one_asset_matches_the_single_asset_engine():
    closes = closes_of(A=[100, 102, 99, 105, 110])
    result = run_portfolio_backtest(closes, constant(closes, A=1.0), cost_bps=10)
    single = run_backtest(closes.rename(columns={"A": "Close"}), [1] * 5, cost_bps=10)
    assert result.equity.tolist() == pytest.approx(single.equity.tolist())


def test_weights_are_held_from_the_next_day():
    closes = closes_of(A=[100, 200, 200])
    weights = pd.DataFrame({"A": [0.0, 1.0, 1.0]}, index=closes.index)
    result = run_portfolio_backtest(closes, weights, cost_bps=0)
    assert result.equity.iloc[-1] == pytest.approx(10_000)


def test_half_and_half_without_rebalancing_drifts_like_buy_and_hold():
    closes = closes_of(A=[100, 110, 121], B=[100, 90, 81])
    result = run_portfolio_backtest(closes, constant(closes, A=0.5, B=0.5), cost_bps=0)
    assert result.equity.iloc[-1] == pytest.approx(10_000 * (0.5 * 1.21 + 0.5 * 0.81))
    assert result.equity.tolist() == pytest.approx(result.benchmark_equity.tolist())
    assert result.weights["A"].iloc[-1] == pytest.approx(0.55)


def test_monthly_rebalancing_trades_back_to_target_and_pays_costs():
    closes = closes_of(A=[100, 110, 121, 121], B=[100, 100, 100, 100])
    weights = constant(closes, A=0.5, B=0.5)
    every_day = pd.Series(True, index=closes.index)
    rebalanced = run_portfolio_backtest(closes, weights, cost_bps=0, rebalance=every_day)
    assert rebalanced.weights["A"].iloc[-1] == pytest.approx(0.5)
    with_costs = run_portfolio_backtest(closes, weights, cost_bps=10, rebalance=every_day)
    assert with_costs.equity.iloc[-1] < rebalanced.equity.iloc[-1]
    assert with_costs.turnover.iloc[1] == pytest.approx(1.0)


def test_leftover_weight_earns_the_cash_rate():
    closes = closes_of(A=[100] * 253)
    result = run_portfolio_backtest(closes, constant(closes, A=0.0), cash_rate=0.04, cost_bps=0)
    assert result.equity.iloc[-1] == pytest.approx(10_400)


def test_single_asset_engine_cash_rate_applies_only_out_of_the_market():
    prices = closes_of(Close=[100] * 253)
    flat = run_backtest(prices, [0] * 253, cost_bps=0, cash_rate=0.04)
    long = run_backtest(prices, [1] * 253, cost_bps=0, cash_rate=0.04)
    assert flat.equity.iloc[-1] == pytest.approx(10_400)
    assert long.equity.iloc[-1] == pytest.approx(10_000)


def test_bad_weights_raise_clear_errors():
    closes = closes_of(A=[100, 101], B=[100, 101])
    with pytest.raises(ValueError, match="at most 1"):
        run_portfolio_backtest(closes, constant(closes, A=0.8, B=0.8))
    with pytest.raises(ValueError, match="long only"):
        run_portfolio_backtest(closes, constant(closes, A=-0.5))
    with pytest.raises(ValueError, match="No prices"):
        run_portfolio_backtest(closes, pd.DataFrame({"C": [1.0, 1.0]}, index=closes.index))
    with pytest.raises(TypeError):
        run_portfolio_backtest(closes, [0.5, 0.5])


def test_top_n_weights_picks_the_best_scores_and_keeps_them_between_decision_days():
    index = pd.bdate_range("2024-01-29", periods=4)
    scores = pd.DataFrame({"A": [3, 1, 1, 1], "B": [2, 3, 3, 3], "C": [1, 2, 2, float("nan")]}, index=index, dtype=float)
    daily = top_n_weights(scores, 2)
    assert daily.iloc[0].tolist() == [0.5, 0.5, 0.0]
    assert daily.iloc[3].tolist() == [0.5, 0.5, 0.0]
    on = pd.Series([True, False, False, False], index=index)
    kept = top_n_weights(scores, 2, on=on)
    assert (kept == kept.iloc[0]).all().all()


def test_month_ends_marks_the_last_trading_day_of_each_month():
    index = pd.bdate_range("2024-01-29", "2024-02-02")
    assert month_ends(index).tolist() == [False, False, True, False, True]


def test_volatility_target_only_scales_down():
    closes = closes_of(A=[100 * (1.03 if i % 2 else 0.97) ** (i % 2) for i in range(60)])
    weights = constant(closes, A=1.0)
    scaled = volatility_target(weights, closes, target=0.10)
    assert scaled["A"].max() <= 1.0
    assert scaled["A"].iloc[-1] < 0.5
    calm = closes_of(A=[100 + 0.01 * i for i in range(60)])
    assert volatility_target(constant(calm, A=1.0), calm, target=0.10)["A"].iloc[-1] == pytest.approx(1.0)


def test_sharpe_is_measured_above_the_cash_rate():
    returns = pd.Series([0.001, -0.0005] * 50)
    assert sharpe_ratio(returns, 0.05) < sharpe_ratio(returns)
    assert daily_cash_return(0.0) == 0.0


def test_recent_period_scores_the_last_fifth_on_its_own():
    closes = closes_of(Close=list(range(100, 400)))
    result = run_backtest(closes, [1] * 300, cost_bps=0)
    table = recent_period(result)
    assert list(table.columns) == ["Strategy", "Benchmark"]
    assert table.columns.name == f"From {closes.index[-61]:%Y-%m-%d}"
    assert table.loc["Total return", "Strategy"] == f"{399 / 339 - 1:.1%}"
    short = run_backtest(closes.iloc[:100], [1] * 100)
    assert recent_period(short) is None


def test_recent_period_sharpe_is_lower_when_measured_above_cash():
    rng = np.random.default_rng(0)
    closes = closes_of(Close=list(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 400))))
    result = run_backtest(closes, [1] * 400, cost_bps=0)
    plain = float(recent_period(result).loc["Sharpe ratio", "Strategy"])
    above_cash = float(recent_period(result, risk_free_rate=0.05).loc["Sharpe ratio", "Strategy"])
    assert above_cash < plain
