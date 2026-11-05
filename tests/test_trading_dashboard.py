import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from backtester import load_prices, run_backtest
from projects.trading_dashboard.checks import run_check

PROJECT = Path(__file__).resolve().parent.parent / "projects" / "trading_dashboard"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


solution = load_module(PROJECT / "solution" / "analytics.py", "dashboard_solution_for_tests")
FUNCTIONS = [
    "to_returns", "equity_curve", "drawdown", "annual_return", "annual_volatility", "sharpe",
    "sortino", "hit_rate", "rolling_sharpe", "rolling_volatility", "capm", "tearsheet",
]


def test_returns_equity_and_drawdown_by_hand():
    prices = pd.Series([100.0, 110.0, 99.0, 121.0])
    returns = solution.to_returns(prices)
    assert np.allclose(returns, [0.10, -0.10, 121 / 99 - 1])
    equity = solution.equity_curve(returns)
    assert np.allclose(equity, [1.10, 0.99, 1.21])
    assert np.allclose(solution.drawdown(equity), [0.0, -0.10, 0.0])


def test_tearsheet_matches_the_backtesters_buy_and_hold_numbers():
    prices = load_prices("SPY")
    benchmark = run_backtest(prices, pd.Series(0, index=prices.index)).benchmark_metrics
    stats = solution.tearsheet(prices["Close"])
    assert stats["Annual return"] == pytest.approx(benchmark["Annual return (CAGR)"])
    assert stats["Annual volatility"] == pytest.approx(benchmark["Annual volatility"])
    assert stats["Sharpe ratio"] == pytest.approx(benchmark["Sharpe ratio"])
    assert stats["Max drawdown"] == pytest.approx(benchmark["Max drawdown"])


def test_capm_recovers_a_known_line():
    market = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 500))
    result = solution.capm(0.0001 + 1.5 * market, market)
    assert result["Beta"] == pytest.approx(1.5)
    assert result["Alpha"] == pytest.approx(0.0001 * 252)
    assert result["R squared"] == pytest.approx(1.0)


@pytest.mark.parametrize("name", FUNCTIONS)
def test_checks_pass_the_reference_solution(name):
    passed, message = run_check(getattr(solution, name))
    assert passed, message


def test_checks_explain_an_unfinished_function():
    def to_returns(prices):
        returns = ...
        return returns

    passed, message = run_check(to_returns)
    assert not passed
    assert "Replace the ..." in message and "Hint" in message


def test_checks_catch_a_wrong_answer_and_an_error():
    def sharpe(returns):
        return returns.mean() / returns.std()  # forgot to annualise

    def drawdown(equity):
        return equity.cumax()  # typo

    assert not run_check(sharpe)[0]
    passed, message = run_check(drawdown)
    assert not passed and "AttributeError" in message


def run_dashboard(app_path, analytics_module):
    """Run an app with a given analytics module (both apps do `import analytics`)."""
    sys.modules["analytics"] = analytics_module
    try:
        return AppTest.from_file(str(app_path), default_timeout=60).run()
    finally:
        sys.modules.pop("analytics", None)


def test_solution_app_shows_every_number():
    at = run_dashboard(PROJECT / "solution" / "app.py", solution)
    assert not at.exception
    assert len(at.metric) == 9  # six headline numbers + alpha, beta, R squared
    assert [tab.label for tab in at.tabs] == ["Performance", "Risk", "CAPM", "Data"]


def test_starter_app_runs_once_analytics_is_filled_in():
    at = run_dashboard(PROJECT / "app.py", solution)
    assert not at.exception
    assert len(at.metric) == 1
    assert len(at.info) == 5  # one placeholder per TODO


def test_starter_analytics_says_what_to_do():
    starter = load_module(PROJECT / "analytics.py", "dashboard_starter_for_tests")
    with pytest.raises(NotImplementedError, match="the take-home notebook"):
        starter.to_returns(pd.Series([1.0, 2.0]))
