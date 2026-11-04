"""Self-checks for the take-home notebook. Call check(your_function) to see if it's right.

Each check runs your function on a tiny made-up example and on real prices,
and compares the answer with the reference solution in solution/analytics.py.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

_SOLUTION_PATH = Path(__file__).resolve().parent / "solution" / "analytics.py"

# A tiny example you can check by hand: +10%, -10%, +21.2%, -5%.
_TINY_PRICES = pd.Series(
    [100.0, 110.0, 99.0, 120.0, 114.0], index=pd.date_range("2024-01-01", periods=5, freq="B"), name="Close"
)

HINTS = {
    "to_returns": "Use .pct_change(), then .dropna() to remove the empty first day.",
    "equity_curve": "Add 1 to every return, then multiply them together day by day with .cumprod().",
    "drawdown": "Divide the equity by its highest value so far (.cummax()), then subtract 1.",
    "annual_return": "Multiply all (1 + return) together with .prod(), raise to the power 252 / len(returns), subtract 1.",
    "annual_volatility": "Daily standard deviation (.std()) times the square root of 252 (np.sqrt(252)).",
    "sharpe": "Average daily return (.mean()) divided by daily standard deviation (.std()), times np.sqrt(252).",
    "sortino": "Same as Sharpe, but the standard deviation only of the negative returns: returns[returns < 0].std().",
    "hit_rate": "(returns > 0) gives True/False per day; the .mean() of that is the share of True.",
    "rolling_sharpe": "Use returns.rolling(window).mean() and returns.rolling(window).std() in the Sharpe formula.",
    "rolling_volatility": "returns.rolling(window).std(), then scale to a year with np.sqrt(252).",
    "capm": "Line up the dates with pd.concat([...], axis=1).dropna(). Beta = cov / var, "
    "alpha = mean(asset) - beta * mean(benchmark), times 252. R squared = corr ** 2.",
    "tearsheet": "Call your earlier functions and return a dict with the six names shown in the notebook.",
}


def _solution():
    spec = importlib.util.spec_from_file_location("trading_dashboard_solution", _SOLUTION_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _examples():
    """(name, prices, benchmark prices) to test on: the tiny example first, then real data."""
    from backtester import load_prices

    tiny_benchmark = pd.Series([50.0, 52.0, 51.0, 53.0, 52.5], index=_TINY_PRICES.index, name="Close")
    return [
        ("the tiny example", _TINY_PRICES, tiny_benchmark),
        ("AAPL vs SPY", load_prices("AAPL")["Close"], load_prices("SPY")["Close"]),
    ]


def _arguments(name, solution, prices, benchmark):
    """The inputs a function takes, built from prices with the reference functions."""
    returns = solution.to_returns(prices)
    if name in ("to_returns", "tearsheet"):
        return (prices,)
    if name == "drawdown":
        return (solution.equity_curve(returns),)
    if name in ("rolling_sharpe", "rolling_volatility") and len(returns) < 10:
        return (returns, 2)  # the tiny example is too short for the default window
    if name == "capm":
        return (returns, solution.to_returns(benchmark))
    return (returns,)


def _same(got, expected):
    """True if got matches expected; otherwise a short reason why not."""
    if got is Ellipsis:
        return "it is still ... (replace it with your code)"
    if isinstance(expected, pd.Series):
        if not isinstance(got, pd.Series):
            return f"expected a pandas Series, got {type(got).__name__}"
        if len(got) != len(expected):
            return f"expected {len(expected)} values, got {len(got)}"
        if not np.allclose(got.to_numpy(dtype=float), expected.to_numpy(dtype=float), equal_nan=True):
            return f"expected {_short(expected)}, got {_short(got)}"
        return True
    if isinstance(expected, dict):
        if not isinstance(got, dict):
            return f"expected a dict, got {type(got).__name__}"
        missing = [key for key in expected if key not in got]
        if missing:
            return f"the dict is missing {missing}"
        for key, value in expected.items():
            verdict = _same(got[key], value)
            if verdict is not True:
                return f"{key!r}: {verdict}"
        return True
    try:
        if math.isclose(float(got), float(expected), rel_tol=1e-6, abs_tol=1e-12):
            return True
    except (TypeError, ValueError):
        return f"expected a number, got {type(got).__name__}"
    return f"expected {float(expected):.4f}, got {float(got):.4f}"


def _short(series):
    values = ", ".join("nan" if pd.isna(v) else f"{v:.4f}" for v in series.head(4))
    return f"[{values}{', ...' if len(series) > 4 else ''}]"


def run_check(function):
    """(passed, message) for one of your notebook functions."""
    name = function.__name__
    solution = _solution()
    if not hasattr(solution, name):
        return False, f"❓ No check for a function called {name!r}. Keep the names from the notebook."
    hint = f"\n   Hint: {HINTS[name]}" if name in HINTS else ""

    for label, prices, benchmark in _examples():
        args = _arguments(name, solution, prices, benchmark)
        try:
            got = function(*args)
        except Exception as error:  # show any mistake as a message, not a crash
            return False, f"❌ {name} raised {type(error).__name__}: {error}{hint}"
        if got is Ellipsis or got is None:
            return False, f"❌ {name} returned {got!r}. Replace the ... with your code and return the answer.{hint}"
        verdict = _same(got, getattr(solution, name)(*args))
        if verdict is not True:
            return False, f"❌ {name} is not right yet on {label}: {verdict}{hint}"
    return True, f"✅ {name} is correct!"


def check(function):
    """Print whether your function gives the right answer."""
    print(run_check(function)[1])
