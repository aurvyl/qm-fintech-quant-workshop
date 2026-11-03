import numpy as np
import pandas as pd
import pytest

from backtester.alpha import (
    alpha_beta,
    find_lookahead,
    format_leaderboard,
    leaderboard,
    monthly_cash_return,
    monthly_returns,
    score,
)
from competition.examples import HOUSE, hold_market, peek_ahead, sector_momentum


def synthetic_closes(days=900, seed=1):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2021-01-01", periods=days, name="Date")
    market = rng.normal(0.0004, 0.01, days)
    other = 0.5 * market + rng.normal(0.0002, 0.008, days)
    return pd.DataFrame(
        {"SPY": 100 * np.cumprod(1 + market), "TLT": 100 * np.cumprod(1 + other)}, index=index
    )


def test_monthly_returns_compound_daily_returns():
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01"])
    months = monthly_returns(pd.Series([0.1, 0.1, -0.5], index=index))
    assert months.tolist() == pytest.approx([0.21, -0.5])


def test_fit_recovers_a_known_line():
    rng = np.random.default_rng(0)
    months = pd.period_range("2010-01", periods=120, freq="M")
    market = rng.normal(0.01, 0.04, 120)
    cash = monthly_cash_return()
    strategy = cash + 0.004 + 1.5 * (market - cash) + rng.normal(0, 0.002, 120)
    daily_index = months.to_timestamp(how="end").normalize()
    fit = alpha_beta(pd.Series(strategy, index=daily_index), pd.Series(market, index=daily_index))
    assert fit.beta == pytest.approx(1.5, abs=0.02)
    assert fit.alpha == pytest.approx(0.004, abs=0.0005)
    assert fit.alpha_annual == pytest.approx(fit.alpha * 12)
    assert fit.alpha_t > 10
    assert fit.months == 120
    assert fit.market_up(0.05) == pytest.approx(fit.alpha + fit.beta * 0.05)


def test_fit_matches_numpy_least_squares():
    closes = synthetic_closes()
    returns = closes.pct_change().iloc[1:]
    fit = alpha_beta(returns["TLT"], returns["SPY"])
    beta, alpha = np.polyfit(fit.points["market"], fit.points["strategy"], 1)
    assert fit.beta == pytest.approx(beta)
    assert fit.alpha == pytest.approx(alpha)


def test_market_against_itself_is_beta_one_alpha_zero():
    returns = synthetic_closes()["SPY"].pct_change().iloc[1:]
    fit = alpha_beta(returns, returns)
    assert fit.beta == pytest.approx(1.0)
    assert fit.alpha == pytest.approx(0.0, abs=1e-12)


def test_too_few_months_is_an_error():
    index = pd.bdate_range("2024-01-01", periods=20)
    with pytest.raises(ValueError, match="at least 3 months"):
        alpha_beta(pd.Series(0.001, index=index), pd.Series(0.001, index=index))


def test_lookahead_check_catches_a_peek_and_passes_honest_rules():
    closes = synthetic_closes()
    assert find_lookahead(peek_ahead, closes)
    assert find_lookahead(hold_market, closes) == []


def test_month_end_rebalancing_is_not_mistaken_for_lookahead():
    sectors = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2021-01-01", periods=700, name="Date")
    closes = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, (700, 9)), axis=0), index=index, columns=sectors)
    assert find_lookahead(sector_momentum, closes) == []


def test_score_disqualifies_cheats_and_leverage():
    closes = synthetic_closes(days=1300)
    cheat = score(peek_ahead, "cheat", closes)
    assert not cheat.qualified and "future prices" in cheat.problem

    levered = score(lambda c: pd.DataFrame({"SPY": 2.0}, index=c.index), "levered", closes)
    assert not levered.qualified and "at most 1" in levered.problem

    idle = score(lambda c: pd.DataFrame({"SPY": 0.0}, index=c.index), "idle", closes)
    assert idle.problem == "never invests"


def test_leaderboard_ranks_by_test_alpha_with_disqualified_last():
    closes = synthetic_closes(days=1300)
    scores = [
        score(hold_market, "market", closes),
        score(lambda c: pd.DataFrame({"TLT": 1.0}, index=c.index), "bonds", closes),
        score(peek_ahead, "cheat", closes),
    ]
    table = leaderboard(scores)
    assert table.iloc[-1]["Strategy"] == "cheat"
    qualified = table[table["Status"] == "OK"]
    assert qualified["Test alpha / year"].is_monotonic_decreasing
    shown = format_leaderboard(table)
    assert shown["Test alpha / year"].str.endswith("%").sum() >= 2


def test_house_strategies_all_return_valid_weights():
    closes = synthetic_closes(days=400)
    for name, strategy in HOUSE.items():
        if "SPY" not in closes or "QQQ" in name or "sector" in name:
            continue
        weights = strategy(closes).fillna(0.0)
        assert (weights >= 0).all().all(), name
        assert (weights.sum(axis=1) <= 1 + 1e-9).all(), name


def test_leaderboard_script_loads_entries_and_reports_broken_files(tmp_path):
    from scripts.leaderboard import load_entries

    (tmp_path / "good.py").write_text(
        "import pandas as pd\nTEAM = 'Good'\ndef strategy(closes):\n    return pd.DataFrame({'SPY': 1.0}, index=closes.index)\n"
    )
    (tmp_path / "broken.py").write_text("this is not python\n")
    (tmp_path / "_helper.py").write_text("")
    entries = {name: (fn, problem) for name, fn, problem in load_entries(tmp_path)}
    assert set(entries) == {"Good", "broken"}
    assert callable(entries["Good"][0]) and entries["Good"][1] is None
    assert entries["broken"][0] is None and "could not load" in entries["broken"][1]


def test_leaderboard_script_refuses_entries_that_load_their_own_data(tmp_path):
    from scripts.leaderboard import load_entries

    (tmp_path / "sneaky.py").write_text(
        "from backtester import load_closes\nTEAM = 'Sneaky'\n"
        "def strategy(closes):\n    real = load_closes(['SPY'])\n    return (real.shift(-1) > real).astype(float)\n"
    )
    [(name, fn, problem)] = load_entries(tmp_path)
    assert fn is None and "shift(-" in problem


def test_explainer_embeds_the_interactive_page():
    pytest.importorskip("IPython")
    from backtester.alpha import explainer

    shown = explainer(height=900).data
    assert shown.startswith("<iframe srcdoc=") and "height:900px" in shown
    assert "Move the line" in shown and "&lt;script&gt;" in shown
