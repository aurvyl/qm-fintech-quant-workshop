import numpy as np
import pandas as pd
import pytest

from backtester import risk


def random_closes(seed=0, days=200, tickers=("RISKY", "CALM"), vols=(0.03, 0.003)):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=days, name="Date")
    data = {t: 100 * np.cumprod(1 + rng.normal(0.0003, v, days)) for t, v in zip(tickers, vols)}
    return pd.DataFrame(data, index=index)


def test_lines_start_before_the_limit_is_hit():
    lines = risk.lines_from_limits(risk.Limits(max_drawdown=0.10, max_daily_loss=0.02))
    assert lines.rotate_at == pytest.approx(0.05)
    assert lines.emergency_at == pytest.approx(0.08)
    assert lines.var_budget == pytest.approx(0.02)
    with pytest.raises(ValueError):
        risk.lines_from_limits(risk.Limits(max_drawdown=10))


def test_one_day_var_matches_the_hand_formula():
    returns = random_closes().pct_change().dropna()
    weights = pd.Series({"RISKY": 0.6, "CALM": 0.4})
    window = returns.iloc[-60:]
    expected = 2.326 * np.sqrt(weights @ window.cov() @ weights)
    assert risk.one_day_var(weights, returns) == pytest.approx(expected)
    assert risk.one_day_var(pd.Series({"RISKY": 0.0}), returns) == 0.0


def test_historical_var_and_expected_shortfall():
    returns = pd.Series(np.linspace(-0.05, 0.05, 101))
    assert risk.historical_var(returns, 0.99) == pytest.approx(0.049)
    assert risk.expected_shortfall(returns, 0.99) == pytest.approx(0.0495)


def test_risk_contributions_add_up_to_one_and_blame_the_risky_asset():
    returns = random_closes().pct_change().dropna()
    shares = risk.risk_contributions(pd.Series({"RISKY": 0.5, "CALM": 0.5}), returns)
    assert shares.sum() == pytest.approx(1.0)
    assert shares["RISKY"] > 0.9


def test_check_plan_lists_every_broken_rule():
    returns = random_closes().pct_change().dropna()
    lines = risk.lines_from_limits(risk.Limits(0.10, 0.02))
    allowed = ["RISKY", "CALM"]
    assert risk.check_plan({"CALM": 0.4}, returns, lines, allowed).passed
    check = risk.check_plan({"RISKY": 0.5, "CALM": 0.62, "XYZ": 0.1}, returns, lines, allowed)
    assert not check.passed
    text = " ".join(check.reasons)
    assert "Unknown tickers" in text and "above 100%" in text and "40% cap" in text and "5% steps" in text
    too_risky = risk.check_plan({"RISKY": 0.4}, returns, lines, allowed)
    assert any("VaR" in r for r in too_risky.reasons)


def test_evaluate_names_the_line_that_was_crossed():
    lines = risk.lines_from_limits(risk.Limits(0.10, 0.02))
    assert risk.evaluate(-0.01, 0.0, 0.01, lines) == (None, [])
    assert risk.evaluate(-0.06, 0.0, 0.01, lines)[0] == "rotate"
    assert risk.evaluate(-0.09, 0.0, 0.01, lines)[0] == "emergency"
    kind, reasons = risk.evaluate(0.0, -0.03, 0.03, lines)
    assert kind == "rotate" and len(reasons) == 2


def test_rotate_manager_moves_risk_into_the_calm_asset_and_passes_the_check():
    closes = random_closes()
    returns = closes.pct_change().iloc[-60:]
    lines = risk.lines_from_limits(risk.Limits(0.10, 0.02))
    held = pd.Series({"RISKY": 0.4, "CALM": 0.0})
    nav = pd.Series([1.0, 0.97])
    context = risk.build_context(closes.index[-1], held, returns, nav, lines, "rotate", [], ["CALM"], held)
    plan = risk.rotate_manager(context)
    assert plan["RISKY"] < 0.4
    assert plan["CALM"] > 0
    assert risk.check_plan(plan, returns, lines, ["RISKY", "CALM"]).passed


def test_buy_and_hold_never_trades_and_matches_its_benchmark():
    closes = random_closes()
    result = risk.run_guarded(closes, {"RISKY": 0.5, "CALM": 0.5}, manager=None)
    assert len(result.log) == 0
    assert result.equity.tolist() == pytest.approx(result.benchmark_equity.tolist())


def test_guarded_portfolio_falls_less_than_buy_and_hold_in_a_crash():
    closes = random_closes(days=200)
    crash = closes.copy()
    crash.loc[crash.index[120]:, "RISKY"] *= 0.97 ** np.arange(1, len(crash) - 119)
    limits = risk.Limits(0.10, 0.02)
    hold = risk.run_guarded(crash, {"RISKY": 0.4, "CALM": 0.2}, limits, manager=None)
    guarded = risk.run_guarded(crash, {"RISKY": 0.4, "CALM": 0.2}, limits, calm=["CALM"])
    assert guarded.metrics["Max drawdown"] > hold.metrics["Max drawdown"]
    assert (guarded.log["action"] == "rotate").any()


def test_cash_exit_sells_everything_once_and_stays_out():
    closes = random_closes(days=200)
    crash = closes.copy()
    crash.loc[crash.index[120]:, "RISKY"] *= 0.97 ** np.arange(1, len(crash) - 119)
    result = risk.run_guarded(crash, {"RISKY": 0.4, "CALM": 0.2}, risk.Limits(0.10, 0.02), manager=risk.cash_exit_manager)
    assert (result.log["action"] == "all cash").sum() == 1
    assert result.weights.iloc[-1].sum() == pytest.approx(0.0)


def test_exit_and_return_sells_then_buys_back_after_the_wait():
    closes = random_closes(days=260)
    crash = closes.copy()
    crash.loc[crash.index[120]:, "RISKY"] *= 0.97 ** np.arange(1, len(crash) - 119)
    result = risk.run_guarded(crash, {"RISKY": 0.4, "CALM": 0.2}, risk.Limits(0.10, 0.02), manager=risk.exit_and_return_manager)
    out_day = result.log.loc[result.log["action"] == "all cash", "date"].iloc[0]
    back_day = result.log.loc[result.log["action"] == "back in", "date"].iloc[0]
    assert len(result.equity.loc[out_day:back_day]) - 1 >= risk.STAY_OUT_DAYS
    assert result.metrics["Number of trades"] >= 2


def test_cash_rate_grows_money_that_sits_in_cash():
    closes = random_closes(days=200)
    flat = closes.copy()
    flat["CALM"] = 100.0
    plain = risk.run_guarded(flat, {"CALM": 0.0, "RISKY": 0.0}, manager=None)
    paid = risk.run_guarded(flat, {"CALM": 0.0, "RISKY": 0.0}, manager=None, cash_rate=0.05)
    assert plain.equity.iloc[-1] == pytest.approx(10_000)
    days = len(paid.equity) - 1
    assert paid.equity.iloc[-1] == pytest.approx(10_000 * 1.05 ** (days / 252), rel=1e-6)


def test_a_failing_plan_is_never_traded():
    closes = random_closes()
    crash = closes.copy()
    crash.loc[crash.index[120]:, "RISKY"] *= 0.97 ** np.arange(1, len(crash) - 119)
    reckless = lambda context: {"RISKY": 1.0}
    result = risk.run_guarded(crash, {"RISKY": 0.4, "CALM": 0.2}, risk.Limits(0.10, 0.02), manager=reckless, calm=["CALM"])
    assert result.log["action"].str.startswith("rejected").any()
    assert result.weights["RISKY"].max() < 0.6


def test_too_little_history_raises():
    with pytest.raises(ValueError, match="warm up"):
        risk.run_guarded(random_closes(days=50), {"RISKY": 1.0})


def test_capm_split_uses_only_earlier_days():
    index = pd.bdate_range("2024-01-01", periods=40)
    market = pd.Series(np.tile([0.01, -0.01], 20), index=index)
    portfolio = 2 * market
    split = risk.capm_split(portfolio, market, index[-1])
    assert split["beta"] == pytest.approx(2.0)
    assert split["market_part"] == pytest.approx(split["total"])
    assert split["own_part"] == pytest.approx(0.0)
    changed = portfolio.copy()
    changed.iloc[-1] = 0.5
    assert risk.capm_split(changed, market, index[-1])["beta"] == pytest.approx(2.0)


def test_unusual_moves_flags_a_big_day():
    returns = random_closes(days=200).pct_change()
    day = returns.index[-1]
    returns.loc[day, "CALM"] = -0.05
    table = risk.unusual_moves(returns, day)
    assert table.index[0] == "CALM"
    assert table.loc["CALM", "z_score"] < -5


def test_what_if_changes_one_day_and_keeps_later_returns():
    closes = random_closes()
    day = closes.index[100]
    shocked = risk.what_if(closes, day, {"RISKY": -0.12})
    moved = shocked["RISKY"].pct_change()
    assert moved.loc[day] == pytest.approx(-0.12)
    assert moved.iloc[101:].tolist() == pytest.approx(closes["RISKY"].pct_change().iloc[101:].tolist())
    assert shocked["CALM"].equals(closes["CALM"])
    with pytest.raises(ValueError):
        risk.what_if(closes, "1999-01-01", {"RISKY": -0.1})


def test_decide_on_a_shock_day_triggers_and_proposes_a_passing_plan():
    closes = random_closes()
    day = closes.index[150]
    shocked = risk.what_if(closes, day, {"RISKY": -0.20})
    decision = risk.decide_on(shocked, {"RISKY": 0.4, "CALM": 0.2}, day, risk.Limits(0.10, 0.02), calm=["CALM"])
    assert decision["trigger"] is not None
    assert decision["plan"] is not None and decision["check"].passed


def test_crisis_table_and_compare_format_for_reading():
    index = pd.bdate_range("2020-01-01", "2022-12-30")
    equity = pd.Series(np.linspace(100, 120, len(index)), index=index)
    table = risk.crisis_table({"A": equity})
    assert table.loc[("2020 Covid crash", "A"), "Worst fall"] == "0.0%"
    closes = random_closes()
    result = risk.run_guarded(closes, {"RISKY": 0.5}, manager=None, name="Hold")
    compared = risk.compare([result])
    assert list(compared.columns) == ["Hold"]
    assert compared.loc["Number of trades", "Hold"] == "0"
