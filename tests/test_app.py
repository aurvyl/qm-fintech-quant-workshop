import datetime
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def run_app():
    return AppTest.from_file(APP, default_timeout=60).run()


def test_app_runs_with_defaults():
    at = run_app()
    assert not at.exception
    assert len(at.metric) == 4


def test_fast_not_below_slow_shows_warning():
    at = run_app()
    at.sidebar.slider[0].set_value(100)
    at.sidebar.slider[1].set_value(50)
    at.run()
    assert not at.exception
    assert len(at.warning) == 1
    assert len(at.metric) == 0


def test_rsi_strategy_runs():
    at = run_app()
    at.sidebar.radio[0].set_value("RSI mean reversion").run()
    assert not at.exception
    assert len(at.metric) == 4


def test_one_day_range_shows_warning():
    at = run_app()
    last_day = at.sidebar.date_input[0].value[1]
    at.sidebar.date_input[0].set_value((last_day, last_day)).run()
    assert not at.exception
    assert len(at.warning) == 1


def test_short_window_warns_that_annualised_numbers_are_unreliable():
    at = run_app()
    last_day = at.sidebar.date_input[0].value[1]
    at.sidebar.date_input[0].set_value((last_day - datetime.timedelta(days=14), last_day)).run()
    assert not at.exception
    assert len(at.metric) == 4
    assert any("annualised" in caption.value for caption in at.caption)
    full = run_app()
    assert not any("annualised" in caption.value for caption in full.caption)


def test_metrics_explain_themselves():
    at = run_app()
    assert all(metric.help for metric in at.metric)


def test_overfitting_lab_compares_training_winner_with_test():
    at = run_app()
    assert [tab.label for tab in at.tabs] == ["📊 Backtest", "🎯 Overfitting lab", "🛡️ Risk rules"]
    (summary,) = at.tabs[1].info
    assert "best settings on 2016–2021" in summary.value
    assert " of 38**" in summary.value  # 38 valid fast < slow pairs in the grid


def test_overfitting_lab_still_shows_when_backtest_settings_are_invalid():
    at = run_app()
    at.sidebar.slider[0].set_value(100)
    at.sidebar.slider[1].set_value(50)
    at.run()
    assert not at.exception
    assert len(at.tabs[1].info) == 1


def test_risk_rules_tab_waits_for_the_toggle_then_compares_three_portfolios():
    at = run_app()
    risk_tab = at.tabs[2]
    assert any("Run the replay" in info.value for info in risk_tab.info)
    at.toggle(key="risk_run").set_value(True)
    at.run(timeout=180)
    assert not at.exception
    tables = [frame.value for frame in at.tabs[2].dataframe]
    assert list(tables[0].columns) == ["Buy & hold", "Sell everything", "Rotate"]
    assert len(at.tabs[2].warning) + len(at.tabs[2].success) == 1
