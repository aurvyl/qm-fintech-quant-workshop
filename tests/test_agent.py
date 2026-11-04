import io
import json

import numpy as np
import pandas as pd
import pytest

from backtester import agent, risk


def closes():
    rng = np.random.default_rng(1)
    index = pd.bdate_range("2020-01-01", periods=200, name="Date")
    data = {t: 100 * np.cumprod(1 + rng.normal(0.0003, v, 200)) for t, v in (("RISKY", 0.03), ("CALM", 0.003))}
    frame = pd.DataFrame(data, index=index)
    frame.loc[frame.index[120]:, "RISKY"] *= 0.97 ** np.arange(1, 81)
    return frame


def context():
    prices = closes()
    returns = prices.pct_change().iloc[-60:]
    lines = risk.lines_from_limits(risk.Limits(0.10, 0.02))
    held = pd.Series({"RISKY": 0.4, "CALM": 0.2})
    return risk.build_context(prices.index[-1], held, returns, pd.Series([1.0, 0.97]), lines, "rotate",
                              ["One-day VaR 5.00% is above the 2.00% budget."], ["CALM"], held)


def scripted(*replies):
    prompts = []
    queue = list(replies)

    def ask(prompt):
        prompts.append(prompt)
        return queue.pop(0)

    ask.prompts = prompts
    return ask


def test_snapshot_hides_raw_data_and_rounds():
    shown = agent.snapshot(context())
    assert "_returns" not in shown and "_lines" not in shown
    assert json.dumps(shown)
    assert all(len(str(v).split(".")[-1]) <= 4 for v in shown["annual_volatility"].values())


def test_parse_reply_reads_json_inside_fences():
    reply = agent.parse_reply('Sure!\n```json\n{"weights": {"CALM": 0.4}, "explanation": "ok"}\n```')
    assert reply == {"weights": {"CALM": 0.4}, "explanation": "ok"}
    with pytest.raises(ValueError):
        agent.parse_reply("no json here")
    with pytest.raises(ValueError):
        agent.parse_reply('{"plan": 1}')


def test_a_passing_plan_is_returned_with_its_explanation():
    ask = scripted(json.dumps({"weights": {"CALM": 0.4}, "explanation": "VaR was 5.00%, so I moved into CALM."}))
    manager = agent.AgentManager(ask=ask)
    assert manager(context()) == {"CALM": 0.4}
    entry = manager.log[0]
    assert entry["explanation"].startswith("VaR")
    assert entry["unsupported_numbers"] == []
    assert "Do not use anything you know about what happened after" in ask.prompts[0]


def test_failed_check_is_sent_back_and_the_model_can_fix_it():
    ask = scripted(
        json.dumps({"weights": {"RISKY": 0.9}, "explanation": "all in"}),
        json.dumps({"weights": {"CALM": 0.4}, "explanation": "fixed"}),
    )
    manager = agent.AgentManager(ask=ask)
    assert manager(context()) == {"CALM": 0.4}
    assert "40% cap" in ask.prompts[1]
    assert [a["passed"] for a in manager.log[0]["attempts"]] == [False, True]


def test_after_three_failures_the_manager_holds():
    bad = json.dumps({"weights": {"RISKY": 0.9}, "explanation": "no"})
    manager = agent.AgentManager(ask=scripted(bad, bad, bad, bad))
    assert manager(context()) is None
    assert len(manager.log[0]["attempts"]) == 3
    assert manager.decisions()["outcome"].tolist() == ["hold"]


def test_unreadable_replies_count_as_attempts():
    manager = agent.AgentManager(ask=scripted("hmm", json.dumps({"weights": {"CALM": 0.4}, "explanation": ""})))
    assert manager(context()) == {"CALM": 0.4}
    assert "error" in manager.log[0]["attempts"][0]


def test_unsupported_numbers_catches_made_up_figures():
    numbers = {"one_day_var": 0.0312, "weights": {"A": 0.25}}
    assert agent.unsupported_numbers("VaR is 3.12% and A is 25%.", numbers) == []
    assert agent.unsupported_numbers("VaR is 3.1%, about 3%.", numbers) == []
    assert agent.unsupported_numbers("A fell 14.7% on weak earnings.", numbers) == ["14.7%"]
    assert agent.unsupported_numbers("Sold 2 holdings.", numbers) == []


def test_anonymise_hides_dates_and_tickers_and_maps_back():
    ctx = context()
    masked, back = agent.anonymise(ctx)
    text = json.dumps(agent.snapshot(masked))
    assert "RISKY" not in text and "CALM" not in text and ctx["date"] not in text
    assert set(back.values()) == {"RISKY", "CALM"}
    assert list(masked["_returns"].columns) == list(back)


def test_anonymous_agent_trades_real_tickers():
    ask = scripted(json.dumps({"weights": {"ASSET_B": 0.4}, "explanation": "moved to the calm asset"}))
    manager = agent.AgentManager(ask=ask, anonymous=True)
    assert manager(context()) == {"CALM": 0.4}
    assert "RISKY" not in ask.prompts[0]


def test_offline_agent_gives_the_rule_based_result():
    prices = closes()
    limits = risk.Limits(0.10, 0.02)
    rules = risk.run_guarded(prices, {"RISKY": 0.4, "CALM": 0.2}, limits, calm=["CALM"])
    offline = agent.OfflineAgent()
    via_agent = risk.run_guarded(prices, {"RISKY": 0.4, "CALM": 0.2}, limits, manager=offline, calm=["CALM"])
    assert via_agent.equity.tolist() == pytest.approx(rules.equity.tolist())
    assert len(offline.log) > 0
    assert all(e["unsupported_numbers"] == [] for e in offline.log)


def test_cached_asks_each_prompt_once(tmp_path):
    calls = []
    ask = agent.cached(lambda p: calls.append(p) or f"reply to {p}", folder=tmp_path)
    assert ask("a") == "reply to a"
    assert ask("a") == "reply to a"
    assert ask("b") == "reply to b"
    assert calls == ["a", "b"]
    assert len(list(tmp_path.iterdir())) == 2


def test_gemini_sends_the_key_in_a_header_not_the_url(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"], seen["headers"], seen["body"] = request.full_url, dict(request.header_items()), json.loads(request.data)
        reply = {"candidates": [{"content": {"parts": [{"text": '{"weights": {}}'}]}}]}
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr(agent.urllib.request, "urlopen", fake_urlopen)
    ask = agent.gemini("secret-key", model="test-model", seconds_between_calls=0)
    assert ask("hello") == '{"weights": {}}'
    assert "secret-key" not in seen["url"] and "test-model" in seen["url"]
    assert seen["headers"]["X-goog-api-key"] == "secret-key"
    assert seen["body"]["contents"][0]["parts"][0]["text"] == "hello"
    with pytest.raises(ValueError):
        agent.gemini("")


def test_gemini_retries_a_rate_limit_then_gives_a_clear_error(monkeypatch):
    calls = []

    def limited(request, timeout):
        calls.append(1)
        raise agent.urllib.error.HTTPError(request.full_url, 429, "Too many", {}, None)

    monkeypatch.setattr(agent.urllib.request, "urlopen", limited)
    monkeypatch.setattr(agent.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="HTTP 429"):
        agent.gemini("key", seconds_between_calls=0)("hello")
    assert len(calls) == 3


def test_workshop_model_sends_the_code_and_shows_the_server_message(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"], seen["body"] = request.full_url, json.loads(request.data)
        return io.BytesIO(json.dumps({"text": "answer"}).encode())

    monkeypatch.setattr(agent.urllib.request, "urlopen", fake_urlopen)
    ask = agent.workshop_model("https://example.test/ask", "alpha-code-1", seconds_between_calls=0)
    assert ask("hello") == "answer"
    assert seen["body"] == {"code": "alpha-code-1", "prompt": "hello"}

    def denied(request, timeout):
        raise agent.urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, io.BytesIO(json.dumps({"error": "That access code is not valid."}).encode()))

    monkeypatch.setattr(agent.urllib.request, "urlopen", denied)
    with pytest.raises(RuntimeError, match="access code is not valid"):
        ask("hello")
    with pytest.raises(ValueError):
        agent.workshop_model("", "code")


def test_exact_snapshot_figures_and_trade_results_are_not_flagged():
    numbers = {"risk_share": {"ASSET_C": 0.2794}, "one_day_var": 0.0283}
    assert agent.unsupported_numbers("ASSET_C carries 0.2794 of the risk and 27.94% of it.", numbers) == []
    assert agent.unsupported_numbers("VaR is 0.0283.", numbers) == []
    weights = {"ASSET_A": 0.3, "ASSET_B": 0.6}
    assert agent.unsupported_numbers("That leaves 10% in cash with 90% invested.", numbers, weights) == []
    assert agent.unsupported_numbers("ASSET_C fell 31.5% last week.", numbers) == ["31.5%"]
