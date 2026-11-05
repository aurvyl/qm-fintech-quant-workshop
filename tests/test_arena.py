import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from arena.demo_model import ask as demo_ask  # noqa: E402
from arena.server import Settings, create_app  # noqa: E402
from arena.store import ArenaStore, FilePersistence, public_view  # noqa: E402
from backtester import arena  # noqa: E402


def canned(alpha, t=1.0, curve=None):
    metrics = {"alpha_annual": alpha, "alpha_t": t, "beta": 0.9, "months": 46, "reliable": t >= 2, "total_return": 1.0,
               "cagr": 0.2, "sharpe": 1.0, "max_drawdown": -0.15, "trades": 10, "days_over_limit": 3, "worst_day": -0.03,
               "costs": 0.004}
    agent = {"asked": 10, "plans_passed": 9, "rejected_attempts": 2, "made_up_numbers": 0, "calls": 12, "budget_used": False}
    return {"metrics": metrics, "agent": agent, "curve": curve or [["2023-01-06", 100.0], ["2023-01-13", 101.0]],
            "decisions": [{"date": "2023-02-01", "trigger": "rotate", "attempts": 1, "outcome": "plan passed",
                           "explanation": "Moved weight."}]}


BARS = [{"name": "Buy and hold", "kind": "baseline", "metrics": canned(0.05)["metrics"], "curve": [["2023-01-06", 100.0]]}]
CODES = ["team-code-001", "team-code-002", "team-code-003"]


def make_client(run=None, **settings):
    def fake_run(guidance, ask, progress=None):
        return canned(float(len(guidance)) / 1000)

    config = Settings(ask=None, team_codes=CODES, admin_code="organiser-code", **settings)
    app = create_app(config, run=run or fake_run, baseline_rows=BARS)
    return TestClient(app), app


def wait_idle(app, seconds=5):
    deadline = time.time() + seconds
    while app.state.store.queue_depth() and time.time() < deadline:
        time.sleep(0.02)


def post(client, code, team, guidance="Prefer gold when you rotate, please."):
    return client.post("/api/submit", json={"code": code, "team": team, "guidance": guidance})


def test_guidance_is_cleaned_and_bounded():
    assert arena.clean_instructions("  Prefer gold when you rotate.\x00 ") == "Prefer gold when you rotate."
    with pytest.raises(ValueError):
        arena.clean_instructions("short")
    with pytest.raises(ValueError):
        arena.clean_instructions("x" * (arena.MAX_INSTRUCTION_CHARS + 1))
    with pytest.raises(ValueError):
        arena.clean_instructions(None)


def test_tickers_in_guidance_are_hidden_from_the_model():
    masked = arena.mask_text("Prefer GLD over TLT and trim AAPL, but keep XLP.")
    assert "GLD" not in masked and "AAPL" not in masked
    assert "ASSET_H" in masked and "ASSET_A" in masked


def test_the_call_budget_stops_the_model_not_the_run():
    seen = []
    limited = arena.Limited(lambda prompt: seen.append(prompt) or "{}", "Prefer GLD.", max_calls=2)
    limited("one"); limited("two")
    with pytest.raises(arena.BudgetUsed):
        limited("three")
    assert limited.calls == 2 and limited.stopped
    assert all("ASSET_H" in prompt for prompt in seen)


def test_an_entry_is_scored_with_the_model_masked_and_the_gate_in_place():
    seen = []

    def spy(prompt):
        seen.append(prompt)
        return demo_ask(prompt)

    result = arena.run_entry("Be cautious and prefer gold when rotating.", spy)
    assert seen and all("day T" in prompt and "AAPL" not in prompt for prompt in seen)
    assert result["agent"]["calls"] == len(seen)
    assert set(result["metrics"]) >= {"alpha_annual", "alpha_t", "beta", "total_return", "max_drawdown", "days_over_limit"}
    assert result["curve"][0][1] == pytest.approx(100.0, abs=3)


def test_a_reckless_model_cannot_trade_outside_the_rules():
    def reckless(prompt):
        return '{"weights": {"ASSET_A": 0.9, "ASSET_B": 0.1}, "explanation": "All in."}'

    result = arena.run_entry("Go all in on the biggest names.", reckless)
    assert result["agent"]["plans_passed"] == 0
    assert result["agent"]["rejected_attempts"] >= result["agent"]["asked"]


def test_running_out_of_calls_leaves_the_portfolio_held():
    result = arena.run_entry("Be cautious and prefer gold when rotating.", demo_ask, max_calls=3)
    assert result["agent"]["budget_used"] and result["agent"]["calls"] == 3


def test_the_baselines_cover_the_market_and_the_rules():
    names = [row["name"] for row in arena.baselines()]
    assert names == ["Buy and hold", "Rotate (rules)", "The market (SPY)"]


def test_only_a_valid_team_code_can_submit():
    client, _ = make_client()
    assert post(client, "wrong-code-123", "Team One").status_code == 403
    assert client.get("/api/mine", headers={"x-arena-code": "wrong-code-123"}).status_code == 403


def test_leaderboard_ranks_teams_by_alpha_and_lists_the_bars():
    client, app = make_client()
    assert post(client, CODES[0], "Short", "Prefer gold when you rotate.").status_code == 202
    assert post(client, CODES[1], "Long", "Prefer gold when you rotate, and keep tech in small steps always.").status_code == 202
    wait_idle(app)
    board = client.get("/api/state").json()
    assert [t["team"] for t in board["teams"]] == ["Long", "Short"]
    assert [t["rank"] for t in board["teams"]] == [1, 2]
    assert board["baselines"][0]["name"] == "Buy and hold"
    assert "guidance" not in board["teams"][0]


def test_one_run_at_a_time_a_fixed_number_of_runs_and_unique_names():
    release = threading.Event()

    def slow(guidance, ask, progress=None):
        release.wait(2)
        return canned(0.1)

    client, app = make_client(run=slow, max_attempts=2)
    assert post(client, CODES[0], "Alpha Team").status_code == 202
    assert post(client, CODES[0], "Alpha Team").status_code == 409
    assert post(client, CODES[1], "alpha team").status_code == 409
    assert post(client, CODES[0], "Other Name").status_code == 409
    release.set()
    wait_idle(app)
    assert post(client, CODES[0], "Alpha Team").status_code == 202
    wait_idle(app)
    assert post(client, CODES[0], "Alpha Team").status_code == 429


def test_entries_run_in_parallel():
    def slow(guidance, ask, progress=None):
        time.sleep(0.4)
        return canned(0.1)

    client, app = make_client(run=slow, workers=3)
    started = time.time()
    for code, name in zip(CODES, ["One", "Two", "Three"]):
        assert post(client, code, name).status_code == 202
    wait_idle(app)
    assert time.time() - started < 1.0


def test_a_failed_run_is_not_counted_and_shows_a_safe_message():
    def broken(guidance, ask, progress=None):
        raise RuntimeError("secret internal detail")

    client, app = make_client(run=broken, max_attempts=1)
    assert post(client, CODES[0], "Fragile").status_code == 202
    wait_idle(app)
    mine = client.get("/api/mine", headers={"x-arena-code": CODES[0]}).json()
    assert mine["entries"][0]["status"] == "failed"
    assert "secret" not in mine["entries"][0]["error"]
    assert mine["attempts_left"] == 1


def test_guidance_stays_private_until_the_organiser_reveals_it():
    client, app = make_client()
    post(client, CODES[0], "Private", "My secret plan for the risk manager.")
    wait_idle(app)
    entry_id = client.get("/api/mine", headers={"x-arena-code": CODES[0]}).json()["entries"][0]["id"]
    assert "guidance" not in client.get(f"/api/entry/{entry_id}").json()
    assert "guidance" not in client.get(f"/api/entry/{entry_id}", headers={"x-arena-code": CODES[1]}).json()
    owner = client.get(f"/api/entry/{entry_id}", headers={"x-arena-code": CODES[0]}).json()
    assert owner["guidance"].startswith("My secret") and owner["decisions"]
    assert client.post("/api/admin", json={"code": "organiser-code", "action": "reveal"}).json()["reveal"] is True
    assert "My secret" in client.get(f"/api/entry/{entry_id}").json()["guidance"]


def test_the_organiser_can_close_the_arena_and_others_cannot():
    client, _ = make_client()
    assert client.post("/api/admin", json={"code": CODES[0], "action": "close"}).status_code == 403
    assert client.post("/api/admin", json={"code": "organiser-code", "action": "close"}).json()["open"] is False
    assert post(client, CODES[0], "Late").status_code == 403
    client.post("/api/admin", json={"code": "organiser-code", "action": "open"})
    assert post(client, CODES[0], "Late").status_code == 202


def test_bad_input_is_refused_with_a_clear_reason():
    client, _ = make_client()
    assert post(client, CODES[0], "x").status_code == 400
    assert post(client, CODES[0], "Okay Team", "short").status_code == 400
    assert post(client, CODES[0], "<script>", "Prefer gold when you rotate.").status_code == 400


def test_too_many_wrong_codes_are_throttled():
    client, _ = make_client()
    for _ in range(20):
        assert post(client, "wrong-code-123", "Team One").status_code == 403
    assert post(client, "wrong-code-123", "Team One").status_code == 429


def test_state_survives_a_restart_and_unfinished_entries_run_again(tmp_path):
    persistence = FilePersistence(tmp_path / "state.json")
    store = ArenaStore(persistence)
    entry = store.add_entry("key1", "Resilient", "Guidance that is long enough.")
    store.update(entry["id"], status="running")
    reloaded = ArenaStore(FilePersistence(tmp_path / "state.json"))
    assert reloaded.get(entry["id"])["status"] == "queued"
    assert reloaded.team_name("key1") == "Resilient"
    assert reloaded.queued_ids() == [entry["id"]]


def test_progress_is_reported_and_kept_private():
    seen = []
    prices = arena.arena_prices()

    def ask(prompt):
        return '{"weights": {"ASSET_A": 0.2, "ASSET_B": 0.2, "ASSET_C": 0.2, "ASSET_D": 0.2, "ASSET_E": 0.2}, "explanation": "Hold the mix."}'

    arena.run_entry("Prefer calm assets when stressed.", ask, prices=prices, max_calls=6, progress=seen.append)
    assert seen and seen[-1]["calls"] <= 6 and 0 <= seen[-1]["percent"] <= 100
    assert seen[-1]["decisions"] == len(seen)
    assert seen[-1]["feed"][0]["date"] == seen[-1]["day"]
    assert all(d["direction"] in ("none", "cut", "add", "swap") for d in seen[-1]["feed"])
    curve = seen[-1]["curve"]
    assert curve and curve[0][1] > 0 and curve[0][0] >= arena.SCORE_START
    passed = [d for d in seen[-1]["feed"] if d["outcome"] == "plan passed"]
    assert all(abs(sum(d["weights"].values()) - 1) < 0.06 for d in passed)


def test_public_view_hides_latest_reasoning_from_others():
    entry = {"id": 1, "team": "t", "status": "running", "progress": {"percent": 10, "feed": ["secret idea"], "holding": {"GLD": 0.2}, "curve": [["2023-01-06", 100.0]]}}
    assert public_view(entry)["progress"] == {"percent": 10}
    assert public_view(entry, include_guidance=True)["progress"]["feed"] == ["secret idea"]
