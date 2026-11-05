"""The arena web app: a page for guidance and a leaderboard, with parallel workers behind it.

Run it on a laptop to try it:   python -m arena.server --demo
Deploy it on Modal:             see arena/README.md

Settings come from the environment (the Modal secret supplies them):
  ARENA_TEAM_CODES    comma-separated codes, one per team (8 characters or more)
  ARENA_ADMIN_CODE    the organiser's code for open, close and reveal
  ARENA_PROXY_URL     the workshop model server's URL
  ARENA_PROXY_CODE    the access code the arena uses on that server
"""

import hmac
import logging
import os
import re
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backtester import arena
from backtester.arena import MAX_INSTRUCTION_CHARS, SCORE_START, clean_instructions, run_entry

from .store import ArenaStore, FilePersistence, code_key, public_view

LOG = logging.getLogger("arena")
STATIC = Path(__file__).parent / "static"
TEAM_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{1,29}$")
FAILED = "The run failed. Nothing was counted. Try again in a minute."
MAX_ATTEMPTS = 100
MAX_QUEUE = 60
WORKERS = 6
BAD_CODES_PER_MINUTE = 20


@dataclass
class Settings:
    ask: object
    team_codes: list = field(default_factory=list)
    admin_code: str = ""
    persistence: object = None
    workers: int = WORKERS
    max_attempts: int = MAX_ATTEMPTS
    max_queue: int = MAX_QUEUE


def settings_from_env(ask, persistence=None) -> Settings:
    def codes(name):
        return [c.strip() for c in os.environ.get(name, "").split(",") if len(c.strip()) >= 8]

    return Settings(ask=ask, team_codes=codes("ARENA_TEAM_CODES"), admin_code=os.environ.get("ARENA_ADMIN_CODE", ""),
                    persistence=persistence)


class Submission(BaseModel):
    code: str = Field(max_length=100)
    team: str = Field(max_length=60)
    guidance: str = Field(max_length=MAX_INSTRUCTION_CHARS * 2)


class Admin(BaseModel):
    code: str = Field(max_length=100)
    action: str = Field(max_length=20)


def _matches(code: str, known: list) -> bool:
    given = code.encode()
    return any(hmac.compare_digest(given, k.encode()) for k in known)


def _ranked(store: ArenaStore, baseline_rows: list) -> dict:
    teams = []
    for entry in store.latest_per_team():
        view = public_view(entry)
        view["attempts"] = store.attempts(entry["key"])
        teams.append(view)
    teams.sort(key=lambda row: row["metrics"]["alpha_annual"], reverse=True)
    for place, row in enumerate(teams, 1):
        row["rank"] = place
    waiting = store.teams_without_result()
    bars = [{"name": b["name"], "metrics": b["metrics"]} for b in baseline_rows]
    return {"teams": teams, "waiting": waiting, "baselines": bars}


def create_app(settings: Settings, run=run_entry, baseline_rows=None) -> FastAPI:
    store = ArenaStore(settings.persistence)
    bars = baseline_rows if baseline_rows is not None else arena.baselines()
    pool = ThreadPoolExecutor(max_workers=settings.workers)
    failures = defaultdict(deque)
    failures_lock = threading.Lock()
    app = FastAPI(title="Agent arena", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store
    app.state.pool = pool

    def work(entry_id: int) -> None:
        entry = store.get(entry_id)
        store.update(entry_id, status="running", started=time.time())
        try:
            result = run(entry["guidance"], settings.ask, progress=lambda p: store.set_progress(entry_id, p))
            store.update(entry_id, status="done", finished=time.time(), result=result)
        except Exception:
            LOG.exception("entry %s failed", entry_id)
            store.update(entry_id, status="failed", finished=time.time(), error=FAILED)

    for entry_id in store.queued_ids():
        pool.submit(work, entry_id)

    def throttle(request: Request) -> None:
        host = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with failures_lock:
            recent = failures[host]
            while recent and now - recent[0] > 60:
                recent.popleft()
            if len(recent) >= BAD_CODES_PER_MINUTE:
                raise HTTPException(429, "Too many wrong codes. Wait a minute.")

    def failed_code(request: Request) -> None:
        host = request.client.host if request.client else "unknown"
        with failures_lock:
            failures[host].append(time.monotonic())

    def team_key(request: Request, code: str) -> str:
        throttle(request)
        if not code or not _matches(code, settings.team_codes):
            failed_code(request)
            raise HTTPException(403, "That team code is not valid.")
        return code_key(code)

    @app.get("/")
    def page():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    def state():
        body = _ranked(store, bars)
        body.update({
            "open": store.flag("open"), "reveal": store.flag("reveal"), "max_attempts": settings.max_attempts,
            "scored_from": SCORE_START, "max_chars": MAX_INSTRUCTION_CHARS, "busy": store.queue_depth(),
        })
        return body

    @app.get("/api/curves")
    def curves():
        out = {b["name"]: b["curve"] for b in bars}
        for entry in store.latest_per_team():
            out[entry["team"]] = entry["result"]["curve"]
        return out

    @app.get("/api/entry/{entry_id}")
    def entry_detail(entry_id: int, x_arena_code: str = Header(default="")):
        entry = store.get(entry_id)
        if not entry:
            raise HTTPException(404, "No such entry.")
        mine = bool(x_arena_code) and _matches(x_arena_code, settings.team_codes) and code_key(x_arena_code) == entry["key"]
        shown = mine or store.flag("reveal")
        return public_view(entry, include_guidance=shown, include_detail=shown)

    @app.get("/api/mine")
    def mine(request: Request, x_arena_code: str = Header(default="")):
        key = team_key(request, x_arena_code)
        entries = []
        for e in store.entries_for(key):
            view = public_view(e, include_guidance=True)
            view["queue_position"] = store.queue_position(e["id"])
            entries.append(view)
        return {"team": store.team_name(key), "entries": entries, "attempts_left": settings.max_attempts - store.counted_for(key)}

    @app.post("/api/submit", status_code=202)
    def submit(body: Submission, request: Request):
        key = team_key(request, body.code)
        if not store.flag("open"):
            raise HTTPException(403, "The arena is closed.")
        team = body.team.strip()
        if not TEAM_NAME.match(team):
            raise HTTPException(400, "Team names use 2 to 30 letters, numbers, spaces, dots, dashes or underscores.")
        existing = store.team_name(key)
        if existing and existing != team:
            raise HTTPException(409, f"This code belongs to {existing}.")
        if store.name_taken(team, key):
            raise HTTPException(409, "Another team already uses that name.")
        try:
            guidance = clean_instructions(body.guidance)
        except ValueError as error:
            raise HTTPException(400, str(error))
        if store.busy(key):
            raise HTTPException(409, "Your last run is still going. Wait for it to finish.")
        if store.counted_for(key) >= settings.max_attempts:
            raise HTTPException(429, f"You have used all {settings.max_attempts} runs.")
        if store.queue_depth() >= settings.max_queue:
            raise HTTPException(503, "The arena is full. Try again in a minute.")
        entry = store.add_entry(key, team, guidance)
        pool.submit(work, entry["id"])
        return {"id": entry["id"]}

    @app.post("/api/admin")
    def admin(body: Admin, request: Request):
        throttle(request)
        if not settings.admin_code or not _matches(body.code, [settings.admin_code]):
            failed_code(request)
            raise HTTPException(403, "That is not the organiser code.")
        actions = {"open": ("open", True), "close": ("open", False), "reveal": ("reveal", True), "hide": ("reveal", False)}
        if body.action not in actions:
            raise HTTPException(400, "Use open, close, reveal or hide.")
        name, value = actions[body.action]
        store.set_flag(name, value)
        return {"open": store.flag("open"), "reveal": store.flag("reveal")}

    return app


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the agent arena on this computer.")
    parser.add_argument("--demo", action="store_true", help="use the free stand-in model (not AI) and demo codes")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--state", default="arena_state.json", help="where to keep teams and entries")
    args = parser.parse_args()
    persistence = FilePersistence(args.state)
    if args.demo:
        from .demo_model import ask

        settings = Settings(ask=ask, team_codes=[f"demo-team-{i:02d}" for i in range(1, 9)], admin_code="demo-organiser",
                            persistence=persistence)
        print("Demo codes: demo-team-01 to demo-team-08. Organiser code: demo-organiser.")
    else:
        from backtester.agent import workshop_model

        ask = workshop_model(os.environ["ARENA_PROXY_URL"], os.environ["ARENA_PROXY_CODE"], seconds_between_calls=0.0)
        settings = settings_from_env(ask, persistence)
    uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
