"""Where the arena keeps teams, entries and the open or closed flags.

State lives in memory behind a lock and is saved after every change through a persistence object with
load() and save(state). FilePersistence is for a laptop; the Modal wrapper supplies one backed by a Modal Dict.
"""

import hashlib
import json
import threading
import time
from pathlib import Path

PRIVATE_PROGRESS = ("feed", "holding")
PUBLIC_FIELDS = ("id", "team", "status", "created", "started", "finished", "error")


class FilePersistence:
    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None

    def save(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state))
        temporary.replace(self.path)


def code_key(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:16]


class ArenaStore:
    def __init__(self, persistence=None, open_at_start: bool = True):
        self.lock = threading.RLock()
        self.persistence = persistence
        self.state = {"entries": [], "teams": {}, "flags": {"open": open_at_start, "reveal": False}, "next_id": 1}
        loaded = persistence.load() if persistence else None
        if loaded:
            self.state.update(loaded)
        for entry in self.state["entries"]:
            if entry["status"] == "running":
                entry["status"] = "queued"
                entry.pop("progress", None)

    def _save(self) -> None:
        if self.persistence:
            self.persistence.save(self.state)

    def flag(self, name: str) -> bool:
        with self.lock:
            return bool(self.state["flags"].get(name))

    def set_flag(self, name: str, value: bool) -> None:
        with self.lock:
            self.state["flags"][name] = bool(value)
            self._save()

    def team_name(self, key: str):
        with self.lock:
            return self.state["teams"].get(key)

    def name_taken(self, name: str, key: str) -> bool:
        with self.lock:
            return any(n.lower() == name.lower() and k != key for k, n in self.state["teams"].items())

    def add_entry(self, key: str, team: str, guidance: str) -> dict:
        with self.lock:
            self.state["teams"][key] = team
            entry = {"id": self.state["next_id"], "key": key, "team": team, "guidance": guidance, "status": "queued",
                     "created": time.time(), "started": None, "finished": None, "result": None, "error": None}
            self.state["next_id"] += 1
            self.state["entries"].append(entry)
            self._save()
            return dict(entry)

    def update(self, entry_id: int, **changes) -> None:
        with self.lock:
            for entry in self.state["entries"]:
                if entry["id"] == entry_id:
                    entry.update(changes)
                    break
            self._save()

    def set_progress(self, entry_id: int, progress: dict) -> None:
        with self.lock:
            for entry in self.state["entries"]:
                if entry["id"] == entry_id:
                    entry["progress"] = progress
                    break

    def queue_position(self, entry_id: int):
        with self.lock:
            waiting = [e["id"] for e in self.state["entries"] if e["status"] == "queued"]
            return waiting.index(entry_id) + 1 if entry_id in waiting else None

    def get(self, entry_id: int):
        with self.lock:
            for entry in self.state["entries"]:
                if entry["id"] == entry_id:
                    return dict(entry)
            return None

    def entries_for(self, key: str) -> list:
        with self.lock:
            return [dict(e) for e in self.state["entries"] if e["key"] == key]

    def counted_for(self, key: str) -> int:
        with self.lock:
            return sum(1 for e in self.state["entries"] if e["key"] == key and e["status"] != "failed")

    def busy(self, key: str) -> bool:
        with self.lock:
            return any(e["key"] == key and e["status"] in ("queued", "running") for e in self.state["entries"])

    def queue_depth(self) -> int:
        with self.lock:
            return sum(1 for e in self.state["entries"] if e["status"] in ("queued", "running"))

    def queued_ids(self) -> list:
        with self.lock:
            return [e["id"] for e in self.state["entries"] if e["status"] == "queued"]

    def latest_per_team(self) -> list:
        with self.lock:
            latest = {}
            for entry in self.state["entries"]:
                if entry["status"] == "done":
                    latest[entry["key"]] = entry
            return [dict(e) for e in latest.values()]

    def teams_without_result(self) -> list:
        with self.lock:
            done = {e["key"] for e in self.state["entries"] if e["status"] == "done"}
            rows = []
            for key, name in self.state["teams"].items():
                if key in done:
                    continue
                mine = [e for e in self.state["entries"] if e["key"] == key]
                row = {"team": name, "status": mine[-1]["status"] if mine else "none"}
                progress = mine[-1].get("progress") if mine else None
                if progress and row["status"] == "running":
                    row["percent"] = progress.get("percent")
                rows.append(row)
            return rows

    def attempts(self, key: str) -> int:
        return self.counted_for(key)


def public_view(entry: dict, include_guidance: bool = False, include_detail: bool = False) -> dict:
    view = {name: entry.get(name) for name in PUBLIC_FIELDS}
    result = entry.get("result")
    if result:
        view["metrics"] = result["metrics"]
        view["agent"] = result["agent"]
        if include_detail:
            view["decisions"] = result["decisions"]
            view["curve"] = result["curve"]
    progress = entry.get("progress")
    if progress and entry.get("status") == "running":
        view["progress"] = progress if include_guidance else {k: v for k, v in progress.items() if k not in PRIVATE_PROGRESS}
    if include_guidance:
        view["guidance"] = entry.get("guidance")
    return view
