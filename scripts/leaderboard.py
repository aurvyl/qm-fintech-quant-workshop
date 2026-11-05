"""Score every entry in competition/entries/ and write competition/leaderboard.md.

Run by the organiser, never by teams, so every entry is scored the same way on the same data.

Usage:
    python scripts/leaderboard.py               # entries plus the house strategies
    python scripts/leaderboard.py --no-house    # entries only

Each entry is one file, competition/entries/<team>.py, with a TEAM name and a strategy(closes)
function. Read every entry before running it: this runs the teams' code on your machine.
"""

import argparse
import datetime as dt
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backtester.alpha import (  # noqa: E402
    CASH_RATE, COST_BPS, MARKET, Score, TEST_PERIOD, competition_closes, data_fingerprint, format_leaderboard,
    leaderboard, score,
)
from competition.examples import HOUSE  # noqa: E402

ENTRIES = REPO_ROOT / "competition" / "entries"
OUTPUT = REPO_ROOT / "competition" / "leaderboard.md"


FORBIDDEN = {
    "shift(-": "shifts data backwards, which reads future prices",
    "read_csv": "reads data files itself instead of using `closes`",
    "load_closes": "loads prices itself instead of using `closes`",
    "load_prices": "loads prices itself instead of using `closes`",
    "open(": "opens files",
    "yfinance": "downloads data",
    "urllib": "uses the network",
    "requests": "uses the network",
}


def source_problem(text: str) -> str | None:
    """A reason to refuse an entry before running it. The look-ahead check cannot see data a strategy loads itself."""
    for word, reason in FORBIDDEN.items():
        if word in text:
            return f"{reason} (`{word}`)"
    return None


def load_entries(folder: Path = ENTRIES) -> list[tuple[str, object, str | None]]:
    """(team name, strategy function or None, problem) for every .py file in the folder."""
    entries = []
    for path in sorted(folder.glob("*.py")):
        if path.name.startswith("_"):
            continue
        problem = source_problem(path.read_text())
        if problem:
            entries.append((path.stem, None, problem))
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"entry_{path.stem}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            entries.append((str(getattr(module, "TEAM", path.stem)), module.strategy, None))
        except Exception as error:
            entries.append((path.stem, None, f"could not load: {type(error).__name__}: {error}"))
    return entries


def score_all(include_house: bool = True) -> list[Score]:
    closes = competition_closes()
    scores = []
    for team, strategy, problem in load_entries():
        if strategy is None:
            scores.append(Score(team, None, None, [], problem))
        else:
            scores.append(score(strategy, team, closes))
    if include_house:
        scores += [score(fn, name, closes) for name, fn in HOUSE.items()]
    return scores


def to_markdown(scores: list[Score]) -> str:
    closes = competition_closes()
    table = format_leaderboard(leaderboard(scores))
    header = "| Rank | " + " | ".join(table.columns) + " |"
    rule = "|" + "---|" * (len(table.columns) + 1)
    rows = [f"| {rank} | " + " | ".join(str(v) for v in row) + " |" for rank, row in table.iterrows()]
    return "\n".join([
        "# Alpha competition leaderboard",
        "",
        f"Scored {dt.date.today():%Y-%m-%d} on data file `{data_fingerprint()}`, "
        f"test period {TEST_PERIOD[0]} to {closes.index[-1]:%Y-%m-%d}.",
        f"Market: {MARKET}. Cash: {CASH_RATE:.0%} a year. Trading cost: {COST_BPS:.0f} bp. Long only, no leverage.",
        "Ranked by test-period alpha t-stat, so a lucky big alpha cannot win. Reliable means a t-stat of 2 or more: under 2 the alpha could easily be luck.",
        "",
        header,
        rule,
        *rows,
        "",
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-house", action="store_true", help="leave out the house strategies")
    args = parser.parse_args(argv)
    text = to_markdown(score_all(include_house=not args.no_house))
    OUTPUT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
