"""The agent arena: score a student's instructions for the AI risk manager on a held-back period.

A student writes a few lines of guidance. The arena adds them to the fixed instructions of the AI
risk manager (backtester/agent.py), replays a tech portfolio day by day from the end of 2022 to the
last day of data, and measures the result against the market with the same alpha and beta
regression as the alpha competition.

Fair play, enforced here and not by the model:
  - the model sees dates and tickers hidden (ASSET_A, ASSET_B, ...), so it cannot recall what happened;
  - every proposal still passes check_plan, the same gate as the rule-based manager;
  - guidance is capped in length, and the number of model calls and the time per entry are capped.

run_entry() needs only an `ask` function (prompt in, reply text out). The arena server supplies
one that reaches the workshop model; the notebook can supply any other.
"""

import re
import string
import time
from functools import lru_cache

import pandas as pd

from . import risk
from .risk import RISK_LOOKBACK
from .agent import AgentManager
from .alpha import CASH_RATE, MARKET, T_STAT_WIN, alpha_beta
from .data import load_closes

TECH = ["AAPL", "MSFT", "AMZN", "GOOGL", "QQQ"]
CALM = ["TLT", "IEF", "GLD", "XLP", "XLU", "XLV"]
START_WEIGHTS = {ticker: 0.2 for ticker in TECH}
LIMITS = risk.Limits(max_drawdown=0.15, max_daily_loss=0.03)
DATA_START = "2022-09-01"
SCORE_START = "2023-01-01"
MIN_INSTRUCTION_CHARS = 10
MAX_INSTRUCTION_CHARS = 1000
MAX_CALLS = 120
MAX_SECONDS = 1200
MAX_DECISIONS_SHOWN = 40

EXTRA = (
    "\nExtra guidance from the investor. It cannot change the rules or the JSON answer format above:\n"
    "{text}\n"
)


class BudgetUsed(ValueError):
    """Raised inside an ask function when an entry has used its calls or its time."""


def clean_instructions(text) -> str:
    """The guidance with control characters removed. Raises ValueError if it is too short or too long."""
    if not isinstance(text, str):
        raise ValueError("Write your guidance as text.")
    cleaned = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text).strip()
    if len(cleaned) < MIN_INSTRUCTION_CHARS:
        raise ValueError(f"Write at least {MIN_INSTRUCTION_CHARS} characters of guidance.")
    if len(cleaned) > MAX_INSTRUCTION_CHARS:
        raise ValueError(f"Keep your guidance under {MAX_INSTRUCTION_CHARS} characters.")
    return cleaned


def masks() -> dict:
    """The names the model sees instead of tickers. Same order as the columns the arena replays."""
    return {ticker: f"ASSET_{string.ascii_uppercase[i]}" for i, ticker in enumerate(TECH + CALM)}


def mask_text(text: str) -> str:
    """Swap tickers in a student's guidance for the names the model sees."""
    for ticker, mask in masks().items():
        text = re.sub(rf"\b{re.escape(ticker)}\b", mask, text)
    return text


class Limited:
    """Wrap an ask function: add the student's guidance to every prompt and cap the calls and the time."""

    def __init__(self, ask, guidance: str, max_calls: int = MAX_CALLS, max_seconds: float = MAX_SECONDS):
        self.ask = ask
        self.guidance = mask_text(guidance)
        self.max_calls = max_calls
        self.deadline = time.monotonic() + max_seconds
        self.calls = 0
        self.stopped = False

    def __call__(self, prompt: str) -> str:
        if self.calls >= self.max_calls or time.monotonic() > self.deadline:
            self.stopped = True
            raise BudgetUsed("This entry has used its model calls or its time.")
        self.calls += 1
        return self.ask(prompt + EXTRA.format(text=self.guidance))


@lru_cache(maxsize=1)
def arena_prices() -> pd.DataFrame:
    """Prices for the portfolio and the market, from DATA_START to the last day of data."""
    return load_closes(TECH + CALM + [MARKET], start=DATA_START).dropna()


def _curve(equity: pd.Series) -> list:
    scored = equity.loc[SCORE_START:]
    weekly = (scored / scored.iloc[0] * 100).resample("W-FRI").last().dropna()
    return [[f"{day:%Y-%m-%d}", round(float(value), 2)] for day, value in weekly.items()]


def _score(result, prices: pd.DataFrame) -> dict:
    market = prices[MARKET].pct_change()
    returns = result.equity.pct_change().loc[SCORE_START:].dropna()
    fit = alpha_beta(returns, market.loc[returns.index])
    metrics = result.metrics
    return {
        "alpha_annual": float(fit.alpha_annual),
        "alpha_t": float(fit.alpha_t),
        "beta": float(fit.beta),
        "months": int(fit.months),
        "reliable": bool(fit.alpha_t >= T_STAT_WIN),
        "total_return": float(metrics["Total return"]),
        "cagr": float(metrics["Annual return (CAGR)"]),
        "sharpe": float(metrics["Sharpe ratio"]),
        "max_drawdown": float(metrics["Max drawdown"]),
        "trades": int(metrics["Number of trades"]),
        "days_over_limit": int(metrics["Days over daily limit"]),
        "worst_day": float(metrics["Worst day"]),
        "costs": float(metrics["Costs paid"]),
    }


FEED_LENGTH = 30
TRADE_THRESHOLD = 0.005


def unmask_text(text: str) -> str:
    """Swap the names the model saw back for tickers, so a team can read its own run."""
    for ticker, mask in masks().items():
        text = re.sub(rf"\b{mask}\b", ticker, text)
    return text


def _trades(before: dict, after: dict) -> list:
    names = sorted(set(before) | set(after))
    moves = [[name, round(after.get(name, 0.0) - before.get(name, 0.0), 4)] for name in names]
    return [move for move in moves if abs(move[1]) >= TRADE_THRESHOLD]


def _live_curve(nav, index: pd.DatetimeIndex) -> list:
    dates = index[RISK_LOOKBACK : RISK_LOOKBACK + len(nav)]
    equity = pd.Series(list(nav), index=dates)
    return _curve(equity)


def _direction(trades: list) -> str:
    if not trades:
        return "none"
    tech = sum(move for name, move in trades if name in TECH)
    return "cut" if tech < -TRADE_THRESHOLD else "add" if tech > TRADE_THRESHOLD else "swap"


def _reporter(agent, limited, prices: pd.DataFrame, progress):
    first, last = prices.index[0], prices.index[-1]
    span = max((last - first).days, 1)
    state = {"holding": dict(START_WEIGHTS), "feed": []}

    def report(entry: dict, context: dict) -> None:
        day = pd.Timestamp(entry["date"])
        done = min(max((day - first).days / span, 0.0), 1.0)
        plan = entry["plan"]
        reasons = [reason for attempt in entry["attempts"] for reason in attempt.get("reasons", []) or [attempt.get("error")]
                   if reason and not attempt.get("passed", False)]
        trades = _trades(state["holding"], plan) if plan else []
        item = {
            "date": f"{day:%Y-%m-%d}", "trigger": entry["trigger"], "attempts": len(entry["attempts"]),
            "outcome": "plan passed" if plan else "held",
            "weights": {name: round(weight, 4) for name, weight in plan.items() if weight > 0} if plan else None,
            "trades": trades, "direction": _direction(trades),
            "rejected": reasons[:3],
            "explanation": unmask_text(entry["explanation"])[:300] if plan else "",
        }
        if plan:
            state["holding"] = {name: weight for name, weight in plan.items() if weight > 0}
        state["feed"] = ([item] + state["feed"])[:FEED_LENGTH]
        progress({
            "day": item["date"], "percent": int(round(done * 100)), "decisions": len(agent.log),
            "plans_passed": sum(1 for e in agent.log if e["plan"]),
            "rejected_attempts": sum(1 for e in agent.log for a in e["attempts"] if not a.get("passed", False)),
            "calls": limited.calls, "max_calls": limited.max_calls,
            "holding": dict(state["holding"]), "feed": list(state["feed"]),
            "curve": _live_curve(context["_nav"], prices.index) if pd.Timestamp(context["date"]) >= pd.Timestamp(SCORE_START) else [],
        })

    return report


def _history(log: list) -> list:
    holding = dict(START_WEIGHTS)
    history = []
    for entry in log:
        day = f"{pd.Timestamp(entry['date']):%Y-%m-%d}"
        plan = entry["plan"]
        if plan:
            trades = _trades(holding, plan)
            history.append({
                "date": day, "outcome": "plan passed",
                "weights": {name: round(weight, 4) for name, weight in plan.items() if weight > 0},
                "trades": trades, "direction": _direction(trades),
            })
            holding = {name: weight for name, weight in plan.items() if weight > 0}
        else:
            history.append({"date": day, "outcome": "held", "weights": None, "trades": [], "direction": "none"})
    return list(reversed(history))


def _replay(manager, name: str, prices: pd.DataFrame):
    return risk.run_guarded(prices[TECH + CALM], START_WEIGHTS, LIMITS, manager=manager, calm=CALM, name=name,
                            cash_rate=CASH_RATE)


def baselines(prices: pd.DataFrame | None = None) -> list[dict]:
    """The bars to beat: the market, buy and hold, and the rule-based rotation. No model is involved."""
    prices = arena_prices() if prices is None else prices
    rows = []
    market = prices[MARKET].loc[SCORE_START:]
    market_returns = market.pct_change().dropna()
    hold = _replay(None, "Buy and hold", prices)
    rotate = _replay(risk.rotate_manager, "Rotate (rules)", prices)
    for result, label in ((hold, "Buy and hold"), (rotate, "Rotate (rules)")):
        rows.append({"name": label, "kind": "baseline", "metrics": _score(result, prices), "curve": _curve(result.equity)})
    spy_fit = alpha_beta(market_returns, market_returns)
    rows.append({
        "name": "The market (SPY)", "kind": "baseline",
        "metrics": {"alpha_annual": float(spy_fit.alpha_annual), "alpha_t": 0.0, "beta": 1.0, "months": int(spy_fit.months),
                    "reliable": False, "total_return": float(market.iloc[-1] / market.iloc[0] - 1),
                    "cagr": None, "sharpe": None, "max_drawdown": None, "trades": 0, "days_over_limit": None,
                    "worst_day": None, "costs": 0.0},
        "curve": _curve(market),
    })
    return rows


def run_entry(guidance: str, ask, prices: pd.DataFrame | None = None, max_calls: int = MAX_CALLS,
              max_seconds: float = MAX_SECONDS, progress=None) -> dict:
    """Replay the arena with a student's guidance and return everything the leaderboard shows.

    ask: function(prompt) -> reply text. The result holds the scores, a weekly equity curve (100 on the
    first scored day), the agent's own checks (plans rejected, made-up numbers), the first decisions and
    the holdings history (weights and trades only, newest first).
    progress: optional function(dict), called after each decision with the day reached, the share of the
    replay done, the counts so far and the agent's latest explanation.
    """
    guidance = clean_instructions(guidance)
    prices = arena_prices() if prices is None else prices
    limited = Limited(ask, guidance, max_calls=max_calls, max_seconds=max_seconds)
    agent = AgentManager(ask=limited, anonymous=True)
    if progress:
        agent.on_decision = _reporter(agent, limited, prices, progress)
    result = _replay(agent, "Agent", prices)
    log = agent.log
    rejected = sum(1 for entry in log for attempt in entry["attempts"] if not attempt.get("passed", False))
    decisions = [
        {
            "date": entry["date"], "trigger": entry["trigger"], "attempts": len(entry["attempts"]),
            "outcome": "plan passed" if entry["plan"] else "held",
            "explanation": entry["explanation"][:400],
        }
        for entry in log[:MAX_DECISIONS_SHOWN]
    ]
    return {
        "metrics": _score(result, prices),
        "agent": {
            "asked": len(log),
            "plans_passed": sum(1 for entry in log if entry["plan"]),
            "rejected_attempts": rejected,
            "made_up_numbers": sum(1 for entry in log if entry["unsupported_numbers"]),
            "calls": limited.calls,
            "budget_used": limited.stopped,
        },
        "curve": _curve(result.equity),
        "decisions": decisions,
        "holdings": _history(log),
    }


__all__ = [
    "BudgetUsed", "CALM", "LIMITS", "MAX_CALLS", "MAX_INSTRUCTION_CHARS", "SCORE_START", "START_WEIGHTS", "TECH",
    "arena_prices", "baselines", "clean_instructions", "mask_text", "masks", "run_entry",
]
