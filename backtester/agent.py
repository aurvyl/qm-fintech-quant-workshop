"""An AI risk manager: the model proposes, the code decides.

The model never trades and never computes a number. On a day a risk line is crossed it is
shown a snapshot of numbers our code computed (risk.build_context), and asked for new target
weights plus a short explanation, as JSON. Our code then:
  1. runs risk.check_plan on the proposal, the same test the rule-based manager must pass;
  2. if it fails, sends the reasons back and lets the model try again (at most 3 attempts);
  3. if nothing passes, holds;
  4. checks that every number in the explanation really appears in the snapshot.

ask is any function: prompt text in, model reply text out. workshop_model() builds one for the
workshop's own server (an access code, no key), and gemini() one for Google's free Gemini API. offline_model() is a stand-in that is NOT AI: it answers with the rule-based
plan, so the whole loop runs in class with no key and no internet. cached() stores every
reply on disk by its prompt, so a backtest gives the same answer twice and costs nothing to rerun.
"""

import hashlib
import json
import re
import string
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .risk import check_plan, rotate_manager

MAX_ATTEMPTS = 3
GEMINI_MODEL = "gemini-flash-latest"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
REQUEST_TIMEOUT = 60
RETRY_STATUSES = {429, 500, 503}
MAX_RETRIES = 2

INSTRUCTIONS = """You are a risk manager for a student's paper portfolio. Nothing you say trades real money.
The investor has set limits on how far the portfolio may fall. A warning line was crossed today.

Your job: propose new target weights that bring the one-day VaR within the budget, by ROTATING weight
from the holdings with the largest risk_share into the calm_candidates, not by selling everything.

Rules:
- Use only tickers listed in "allowed". Weights are fractions of the portfolio, in steps of 0.05,
  none above 0.40, adding up to at most 1. Whatever is left over is cash.
- Use only the numbers in the snapshot. Do not use anything you know about what happened after {date}.
- Prefer the smallest change that fixes the problem.
- If you think no change is needed, return the current weights.

Answer with JSON only, in exactly this shape:
{{"weights": {{"TICKER": 0.20, "...": 0.10}}, "explanation": "under 80 words, plain English, quoting numbers from the snapshot"}}

Snapshot:
{snapshot}
"""

RETRY = """Your plan failed the risk check for these reasons:
{reasons}
Fix it and answer again with JSON only, in the same shape."""


def snapshot(context: dict) -> dict:
    """The numbers the model may see: everything in the context except the private raw data."""
    clean = {k: v for k, v in context.items() if not k.startswith("_")}
    return json.loads(json.dumps(clean, default=_rounded), parse_float=lambda x: round(float(x), 4))


def _rounded(value):
    if isinstance(value, float):
        return round(value, 4)
    return str(value)


def anonymise(context: dict) -> tuple[dict, dict]:
    """Hide the date and the ticker names, so the model cannot recall what happened next.

    Returns (masked context, mapping from mask back to ticker). Used for the look-ahead experiment.
    """
    tickers = context["allowed"]
    masks = {t: f"ASSET_{string.ascii_uppercase[i]}" for i, t in enumerate(tickers)}

    def swap(value):
        if isinstance(value, dict):
            return {masks.get(k, k): swap(v) for k, v in value.items()}
        if isinstance(value, list):
            return [masks.get(v, v) if isinstance(v, str) else swap(v) for v in value]
        if isinstance(value, str):
            for ticker, mask in masks.items():
                value = re.sub(rf"\b{re.escape(ticker)}\b", mask, value)
            return value
        return value

    masked = {k: (v if k.startswith("_") else swap(v)) for k, v in context.items()}
    masked["date"] = "day T"
    masked["_returns"] = context["_returns"].rename(columns=masks)
    return masked, {mask: ticker for ticker, mask in masks.items()}


def parse_reply(text: str) -> dict:
    """Pull the JSON object out of a model reply, even if it is wrapped in ``` fences or prose."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("The reply contained no JSON object.")
    reply = json.loads(match.group(0))
    weights = reply.get("weights")
    if not isinstance(weights, dict):
        raise ValueError('The reply has no "weights" object.')
    return {"weights": {str(k): float(v) for k, v in weights.items()}, "explanation": str(reply.get("explanation", ""))}


NUMBER = re.compile(r"-?\d+(?:\.\d+)?%?")


def numbers_in(snapshot_numbers: dict) -> set[float]:
    """Every number in a snapshot, in the forms a person might write it (0.1234 -> 0.1234, 0.12, 12.34, 12.3, 12)."""
    found = set()

    def walk(value):
        if isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            for x in (value, value * 100):
                for digits in (0, 1, 2, 3, 4):
                    found.add(round(abs(x), digits))

    walk(snapshot_numbers)
    return found


def unsupported_numbers(explanation: str, snapshot_numbers: dict, weights: dict | None = None) -> list[str]:
    """Numbers in the explanation that appear nowhere in the snapshot or the proposed weights.

    The proposed weights, their total and the cash left over (1 minus the total) count as known.
    A non-empty list means the model stated a figure our code never gave it: treat it as made up.
    Small whole numbers (0 to 10) are ignored, as they are usually counts ("3 holdings").
    """
    known = numbers_in({"s": snapshot_numbers, "w": weights or {}})
    if weights:
        invested = sum(weights.values())
        known |= numbers_in({"invested": invested, "cash": max(1.0 - invested, 0.0)})
    flagged = []
    for token in NUMBER.findall(explanation):
        value = abs(float(token.rstrip("%")))
        if value <= 10 and value == int(value) and not token.endswith("%"):
            continue
        decimals = len(token.rstrip("%").split(".")[1]) if "." in token else 0
        if round(value, decimals) not in known:
            flagged.append(token)
    return flagged


@dataclass
class AgentManager:
    """A risk manager backed by a language model. Pass it to risk.run_guarded as `manager`.

    ask:          function(prompt) -> reply text.
    anonymous:    hide dates and ticker names from the model (look-ahead experiment).
    max_attempts: how many proposals the model gets before we hold.
    log:          one entry per decision: the attempts, the checks, the explanation and any made-up numbers.
    on_decision:  optional function(entry, context), called after each decision.
    """

    ask: object
    anonymous: bool = False
    max_attempts: int = MAX_ATTEMPTS
    log: list = field(default_factory=list)
    on_decision: object = None

    def __call__(self, context: dict) -> dict | None:
        shown, back = anonymise(context) if self.anonymous else (context, None)
        numbers = snapshot(shown)
        prompt = INSTRUCTIONS.format(date=numbers["date"], snapshot=json.dumps(numbers, indent=1))
        entry = {"date": context["date"], "trigger": context["trigger"], "attempts": [], "plan": None,
                 "explanation": "", "unsupported_numbers": []}
        conversation = prompt
        for _ in range(self.max_attempts):
            try:
                reply = parse_reply(self.ask(conversation))
            except (ValueError, json.JSONDecodeError) as error:
                entry["attempts"].append({"error": str(error)})
                conversation = prompt + "\n\n" + RETRY.format(reasons=f"- {error}")
                continue
            weights = reply["weights"]
            if back:
                weights = {back.get(k, k): v for k, v in weights.items()}
            check = check_plan(weights, context["_returns"], context["_lines"], context["allowed"])
            entry["attempts"].append({"weights": weights, "passed": check.passed, "reasons": check.reasons})
            if check.passed:
                entry["plan"] = weights
                entry["explanation"] = reply["explanation"]
                entry["unsupported_numbers"] = unsupported_numbers(reply["explanation"], numbers, reply["weights"])
                break
            conversation = prompt + "\n\n" + RETRY.format(reasons="\n".join(f"- {r}" for r in check.reasons))
        self.log.append(entry)
        if self.on_decision:
            self.on_decision(entry, context)
        return entry["plan"]

    def decisions(self) -> pd.DataFrame:
        """The log as a table: one row per day the model was asked."""
        rows = [
            {
                "date": e["date"],
                "trigger": e["trigger"],
                "attempts": len(e["attempts"]),
                "outcome": "plan passed" if e["plan"] else "hold",
                "explanation": e["explanation"],
                "unsupported numbers": ", ".join(e["unsupported_numbers"]),
            }
            for e in self.log
        ]
        return pd.DataFrame(rows, columns=["date", "trigger", "attempts", "outcome", "explanation", "unsupported numbers"])


def _paced(seconds_between_calls: float, name: str, post):
    """Wrap post(prompt) so calls are spaced out and a busy reply is retried at most twice."""
    last_call = [0.0]

    def ask(prompt: str) -> str:
        for attempt in range(MAX_RETRIES + 1):
            wait = last_call[0] + seconds_between_calls - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            last_call[0] = time.monotonic()
            try:
                return post(prompt)
            except urllib.error.HTTPError as error:
                if error.code not in RETRY_STATUSES or attempt == MAX_RETRIES:
                    raise RuntimeError(_http_message(name, error)) from None
                time.sleep(10 * 2**attempt)
        raise RuntimeError(f"{name} did not answer.")

    return ask


def _http_message(name: str, error) -> str:
    try:
        detail = json.loads(error.read().decode()).get("error")
    except Exception:
        detail = None
    if isinstance(detail, str) and detail:
        return f"{name}: {detail}"
    return f"{name} answered with HTTP {error.code}. Check the key or access code."


def _post_json(url: str, payload: dict, headers: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        return json.load(response)


def gemini(api_key: str, model: str = GEMINI_MODEL, temperature: float = 0.0, seconds_between_calls: float = 4.0):
    """An `ask` function for Google's Gemini API. Get a free key at https://aistudio.google.com/apikey.

    Never paste the key into a notebook cell you save or share. In Colab, store it under the key icon
    (Secrets) as GEMINI_API_KEY and read it with google.colab.userdata.get("GEMINI_API_KEY").
    Calls are spaced at least `seconds_between_calls` apart to stay inside the free tier's rate limit,
    and a busy or rate-limited reply is retried at most twice, waiting longer each time.
    """
    if not api_key:
        raise ValueError("No API key given. Use offline_model() to run without one.")
    url = GEMINI_URL.format(model=model)

    def post(prompt: str) -> str:
        reply = _post_json(url, {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
        }, {"x-goog-api-key": api_key})
        return reply["candidates"][0]["content"]["parts"][0]["text"]

    return _paced(seconds_between_calls, "Gemini", post)


def workshop_model(url: str, code: str, seconds_between_calls: float = 1.0):
    """An `ask` function for the workshop's own model server. Your organiser gives you the URL and a code.

    The code is not a key: it only lets you use the workshop server, which holds the real key and
    sets the model and its limits. Keep it in Colab Secrets as WORKSHOP_CODE all the same.
    """
    if not url or not code:
        raise ValueError("No workshop URL or access code given. Use offline_model() to run without one.")

    def post(prompt: str) -> str:
        return _post_json(url, {"code": code, "prompt": prompt}, {})["text"]

    return _paced(seconds_between_calls, "The workshop model", post)


def offline_model():
    """A stand-in for a model, for running the loop with no key. It is NOT AI.

    It reads the snapshot in the prompt and answers with the rule-based plan (risk.rotate_manager),
    in the same JSON shape a model would use. Results with it equal the rule-based results.
    """
    contexts = {}

    def remember(context: dict) -> None:
        contexts[context["date"]] = context

    def ask(prompt: str) -> str:
        date = re.search(r'"date": "([^"]+)"', prompt).group(1)
        context = contexts[date]
        plan = rotate_manager(context) or context["weights"]
        changes = {t: plan.get(t, 0.0) - context["weights"].get(t, 0.0) for t in set(plan) | set(context["weights"])}
        sold = sorted((t for t, d in changes.items() if d <= -0.025), key=changes.get)
        bought = sorted((t for t, d in changes.items() if d >= 0.025), key=lambda t: -changes[t])
        text = f"One-day VaR is {context['one_day_var']:.2%} against a budget of {context['limits']['var_budget']:.2%}. "
        if sold and bought:
            text += f"Moving weight from {', '.join(sold)} into {', '.join(bought)}."
        else:
            text += "Keeping the current weights."
        return json.dumps({"weights": plan, "explanation": text})

    ask.remember = remember
    return ask


class OfflineAgent(AgentManager):
    """AgentManager wired to offline_model(), so it sees each day's context before answering."""

    def __init__(self, **kwargs):
        model = offline_model()
        super().__init__(ask=model, **kwargs)
        self._model = model

    def __call__(self, context: dict) -> dict | None:
        self._model.remember(anonymise(context)[0] if self.anonymous else context)
        return super().__call__(context)


def cached(ask, folder="data/agent_cache"):
    """Wrap an `ask` function so each distinct prompt is sent once and its reply saved to `folder`.

    Rerunning a backtest then repeats the same answers, costs nothing and works offline.
    Delete the folder to ask again.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    def ask_cached(prompt: str) -> str:
        key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        path = folder / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())["reply"]
        reply = ask(prompt)
        path.write_text(json.dumps({"reply": reply}))
        return reply

    return ask_cached
