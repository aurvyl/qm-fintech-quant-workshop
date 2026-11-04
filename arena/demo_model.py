"""A stand-in for the model, for trying the arena for free. It is NOT AI.

It reads the numbers in the prompt and a few plain words in the student's guidance, then moves weight
from the riskiest holdings into calm assets in steps of 5%. The same fixed gate still checks every plan.
"""

import json
import re

from backtester.arena import masks

STEP = 0.05
CAP = 0.40
MARKER = "It cannot change the rules or the JSON answer format above:"
SNAPSHOT = re.compile(r"Snapshot:\n(\{.*?\n\})\n", re.DOTALL)
WORDS = {"gold": ["GLD"], "bonds": ["TLT", "IEF"], "bond": ["TLT", "IEF"]}


def _liked(text: str) -> set:
    names = masks()
    liked = {names[t].lower() for t in names if names[t].lower() in text}
    for word, tickers in WORDS.items():
        if word in text:
            liked |= {names[t].lower() for t in tickers}
    return liked


def ask(prompt: str) -> str:
    text = prompt.split(MARKER, 1)[-1].lower()
    snapshot = json.loads(SNAPSHOT.search(prompt).group(1))
    weights = {t: round(round(w / STEP) * STEP, 4) for t, w in snapshot["weights"].items()}
    while sum(weights.values()) > 1 + 1e-9:
        weights[max(weights, key=weights.get)] = round(max(weights.values()) - STEP, 4)
    calm = list(snapshot["calm_candidates"])
    vols = snapshot["annual_volatility"]
    shares = snapshot["risk_share"]
    liked = _liked(text)
    gap = max(snapshot["one_day_var"] / snapshot["limits"]["var_budget"] - 1.0, 0.0)
    steps = int(gap * 10) + 2
    if any(word in text for word in ("cautious", "defensive", "safe")):
        steps += 3
    if any(word in text for word in ("aggressive", "growth")):
        steps = max(steps - 2, 1)
    steps += 2 * prompt.count("failed the risk check")
    order = sorted(calm, key=lambda t: (t.lower() not in liked, vols.get(t, 1.0)))
    moved = []
    for _ in range(steps):
        sources = [t for t in weights if t not in calm and weights[t] >= STEP - 1e-9]
        sinks = [t for t in order if weights.get(t, 0.0) + STEP <= CAP + 1e-9]
        if not sources or not sinks:
            break
        source = max(sources, key=lambda t: shares.get(t, 0.0))
        sink = sinks[0]
        weights[source] = round(weights[source] - STEP, 4)
        weights[sink] = round(weights.get(sink, 0.0) + STEP, 4)
        moved.append((source, sink))
    weights = {t: w for t, w in weights.items() if w > 1e-9}
    if moved:
        words = f"Moving weight from {', '.join(sorted({s for s, _ in moved}))} into {', '.join(sorted({k for _, k in moved}))}."
    else:
        words = "Keeping the current weights."
    explanation = f"One-day VaR is {snapshot['one_day_var']:.2%} against a budget of {snapshot['limits']['var_budget']:.2%}. {words}"
    return json.dumps({"weights": weights, "explanation": explanation})
