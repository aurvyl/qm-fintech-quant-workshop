"""Risk rules for a whole portfolio, and a day-by-day simulator that obeys them.

An investor states two limits in plain words:
    "never fall more than 10% below my peak"   -> Limits(max_drawdown=0.10)
    "never lose more than 2% in one day"       -> Limits(max_daily_loss=0.02)

Limits are turned into earlier warning lines (lines_from_limits), because by the time
a limit is hit it is too late to act. On each day's close the simulator checks the lines
(evaluate). When one is crossed, a risk manager proposes new weights. The proposal must
pass check_plan, the same fixed test whoever wrote it, a rule or an AI model. Trades fill
at the next close and pay a cost on turnover.

The manager is any function: context dict in, target weights (dict) or None out.
rotate_manager is the rule-based one. backtester.agent builds one that asks an AI model.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS, compute_metrics, drawdown_series, format_metrics

VAR_Z_99 = 2.326
STEP = 0.05
MAX_WEIGHT = 0.40
RISK_LOOKBACK = 60
COOLDOWN_DAYS = 5
RERISK_AFTER_DAYS = 10
STAY_OUT_DAYS = 60
MAX_STEPS = 12
MIN_TURNOVER = 0.02

CRISES = {
    "2020 Covid crash": ("2020-02-19", "2020-08-18"),
    "2022 bear market": ("2022-01-03", "2022-12-30"),
}


@dataclass
class Limits:
    """The investor's two rules for the whole portfolio, as fractions (0.10 = 10%)."""

    max_drawdown: float = 0.10
    max_daily_loss: float = 0.02


@dataclass
class Lines:
    """The numbers the simulator actually watches, derived from the investor's limits."""

    rotate_at: float
    emergency_at: float
    var_budget: float
    shock_day: float


def lines_from_limits(limits: Limits) -> Lines:
    """Early warning lines: start trimming risk at half the drawdown budget, cut hard at 80% of it.

    var_budget: the largest one-day loss we accept as likely (99% one-day VaR), set to the daily limit.
    shock_day:  a single day losing more than the daily limit is itself a trigger.
    """
    if not 0 < limits.max_drawdown < 1 or not 0 < limits.max_daily_loss < 1:
        raise ValueError("Limits are fractions between 0 and 1, e.g. 0.10 for 10%.")
    return Lines(
        rotate_at=0.5 * limits.max_drawdown,
        emergency_at=0.8 * limits.max_drawdown,
        var_budget=limits.max_daily_loss,
        shock_day=limits.max_daily_loss,
    )


def one_day_var(weights: pd.Series, returns: pd.DataFrame, lookback: int = RISK_LOOKBACK, z: float = VAR_Z_99) -> float:
    """99% one-day Value at Risk: a loss we expect to beat on only about 1 day in 100.

    Parametric version: z x portfolio volatility, using the covariance of the last `lookback` days.
    0.02 means "on a bad day (1 in 100) expect to lose about 2% or more".
    """
    weights = weights[weights.abs() > 0]
    if weights.empty:
        return 0.0
    window = returns[weights.index].iloc[-lookback:].dropna()
    if len(window) < 2:
        return float("nan")
    cov = window.cov().to_numpy()
    w = weights.to_numpy()
    return float(z * np.sqrt(max(w @ cov @ w, 0.0)))


def historical_var(returns: pd.Series, confidence: float = 0.99) -> float:
    """The loss exceeded on only (1 - confidence) of past days, read straight from history."""
    return float(-np.quantile(returns.dropna(), 1 - confidence))


def expected_shortfall(returns: pd.Series, confidence: float = 0.99) -> float:
    """The average loss on the days that were worse than the VaR: how bad the bad days really were."""
    clean = returns.dropna()
    cutoff = np.quantile(clean, 1 - confidence)
    return float(-clean[clean <= cutoff].mean())


def annual_vols(returns: pd.DataFrame, lookback: int = RISK_LOOKBACK) -> pd.Series:
    """Each asset's recent volatility, scaled to a year."""
    return returns.iloc[-lookback:].std() * np.sqrt(TRADING_DAYS)


def risk_contributions(weights: pd.Series, returns: pd.DataFrame, lookback: int = RISK_LOOKBACK) -> pd.Series:
    """How much of the portfolio's risk each holding is responsible for. Shares add up to 1."""
    weights = weights[weights > 0]
    if weights.empty:
        return pd.Series(dtype=float)
    cov = returns[weights.index].iloc[-lookback:].cov()
    marginal = cov @ weights
    total = float(weights @ marginal)
    if total <= 0:
        return pd.Series(0.0, index=weights.index)
    return weights * marginal / total


@dataclass
class Check:
    """The verdict on a proposed set of weights."""

    passed: bool
    reasons: list[str] = field(default_factory=list)
    numbers: dict = field(default_factory=dict)


def check_plan(weights: dict, returns: pd.DataFrame, lines: Lines, allowed: list[str]) -> Check:
    """The fixed acceptance test every proposal must pass before it trades.

    Rules: only allowed tickers, no negative weights, total at most 100%, no holding above 40%,
    weights in steps of 5%, and 99% one-day VaR within the daily-loss budget.
    """
    reasons = []
    unknown = sorted(set(weights) - set(allowed))
    if unknown:
        reasons.append(f"Unknown tickers: {unknown}.")
    series = pd.Series({t: float(w) for t, w in weights.items() if t in allowed}, dtype=float)
    if (series < 0).any():
        reasons.append("Weights must not be negative.")
    total = float(series.sum())
    if total > 1 + 1e-6:
        reasons.append(f"Weights add up to {total:.0%}, above 100%.")
    largest = float(series.max()) if len(series) else 0.0
    if largest > MAX_WEIGHT + 1e-9:
        reasons.append(f"Largest holding is {largest:.0%}, above the {MAX_WEIGHT:.0%} cap.")
    off_grid = [t for t, w in series.items() if abs(w / STEP - round(w / STEP)) > 1e-6]
    if off_grid:
        reasons.append(f"Weights must move in {STEP:.0%} steps; not true for {off_grid}.")
    var = one_day_var(series.clip(lower=0), returns) if not reasons else float("nan")
    if np.isfinite(var) and var > lines.var_budget:
        reasons.append(f"One-day VaR is {var:.2%}, above the {lines.var_budget:.2%} budget.")
    numbers = {"one_day_var": var, "largest_weight": largest, "total_weight": total}
    return Check(passed=not reasons, reasons=reasons, numbers=numbers)


def evaluate(drawdown: float, day_return: float, var: float, lines: Lines) -> tuple[str | None, list[str]]:
    """Which line, if any, today's close has crossed. Returns (kind, reasons).

    kind is "emergency", "rotate" or None.
    """
    reasons = []
    if drawdown <= -lines.emergency_at:
        return "emergency", [f"Drawdown {drawdown:.1%} is past the emergency line of -{lines.emergency_at:.1%}."]
    if drawdown <= -lines.rotate_at:
        reasons.append(f"Drawdown {drawdown:.1%} is past the rotate line of -{lines.rotate_at:.1%}.")
    if day_return <= -lines.shock_day:
        reasons.append(f"Today's loss of {day_return:.1%} is beyond the daily limit of -{lines.shock_day:.1%}.")
    if np.isfinite(var) and var > lines.var_budget:
        reasons.append(f"One-day VaR {var:.2%} is above the {lines.var_budget:.2%} budget.")
    return ("rotate" if reasons else None), reasons


def build_context(day, weights: pd.Series, returns: pd.DataFrame, nav: pd.Series, lines: Lines,
                  kind: str | None, reasons: list[str], calm: list[str], start_weights: pd.Series,
                  drawdown: float | None = None) -> dict:
    """Everything a risk manager may look at on one day. Every number here was computed by this code."""
    universe = list(returns.columns)
    vols = annual_vols(returns)
    contributions = risk_contributions(weights, returns)
    return {
        "date": pd.Timestamp(day).strftime("%Y-%m-%d"),
        "trigger": kind,
        "reasons": reasons,
        "drawdown": float(nav.iloc[-1] / nav.max() - 1) if drawdown is None else drawdown,
        "today_return": float(nav.iloc[-1] / nav.iloc[-2] - 1) if len(nav) > 1 else 0.0,
        "one_day_var": one_day_var(weights, returns),
        "weights": {t: float(w) for t, w in weights.items() if w > 1e-9},
        "start_weights": {t: float(w) for t, w in start_weights.items() if w > 0},
        "annual_volatility": {t: float(vols[t]) for t in universe},
        "risk_share": {t: float(c) for t, c in contributions.items()},
        "last_5d_return": {t: float((1 + returns[t].iloc[-5:]).prod() - 1) for t in universe},
        "calm_candidates": list(calm),
        "allowed": universe,
        "limits": {"var_budget": lines.var_budget, "rotate_at": lines.rotate_at, "emergency_at": lines.emergency_at},
        "_returns": returns,
        "_lines": lines,
    }


def _round_to_step(weights: pd.Series) -> pd.Series:
    return (weights / STEP).round() * STEP


def rotate_manager(context: dict) -> dict | None:
    """The rule-based risk manager. Rotates, never panic-sells.

    Moves 5% at a time from the holding contributing the most risk into the calmest candidate
    (lowest recent volatility) that still has room under the 40% cap, until check_plan passes.
    In an emergency it keeps going until VaR is under half the budget. Returns None (hold) if
    no passing plan is found within 12 steps.
    """
    returns, lines = context["_returns"], context["_lines"]
    allowed = context["allowed"]
    weights = _round_to_step(pd.Series(context["weights"], dtype=float).reindex(allowed).fillna(0.0))
    vols = pd.Series(context["annual_volatility"])
    calm = sorted(context["calm_candidates"], key=lambda t: vols.get(t, np.inf))
    target_var = lines.var_budget * (0.5 if context["trigger"] == "emergency" else 0.9)
    for _ in range(MAX_STEPS):
        plan = {t: round(float(w), 4) for t, w in weights.items() if w > 1e-9}
        check = check_plan(plan, returns, lines, allowed)
        if check.passed and check.numbers["one_day_var"] <= target_var:
            return plan
        contributions = risk_contributions(weights, returns).drop(calm, errors="ignore")
        contributions = contributions[weights.reindex(contributions.index) >= STEP - 1e-9]
        if contributions.empty:
            break
        source = contributions.idxmax()
        room = [t for t in calm if weights[t] + STEP <= MAX_WEIGHT + 1e-9]
        if not room:
            break
        weights[source] -= STEP
        weights[room[0]] += STEP
    final = {t: round(float(w), 4) for t, w in weights.items() if w > 1e-9}
    return final if check_plan(final, returns, lines, allowed).passed else None


def cash_exit_manager(context: dict) -> dict | None:
    """The panic button: on the first real loss past a line, sell everything to cash and never come back.

    It ignores the VaR warning and reacts only to losses: the drawdown or the day's fall.
    """
    lines = context["_lines"]
    losing = context["drawdown"] <= -lines.rotate_at or context["today_return"] <= -lines.shock_day
    return {} if losing else None


def exit_and_return_manager(context: dict) -> dict | None:
    """The fairer panic button: sell everything on the first real loss, wait, then buy back.

    It reacts exactly like cash_exit_manager, but run_guarded puts the portfolio back to its
    start weights after STAY_OUT_DAYS trading days (about three months) in cash, whatever the market did.
    """
    return cash_exit_manager(context)


@dataclass
class GuardedResult:
    """What a guarded portfolio did, day by day, and why."""

    name: str
    equity: pd.Series
    weights: pd.DataFrame
    log: pd.DataFrame
    benchmark_equity: pd.Series
    metrics: dict

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({self.name: format_metrics(self.metrics)})


def run_guarded(
    closes: pd.DataFrame,
    start_weights: dict,
    limits: Limits = Limits(),
    manager=rotate_manager,
    calm: list[str] | None = None,
    cost_bps: float = 5.0,
    start_cash: float = 10_000.0,
    name: str = "Guarded",
    warmup: int = RISK_LOOKBACK,
    cash_rate: float = 0.0,
) -> GuardedResult:
    """Run a portfolio day by day under the investor's limits.

    closes:        prices, one column per ticker. The first `warmup` days only feed the risk numbers.
    start_weights: the portfolio on day one, e.g. {"AAPL": 0.25, "MSFT": 0.25, ...}.
    manager:       function(context) -> target weights dict, {} for all cash, or None to hold.
                   Called only on days a line is crossed, at most once every 5 days.
                   manager=None means plain buy and hold: no checks, no trades.
    calm:          tickers the manager may rotate into. Default: anything not in start_weights.
    After an emergency cut the peak is reset to the portfolio's value that day, so the limit then
    counts from the new, smaller portfolio instead of keeping it defensive for ever.
    cash_rate:     yearly interest on money held in cash (0.03 = 3%). The default is 0. Sharpe is measured above it.
    Re-risking: after 10 quiet days with the drawdown back above half the rotate line, the
    portfolio steps 5% back towards its start weights, if check_plan allows it.
    The manager's plan is always re-checked here; a failing plan does not trade.
    """
    closes = closes.ffill().dropna()
    if len(closes) <= warmup + 1:
        raise ValueError(f"Need more than {warmup + 1} days of prices: the first {warmup} only warm up the risk numbers.")
    allowed = list(closes.columns)
    start = pd.Series(start_weights, dtype=float).reindex(allowed).fillna(0.0)
    calm = [t for t in allowed if start[t] == 0] if calm is None else list(calm)
    lines = lines_from_limits(limits)
    returns = closes.pct_change()

    cash_daily = (1 + cash_rate) ** (1 / TRADING_DAYS) - 1
    held = start.copy()
    nav = [start_cash]
    dates = closes.index[warmup:]
    weight_rows, log_rows = [held.copy()], []
    pending, last_trade, quiet, peak, out_since = None, -COOLDOWN_DAYS - 1, 0, start_cash, 0
    for i, day in enumerate(dates):
        position = warmup + i
        cost = 0.0
        if pending is not None:
            turnover = float((pending - held).abs().sum())
            cost = turnover * cost_bps / 10_000
            held = pending
            pending = None
        if i > 0:
            r = returns.iloc[position].fillna(0.0)
            idle = float(1 - held.sum())
            daily = float((held * r).sum()) + idle * cash_daily - cost
            nav.append(nav[-1] * (1 + daily))
            grown = held * (1 + r)
            total = float(grown.sum() + idle * (1 + cash_daily))
            held = grown / total if total > 0 else held * 0
            weight_rows.append(held.copy())
        history = returns.iloc[position - warmup + 1 : position + 1]
        nav_series = pd.Series(nav)
        peak = max(peak, nav[-1])
        drawdown = float(nav[-1] / peak - 1)
        day_return = float(nav_series.iloc[-1] / nav_series.iloc[-2] - 1) if len(nav) > 1 else 0.0
        var = one_day_var(held, history)
        kind, reasons = evaluate(drawdown, day_return, var, lines)
        cooling = i - last_trade <= COOLDOWN_DAYS
        if manager is _wait_then_return and i - out_since >= STAY_OUT_DAYS and pending is None:
            pending = start.copy()
            peak = nav[-1]
            last_trade = i
            quiet = 0
            manager = exit_and_return_manager
            log_rows.append({"date": day, "trigger": None, "reasons": f"{STAY_OUT_DAYS} trading days in cash.", "action": "back in",
                             "one_day_var": var, "turnover": float((start - held).abs().sum())})
            continue
        if manager is None:
            continue
        if kind and not cooling:
            quiet = 0
            context = build_context(day, held, history, nav_series, lines, kind, reasons, calm, start, drawdown)
            plan = manager(context)
            row = {"date": day, "trigger": kind, "reasons": " ".join(reasons), "action": "hold", "one_day_var": var}
            if plan is not None:
                check = check_plan(plan, history, lines, allowed) if plan else Check(passed=True)
                target = pd.Series(plan, dtype=float).reindex(allowed).fillna(0.0)
                turnover = float((target - held).abs().sum())
                if check.passed and turnover > MIN_TURNOVER:
                    pending = target
                    row["action"] = "all cash" if not plan else "rotate"
                    row["turnover"] = turnover
                    if kind == "emergency":
                        peak = nav[-1]
                elif check.passed:
                    row["action"] = "hold: plan matches the portfolio"
                else:
                    row["action"] = "rejected: " + " ".join(check.reasons)
            last_trade = i
            log_rows.append(row)
            if manager is cash_exit_manager and plan == {}:
                manager = _stay_out
            elif manager is exit_and_return_manager and plan == {}:
                manager = _wait_then_return
                out_since = i
        elif not kind:
            quiet += 1
            if quiet >= RERISK_AFTER_DAYS and drawdown > -lines.rotate_at / 2 and manager not in (_stay_out, _wait_then_return, cash_exit_manager, exit_and_return_manager):
                step = _rerisk_step(held, start)
                if step is not None and check_plan(_as_plan(step), history, lines, allowed).passed:
                    pending = step
                    last_trade = i
                    quiet = 0
                    log_rows.append({"date": day, "trigger": None, "reasons": f"Quiet for {RERISK_AFTER_DAYS} days.", "action": "re-risk 5%",
                                     "one_day_var": var, "turnover": float((step - held).abs().sum())})

    equity = pd.Series(nav, index=dates)
    weights = pd.DataFrame(weight_rows, index=dates)
    hold_equity = start_cash * (closes.loc[dates] / closes.loc[dates].iloc[0] * start).sum(axis=1) / max(start.sum(), 1e-12)
    log = pd.DataFrame(log_rows, columns=["date", "trigger", "reasons", "action", "one_day_var", "turnover"])
    metrics = guard_metrics(equity, limits, log, cost_bps, cash_rate)
    return GuardedResult(name=name, equity=equity, weights=weights, log=log, benchmark_equity=hold_equity, metrics=metrics)


def _stay_out(context: dict) -> dict | None:
    return None


def _wait_then_return(context: dict) -> dict | None:
    return None


def _as_plan(weights: pd.Series) -> dict:
    return {t: round(float(w), 4) for t, w in weights.items() if w > 1e-9}


def _rerisk_step(held: pd.Series, start: pd.Series) -> pd.Series | None:
    rounded = _round_to_step(held)
    gap = start - rounded
    if gap.abs().max() < STEP - 1e-9:
        return None
    step = rounded.copy()
    step[gap.idxmax()] += STEP
    step[gap.idxmin()] -= STEP
    return step if (step >= -1e-9).all() else None


def guard_metrics(equity: pd.Series, limits: Limits, log: pd.DataFrame, cost_bps: float, cash_rate: float = 0.0) -> dict:
    """The usual metrics plus the ones an investor with limits cares about."""
    returns = equity.pct_change().iloc[1:]
    metrics = compute_metrics(equity, returns, risk_free_rate=cash_rate)
    metrics.pop("Win rate")
    trades = log[log["action"].isin(["rotate", "all cash", "re-risk 5%", "back in"])] if len(log) else log
    metrics["Number of trades"] = len(trades)
    metrics["Days over daily limit"] = int((returns <= -limits.max_daily_loss).sum())
    metrics["Worst day"] = float(returns.min()) if len(returns) else float("nan")
    metrics["Costs paid"] = float(trades["turnover"].fillna(0).sum() * cost_bps / 10_000) if len(trades) else 0.0
    return metrics


def compare(results: list) -> pd.DataFrame:
    """Several guarded results side by side, formatted for reading."""
    table = pd.DataFrame({r.name: r.metrics for r in results})
    out = table.astype(object)
    for row in table.index:
        for col in table.columns:
            value = table.loc[row, col]
            if row in ("Number of trades", "Days over daily limit"):
                out.loc[row, col] = f"{int(value)}"
            elif row == "Sharpe ratio":
                out.loc[row, col] = "n/a" if pd.isna(value) else f"{value:.2f}"
            else:
                out.loc[row, col] = "n/a" if pd.isna(value) else f"{value:.1%}"
    return out


def crisis_table(equities: dict[str, pd.Series], crises: dict = CRISES) -> pd.DataFrame:
    """Return and worst fall of each portfolio inside each crisis window."""
    rows = {}
    for label, (start, end) in crises.items():
        for name, equity in equities.items():
            window = equity.loc[start:end]
            if len(window) < 2:
                continue
            rows[(label, name)] = {
                "Return": f"{window.iloc[-1] / window.iloc[0] - 1:.1%}",
                "Worst fall": f"{drawdown_series(window).min():.1%}",
            }
    table = pd.DataFrame(rows).T
    table.index.names = ["Crisis", "Portfolio"]
    return table


def capm_split(portfolio_returns: pd.Series, market_returns: pd.Series, day, lookback: int = 126) -> dict:
    """Split one day's move into the market part and the part that is down to our own holdings.

    beta is estimated on the `lookback` days BEFORE `day` only, so the split uses no future data.
    market part = beta x market return that day; own part = the rest.
    """
    day = pd.Timestamp(day)
    joined = pd.concat({"p": portfolio_returns, "m": market_returns}, axis=1).dropna()
    past = joined.loc[: day - pd.Timedelta(days=1)].iloc[-lookback:]
    if len(past) < 20 or day not in joined.index:
        raise ValueError("Need at least 20 earlier days and a return on that day.")
    beta = float(past["p"].cov(past["m"]) / past["m"].var())
    total, market = float(joined.loc[day, "p"]), float(joined.loc[day, "m"])
    return {"beta": beta, "total": total, "market_part": beta * market, "own_part": total - beta * market}


def unusual_moves(returns: pd.DataFrame, day, lookback: int = 60) -> pd.DataFrame:
    """How unusual each asset's move was on `day`: a z-score against the previous `lookback` days.

    |z| above 3 is rare in calm times. spike_ratio compares the last 20 days' volatility
    with the last 120: above 1.5 means risk has jumped recently.
    """
    day = pd.Timestamp(day)
    past = returns.loc[: day - pd.Timedelta(days=1)]
    today = returns.loc[day]
    window = past.iloc[-lookback:]
    z = (today - window.mean()) / window.std()
    spike = past.iloc[-20:].std() / past.iloc[-120:].std()
    return pd.DataFrame({"return": today, "z_score": z, "spike_ratio": spike}).sort_values("z_score")


def what_if(closes: pd.DataFrame, day, shocks: dict) -> pd.DataFrame:
    """A copy of the prices where, on `day`, each ticker in `shocks` moved by that amount instead.

    shocks: e.g. {"AAPL": -0.12} for "what if Apple had fallen 12% that day". Later prices move
    with it, so the rest of history keeps its own daily returns.
    """
    day = pd.Timestamp(day)
    if day not in closes.index:
        raise ValueError(f"{day:%Y-%m-%d} is not a trading day in these prices.")
    shocked = closes.copy()
    position = closes.index.get_loc(day)
    for ticker, move in shocks.items():
        actual = closes[ticker].iloc[position] / closes[ticker].iloc[position - 1] - 1
        factor = (1 + move) / (1 + actual)
        shocked.loc[day:, ticker] = closes.loc[day:, ticker] * factor
    return shocked


def decide_on(closes: pd.DataFrame, weights: dict, day, limits: Limits = Limits(), manager=rotate_manager,
              calm: list[str] | None = None, lookback: int = RISK_LOOKBACK) -> dict:
    """What the lines and the manager say on one day for a given portfolio. Used for what-if shocks."""
    day = pd.Timestamp(day)
    lines = lines_from_limits(limits)
    returns = closes.ffill().pct_change().loc[:day].iloc[-lookback:]
    allowed = list(closes.columns)
    held = pd.Series(weights, dtype=float).reindex(allowed).fillna(0.0)
    path = (1 + (returns * held).sum(axis=1)).cumprod()
    nav = pd.concat([pd.Series([1.0]), path.reset_index(drop=True)])
    drawdown = float(nav.iloc[-1] / nav.max() - 1)
    day_return = float(path.iloc[-1] / path.iloc[-2] - 1)
    var = one_day_var(held, returns)
    kind, reasons = evaluate(drawdown, day_return, var, lines)
    calm = [t for t in allowed if held[t] == 0] if calm is None else list(calm)
    context = build_context(day, held, returns, nav.reset_index(drop=True), lines, kind or "rotate", reasons, calm, held)
    plan = manager(context) if kind else None
    check = check_plan(plan, returns, lines, allowed) if plan else None
    return {"trigger": kind, "reasons": reasons, "today_return": day_return, "one_day_var": var,
            "plan": plan, "check": check, "context": context}
