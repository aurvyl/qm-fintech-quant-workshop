"""Alpha and beta: how much of a strategy's return is just the market, and how much is extra?

Take every month's return above cash, for your strategy and for the market (SPY).
Plot one dot per month and draw the best straight line through the dots:

    your excess return = alpha + beta x market excess return + noise

beta  is the slope: how much you move when the market moves 1%.
alpha is where the line crosses zero: what you earned in a month when the market did nothing.

The alpha competition scores every strategy the same way, on the same fixed data,
and ranks on a test period the strategy was not built on. See notebook 07.
"""

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import DATA_PATH, available_tickers, load_closes
from .portfolio import daily_cash_return, run_portfolio_backtest

MARKET = "SPY"
CASH_RATE = 0.03
COST_BPS = 5.0
DEV_PERIOD = ("2016-01-01", "2022-12-31")
TEST_PERIOD = ("2023-01-01", None)
MIN_MONTHS = 24
LOOKAHEAD_CUTS = 6


def monthly_returns(daily: pd.Series) -> pd.Series:
    """Compound daily returns into one return per calendar month."""
    daily = daily.dropna()
    return (1 + daily).groupby(daily.index.to_period("M")).prod() - 1


def monthly_cash_return(cash_rate: float = CASH_RATE) -> float:
    """What cash pays in one month at a yearly rate (0.03 = 3% a year)."""
    return (1 + cash_rate) ** (1 / 12) - 1


@dataclass
class AlphaBeta:
    """The fitted line through the monthly dots, plus how much to trust it."""

    alpha: float
    beta: float
    alpha_t: float
    r_squared: float
    months: int
    points: pd.DataFrame = field(repr=False)

    @property
    def alpha_annual(self) -> float:
        """Monthly alpha x 12: the simple yearly figure most people quote."""
        return self.alpha * 12

    def market_up(self, move: float = 0.05) -> float:
        """Your expected excess return in a month the market beats cash by `move`."""
        return self.alpha + self.beta * move

    def summary(self) -> dict:
        return {
            "Beta": f"{self.beta:.2f}",
            "Alpha per month": f"{self.alpha:+.2%}",
            "Alpha per year": f"{self.alpha_annual:+.1%}",
            "Alpha t-stat": f"{self.alpha_t:.2f}",
            "R squared": f"{self.r_squared:.2f}",
            "Months": f"{self.months}",
        }

    def plot(self, title: str = "Monthly returns above cash", ax=None):
        """Dots, your fitted line, and the market's own line (beta 1, alpha 0)."""
        from matplotlib.figure import Figure

        if ax is None:
            fig = Figure(figsize=(6, 6))
            ax = fig.subplots()
        x = self.points["market"] * 100
        y = self.points["strategy"] * 100
        edge = max(float(np.abs(x).max()), 1.0) * 1.1
        line = np.array([-edge, edge])
        ax.axhline(0, color="0.8", lw=1)
        ax.axvline(0, color="0.8", lw=1)
        ax.scatter(x, y, s=22, alpha=0.6, color="tab:blue", label="Monthly results")
        ax.plot(line, self.alpha * 100 + self.beta * line, color="tab:orange", lw=2,
                label=f"Your line: beta {self.beta:.2f}, alpha {self.alpha:+.2%}/month")
        ax.plot(line, line, color="0.4", ls="--", lw=1.5, label="The market itself (beta 1, alpha 0)")
        ax.scatter([0], [self.alpha * 100], s=70, color="tab:orange", zorder=3)
        ax.set_xlim(-edge, edge)
        ax.set_xlabel("Market return above cash (% per month)")
        ax.set_ylabel("Your return above cash (% per month)")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
        return ax.figure


def alpha_beta(strategy_daily: pd.Series, market_daily: pd.Series, cash_rate: float = CASH_RATE) -> AlphaBeta:
    """Fit the straight line through monthly excess returns by least squares.

    Both inputs are daily returns (0.01 = +1%). Cash is subtracted from both, month by month.
    alpha_t is alpha divided by its standard error: roughly, above 2 is unlikely to be pure luck.
    """
    cash = monthly_cash_return(cash_rate)
    months = pd.concat({"strategy": monthly_returns(strategy_daily), "market": monthly_returns(market_daily)}, axis=1)
    points = (months.dropna() - cash)
    n = len(points)
    if n < 3:
        raise ValueError(f"Need at least 3 months to fit a line, got {n}.")
    x, y = points["market"].to_numpy(), points["strategy"].to_numpy()
    x_mean, y_mean = x.mean(), y.mean()
    spread = ((x - x_mean) ** 2).sum()
    if spread < 1e-18:
        raise ValueError("The market never moved in these months, so beta cannot be measured.")
    beta = float(((x - x_mean) * (y - y_mean)).sum() / spread)
    alpha = float(y_mean - beta * x_mean)
    residuals = y - alpha - beta * x
    total = ((y - y_mean) ** 2).sum()
    r_squared = float(1 - (residuals**2).sum() / total) if total > 1e-18 else float("nan")
    noise = (residuals**2).sum() / (n - 2)
    alpha_se = math.sqrt(noise * (1 / n + x_mean**2 / spread))
    alpha_t = alpha / alpha_se if alpha_se > 1e-12 else float("nan")
    return AlphaBeta(alpha=alpha, beta=beta, alpha_t=alpha_t, r_squared=r_squared, months=n, points=points)


def data_fingerprint() -> str:
    """First 12 characters of the price file's SHA-256. Every score is tied to this exact data."""
    if not DATA_PATH.exists():
        return "downloaded"
    return hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()[:12]


def competition_closes() -> pd.DataFrame:
    """The fixed competition data: closing prices of every ticker in the workshop file."""
    return load_closes(available_tickers()).dropna()


def find_lookahead(strategy, closes: pd.DataFrame, cuts: int = LOOKAHEAD_CUTS) -> list[pd.Timestamp]:
    """Days on which the strategy's choice changes when the future is swapped for a different one.

    At each cut date every later daily move is flipped (up becomes down) while the calendar stays
    the same. An honest strategy picks the same weights up to the cut date either way.
    Different weights mean it peeked at later prices.
    """
    full = _weights(strategy, closes).reindex(index=closes.index, columns=closes.columns).fillna(0.0)
    first = max(260, len(closes) // 10)
    positions = np.linspace(first, len(closes) - 2, cuts).astype(int)
    peeked = []
    for i in positions:
        day = closes.index[i]
        other = _weights(strategy, _mirror_after(closes, i))
        other = other.reindex(index=closes.index, columns=closes.columns).fillna(0.0)
        if not np.allclose(full.loc[:day].to_numpy(), other.loc[:day].to_numpy(), atol=1e-9):
            peeked.append(day)
    return peeked


def _mirror_after(closes: pd.DataFrame, i: int) -> pd.DataFrame:
    moves = closes.pct_change().iloc[i + 1 :].clip(upper=0.5)
    future = closes.iloc[i] * (1 - moves).cumprod()
    return pd.concat([closes.iloc[: i + 1], future])


def _weights(strategy, closes: pd.DataFrame) -> pd.DataFrame:
    weights = strategy(closes.copy())
    if isinstance(weights, pd.Series):
        weights = weights.to_frame(MARKET)
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("A competition strategy returns a DataFrame of weights: one row per day, one column per ticker.")
    return weights.astype(float).fillna(0.0)


@dataclass
class Score:
    """One strategy's competition result."""

    name: str
    dev: AlphaBeta | None
    test: AlphaBeta | None
    lookahead_days: list
    problem: str | None
    result: object = field(repr=False, default=None)
    fingerprint: str = ""

    @property
    def qualified(self) -> bool:
        return self.problem is None

    def row(self) -> dict:
        def pct(fit, name):
            return getattr(fit, name) if fit is not None else float("nan")

        return {
            "Strategy": self.name,
            "Test alpha / year": pct(self.test, "alpha_annual"),
            "Test beta": pct(self.test, "beta"),
            "Test alpha t-stat": pct(self.test, "alpha_t"),
            "Build-period alpha / year": pct(self.dev, "alpha_annual"),
            "Status": "OK" if self.qualified else f"Out: {self.problem}",
        }

    def summary(self) -> pd.DataFrame:
        columns = {}
        if self.dev is not None:
            columns[f"Build period {DEV_PERIOD[0][:4]} to {DEV_PERIOD[1][:4]}"] = self.dev.summary()
        if self.test is not None:
            columns[f"Test period from {TEST_PERIOD[0][:4]} (scored)"] = self.test.summary()
        table = pd.DataFrame(columns)
        table.columns.name = self.name if self.qualified else f"{self.name} (out: {self.problem})"
        return table

    def plot(self):
        """The build period and the scored test period side by side."""
        from matplotlib.figure import Figure

        fig = Figure(figsize=(12, 6))
        left, right = fig.subplots(1, 2, sharey=True)
        if self.dev is not None:
            self.dev.plot("Build period", ax=left)
        if self.test is not None:
            self.test.plot("Test period (this is scored)", ax=right)
        fig.suptitle(self.name)
        fig.tight_layout()
        return fig


def score(strategy, name: str = "My strategy", closes: pd.DataFrame | None = None,
          check_lookahead: bool = True) -> Score:
    """Score a strategy exactly as the competition does.

    strategy: a function that takes `closes` (one column per ticker) and returns target weights,
              decided at each day's close: one row per day, one column per ticker, each row adding
              up to at most 1, no negatives. Whatever is left sits in cash and earns CASH_RATE.
              Returning a Series instead means "this share of the money in SPY".
    Rules: 5 bp cost on every trade, no leverage, no shorting, the one-day delay is applied for you,
           and the strategy must not use future prices. Rank = test-period alpha per year.
    """
    closes = competition_closes() if closes is None else closes
    fingerprint = data_fingerprint()
    try:
        weights = _weights(strategy, closes)
        result = run_portfolio_backtest(closes, weights, cost_bps=COST_BPS, cash_rate=CASH_RATE, benchmark=MARKET)
    except Exception as error:
        return Score(name, None, None, [], f"{type(error).__name__}: {error}", fingerprint=fingerprint)

    market = closes[MARKET].pct_change().iloc[1:]
    fits = {}
    for label, (start, end) in (("dev", DEV_PERIOD), ("test", TEST_PERIOD)):
        window = slice(pd.Timestamp(start), None if end is None else pd.Timestamp(end))
        try:
            fits[label] = alpha_beta(result.returns.loc[window], market.loc[window])
        except ValueError:
            fits[label] = None

    problem = None
    peeked = find_lookahead(strategy, closes) if check_lookahead else []
    if peeked:
        problem = f"uses future prices (seen on {peeked[0]:%Y-%m-%d})"
    elif fits["test"] is None or fits["test"].months < MIN_MONTHS:
        problem = f"fewer than {MIN_MONTHS} test months"
    elif result.turnover.sum() == 0:
        problem = "never invests"
    return Score(name, fits["dev"], fits["test"], peeked, problem, result=result, fingerprint=fingerprint)


def leaderboard(scores: list[Score]) -> pd.DataFrame:
    """Rank by test-period alpha per year. Ties go to the higher t-stat. Disqualified entries sit at the bottom."""
    table = pd.DataFrame([s.row() for s in scores])
    if table.empty:
        return table
    table["_out"] = table["Status"] != "OK"
    table = table.sort_values(["_out", "Test alpha / year", "Test alpha t-stat"],
                              ascending=[True, False, False], na_position="last").drop(columns="_out")
    table.index = pd.RangeIndex(1, len(table) + 1, name="Rank")
    return table


def format_leaderboard(table: pd.DataFrame) -> pd.DataFrame:
    """Readable copy of a leaderboard: percentages and two decimals."""
    shown = table.copy()
    for column in ("Test alpha / year", "Build-period alpha / year"):
        shown[column] = shown[column].map(lambda v: "n/a" if pd.isna(v) else f"{v:+.1%}")
    for column in ("Test beta", "Test alpha t-stat"):
        shown[column] = shown[column].map(lambda v: "n/a" if pd.isna(v) else f"{v:.2f}")
    return shown


__all__ = [
    "AlphaBeta", "CASH_RATE", "COST_BPS", "DEV_PERIOD", "MARKET", "MIN_MONTHS", "Score", "TEST_PERIOD",
    "alpha_beta", "competition_closes", "daily_cash_return", "data_fingerprint", "find_lookahead",
    "format_leaderboard", "leaderboard", "monthly_cash_return", "monthly_returns", "score",
]
