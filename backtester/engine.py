"""The backtest engine: turn a strategy's positions into a portfolio value over time.

The whole engine is one function, run_backtest(). Read it top to bottom —
every step is commented.
"""

from dataclasses import dataclass

import pandas as pd

from .metrics import compute_metrics, format_metrics

TRADE_COLUMNS = ["entry_date", "exit_date", "direction", "entry_price", "exit_price", "return", "open"]


@dataclass
class BacktestResult:
    """Everything a backtest produces."""

    equity: pd.Series  # portfolio value each day (day 0 = start_cash)
    returns: pd.Series  # strategy's daily returns after costs
    positions: pd.Series  # position actually held each day (after the one-day shift)
    benchmark_equity: pd.Series  # value of simply buying and holding
    trades: pd.DataFrame  # one row per round-trip trade
    metrics: dict  # strategy metrics (raw numbers)
    benchmark_metrics: dict  # buy-and-hold metrics (raw numbers)

    def summary(self) -> pd.DataFrame:
        """Side-by-side table: strategy vs buy-and-hold, formatted for reading."""
        return pd.DataFrame(
            {"Strategy": format_metrics(self.metrics), "Buy & hold": format_metrics(self.benchmark_metrics)}
        )

    def plot(self):
        """Portfolio value and drawdown charts. Returns a matplotlib Figure."""
        from .plotting import plot_result  # imported here so the engine loads without matplotlib

        return plot_result(self)


def run_backtest(
    prices: pd.DataFrame,
    positions: pd.Series,
    cost_bps: float = 5.0,
    start_cash: float = 10_000.0,
    cash_rate: float = 0.0,
) -> BacktestResult:
    """Simulate trading `positions` on `prices`.

    prices:     DataFrame with a "Close" column, indexed by date (e.g. from load_prices).
    positions:  Series of 1 (long), 0 (cash) or -1 (short) — one value per day, decided
                using information available at that day's close. NaN counts as 0.
                A plain list or numpy array also works if it has one value per price row.
    cost_bps:   trading cost in basis points (1 bp = 0.01%), charged on every position change.
    start_cash: starting portfolio value.
    cash_rate:  yearly interest earned on days out of the market (0.04 = 4%). Default 0, as in the workshop.
                Sharpe is then measured above this rate.

    Simplifications (good exercises to remove!): no borrowing fees for shorts,
    no slippage, always 100% in or out, one asset at a time.
    """
    close = prices["Close"]
    if len(close) < 2:
        raise ValueError("Need at least 2 days of prices to run a backtest — check your date range.")

    # 1. Line positions up with the price dates. Missing or NaN days mean "no position".
    signal = _align_positions(positions, close)
    allowed = signal.isin([-1.0, 0.0, 1.0])
    if not allowed.all():
        bad = sorted(signal[~allowed].unique().tolist())[:5]  # plain floats, so the message reads "[2.0]"
        raise ValueError(f"Positions must be -1, 0 or 1 — found {bad}.")

    # 2. THE MOST IMPORTANT LINE IN THIS REPO.
    #    A signal is calculated from day t's closing price, so the earliest we can act on it
    #    is at that close — and the first return we can earn is day t+1's.
    #    shift(1) moves every position one day later. Delete it and the strategy
    #    "knows" each day's price move before it happens: look-ahead bias.
    held = signal.shift(1).fillna(0.0)

    # 3. The asset's daily % change: today's close vs yesterday's.
    asset_returns = (close / close.shift(1) - 1).fillna(0.0)

    # 4. Trading costs: pay cost_bps on the size of every position change.
    #    0 -> 1 costs 1x; 1 -> -1 costs 2x (you sell AND go short).
    costs = held.diff().abs().fillna(0.0) * cost_bps / 10_000

    # 5. Strategy return each day = position held x asset return - costs.
    daily = held * asset_returns - costs
    if cash_rate:
        daily = daily + (1 - held.abs()) * ((1 + cash_rate) ** (1 / 252) - 1)
        daily.iloc[0] = 0.0

    # 6. Grow the starting cash day by day. Day 0 has no return, so it stays at start_cash.
    equity = start_cash * (1 + daily).cumprod()
    benchmark_equity = start_cash * close / close.iloc[0]

    returns = daily.iloc[1:]
    benchmark_returns = asset_returns.iloc[1:]
    trades = _extract_trades(signal, close)

    return BacktestResult(
        equity=equity,
        returns=returns,
        positions=held,
        benchmark_equity=benchmark_equity,
        trades=trades,
        metrics=compute_metrics(equity, returns, trades, risk_free_rate=cash_rate),
        benchmark_metrics=compute_metrics(benchmark_equity, benchmark_returns, risk_free_rate=cash_rate),
    )


def _align_positions(positions, close: pd.Series) -> pd.Series:
    """Put the strategy's positions on the same dates as the prices, as floats.

    A Series is matched to the prices by date, so positions computed on the full history
    still work when the prices are a shorter slice. A list or numpy array is matched by row.
    """
    if isinstance(positions, pd.DataFrame):
        raise TypeError("positions must be a Series (one column), not a DataFrame — e.g. pass df['position'].")
    if not isinstance(positions, pd.Series):
        if len(positions) != len(close):
            raise ValueError(f"Got {len(positions)} positions for {len(close)} days of prices — they must match.")
        positions = pd.Series(positions, index=close.index)
    elif not positions.index.isin(close.index).any():
        raise ValueError(
            "The positions' dates don't match the price dates — build positions from the same "
            "prices DataFrame, e.g. my_strategy(prices), so they share its date index."
        )
    return positions.reindex(close.index).astype(float).fillna(0.0)


def _extract_trades(signal: pd.Series, close: pd.Series) -> pd.DataFrame:
    """List every round trip: when we got in, when we got out, and what it returned.

    Trades are priced at the close of the day the signal changed — the same moment
    run_backtest assumes we trade. Returns here are before costs.
    A trade still open on the last day is closed at the last close and marked open=True:
    the data ended, not the strategy.
    """
    rows = []
    current, entry_date = 0.0, None
    for date, position in signal.items():
        if position == current:
            continue
        if current != 0:
            rows.append(_trade_row(entry_date, date, current, close, still_open=False))
        current, entry_date = position, date

    last_date = signal.index[-1]
    if current != 0 and entry_date != last_date:
        rows.append(_trade_row(entry_date, last_date, current, close, still_open=True))
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def _trade_row(entry_date, exit_date, direction: float, close: pd.Series, still_open: bool) -> dict:
    entry_price, exit_price = close.loc[entry_date], close.loc[exit_date]
    return {
        "entry_date": entry_date,
        "exit_date": exit_date,
        "direction": "long" if direction > 0 else "short",
        "entry_price": entry_price,
        "exit_price": exit_price,
        "return": direction * (exit_price / entry_price - 1),
        "open": still_open,  # True = still held when the data ended
    }
