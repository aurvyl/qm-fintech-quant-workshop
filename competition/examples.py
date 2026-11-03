"""House strategies for the alpha competition leaderboard.

Each one takes `closes` (one column per ticker) and returns target weights:
one row per day, one column per ticker, each row adding up to at most 1.
"""

import pandas as pd

from backtester.portfolio import month_ends, top_n_weights

SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]


def hold_market(closes: pd.DataFrame) -> pd.DataFrame:
    """All in SPY, all the time. Beta 1 and alpha close to 0 (minus a small trading cost)."""
    return pd.DataFrame({"SPY": 1.0}, index=closes.index)


def hold_nasdaq(closes: pd.DataFrame) -> pd.DataFrame:
    """All in QQQ. A higher beta than the market: does it also have alpha?"""
    return pd.DataFrame({"QQQ": 1.0}, index=closes.index)


def half_market(closes: pd.DataFrame) -> pd.DataFrame:
    """Start half SPY, half cash, never rebalance. A lower beta, alpha near 0: less risk is not skill."""
    return pd.DataFrame({"SPY": 0.5}, index=closes.index)


def sector_momentum(closes: pd.DataFrame, lookback: int = 126, top: int = 3) -> pd.DataFrame:
    """Each month, hold the three sectors that rose most over the last six months."""
    momentum = closes[SECTORS] / closes[SECTORS].shift(lookback) - 1
    return top_n_weights(momentum, top, on=month_ends(closes.index))


def trend_or_bonds(closes: pd.DataFrame, slow: int = 200) -> pd.DataFrame:
    """Hold SPY while it is above its 200-day average, otherwise hold long bonds (TLT)."""
    above = closes["SPY"] > closes["SPY"].rolling(slow).mean()
    return pd.DataFrame({"SPY": above.astype(float), "TLT": (~above).astype(float)}, index=closes.index)


def peek_ahead(closes: pd.DataFrame) -> pd.DataFrame:
    """A cheat: holds SPY only on days before it goes up. The look-ahead check catches it."""
    tomorrow_up = closes["SPY"].shift(-1) > closes["SPY"]
    return pd.DataFrame({"SPY": tomorrow_up.astype(float)}, index=closes.index)


HOUSE = {
    "House: hold the market (SPY)": hold_market,
    "House: hold the Nasdaq (QQQ)": hold_nasdaq,
    "House: half market, half cash": half_market,
    "House: sector momentum, top 3": sector_momentum,
    "House: trend or bonds": trend_or_bonds,
    "House: peeks at tomorrow (cheat)": peek_ahead,
}
