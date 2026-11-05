"""Load the workshop price data.

All prices live in one CSV (data/prices.csv) in "long" format: one row per
ticker per trading day. These helpers turn it into the shapes you want.
Prices are already adjusted for splits and dividends.
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "prices.csv"
# Used when the CSV isn't on disk — e.g. a notebook opened in Colab without the repo.
DATA_URL = "https://raw.githubusercontent.com/aurvyl/qm-fintech-quant-workshop/main/data/prices.csv"

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@lru_cache(maxsize=1)
def _load_all() -> pd.DataFrame:
    """Read the whole CSV once and keep it in memory."""
    source = DATA_PATH if DATA_PATH.exists() else DATA_URL
    return pd.read_csv(source, parse_dates=["Date"])


def available_tickers() -> list[str]:
    """All tickers in the dataset, sorted A-Z."""
    return sorted(_load_all()["Ticker"].unique())


def _to_timestamp(value):
    return None if value is None else pd.Timestamp(value)


def load_prices(ticker: str, start=None, end=None) -> pd.DataFrame:
    """Daily Open/High/Low/Close/Volume for one ticker, indexed by date.

    start / end: optional, inclusive. Strings like "2020-01-01" or date objects both work.
    """
    ticker = ticker.strip().upper()
    df = _load_all()
    if ticker not in set(df["Ticker"]):
        raise ValueError(f"Unknown ticker {ticker!r}. Available: {', '.join(available_tickers())}")

    prices = df[df["Ticker"] == ticker].set_index("Date")[PRICE_COLUMNS].sort_index()
    return prices.loc[_to_timestamp(start) : _to_timestamp(end)].copy()


def load_closes(tickers, start=None, end=None) -> pd.DataFrame:
    """Closing prices for several tickers side by side — one column per ticker."""
    if isinstance(tickers, str):
        tickers = [tickers]
    return pd.DataFrame({t.strip().upper(): load_prices(t, start, end)["Close"] for t in tickers})
