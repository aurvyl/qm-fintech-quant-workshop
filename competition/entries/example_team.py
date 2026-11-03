"""Example entry. Copy this file to competition/entries/<your_team>.py and change it.

Rules (the scorer enforces them):
  * strategy(closes) returns weights: one row per day, one column per ticker, 0 or more,
    each row adding up to at most 1. Whatever is left is cash.
  * Decide each day using prices up to that day only. Peeking at later prices disqualifies you.
  * No leverage, no shorting, 5 bp cost on every trade. Rank = alpha per year in the test period.
"""

import pandas as pd

TEAM = "Example team"


def strategy(closes: pd.DataFrame) -> pd.DataFrame:
    defensive = ["XLP", "XLU", "XLV"]
    calm = closes["SPY"].pct_change().rolling(20).std() < closes["SPY"].pct_change().rolling(250).std()
    weights = pd.DataFrame(0.0, index=closes.index, columns=["SPY", *defensive])
    weights.loc[calm, "SPY"] = 1.0
    weights.loc[~calm, defensive] = 1 / len(defensive)
    return weights
