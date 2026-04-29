from __future__ import annotations

import pandas as pd

from indicators.pine_primitives import sma, true_range


def atr_sma(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    tr = true_range(high=high, low=low, close=close)
    return sma(tr, length)
