from __future__ import annotations

import pandas as pd


def compute_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 14) -> pd.Series:
    typical_price = (high + low + close) / 3.0
    money_flow = typical_price * volume
    direction = typical_price.diff()

    positive_flow = money_flow.where(direction > 0.0, 0.0)
    negative_flow = money_flow.where(direction < 0.0, 0.0)

    positive_sum = positive_flow.rolling(window=length, min_periods=length).sum()
    negative_sum = negative_flow.rolling(window=length, min_periods=length).sum()

    ratio = positive_sum / negative_sum
    mfi = 100.0 - (100.0 / (1.0 + ratio))

    mfi = mfi.where(negative_sum != 0, 100.0)
    both_zero = (positive_sum == 0) & (negative_sum == 0)
    mfi = mfi.where(~both_zero, 50.0)
    return mfi.astype(float)
