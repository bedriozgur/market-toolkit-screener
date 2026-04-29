from __future__ import annotations

import numpy as np
import pandas as pd


def nz(series: pd.Series, replacement: float = 0.0) -> pd.Series:
    return series.fillna(replacement)


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length, min_periods=length).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    c1 = (high - low).abs()
    c2 = (high - prev_close).abs()
    c3 = (low - prev_close).abs()
    return pd.concat([c1, c2, c3], axis=1).max(axis=1).astype(float)


def rma(series: pd.Series, length: int) -> pd.Series:
    values = series.astype(float).to_numpy()
    out = np.full(len(values), np.nan, dtype=float)

    if length <= 0:
        raise ValueError('length must be positive')

    count = 0
    first_idx = None
    for i, value in enumerate(values):
        if not np.isnan(value):
            count += 1
        if count == length:
            first_idx = i
            break

    if first_idx is None:
        return pd.Series(out, index=series.index, dtype=float)

    window = values[first_idx - length + 1:first_idx + 1]
    prev = float(np.nanmean(window))
    out[first_idx] = prev

    alpha = 1.0 / float(length)
    for i in range(first_idx + 1, len(values)):
        value = values[i]
        if np.isnan(value):
            out[i] = prev
        else:
            prev = alpha * value + (1.0 - alpha) * prev
            out[i] = prev

    return pd.Series(out, index=series.index, dtype=float)


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return ((a > b) & (a.shift(1) <= b.shift(1))).fillna(False).astype(bool)


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return ((a < b) & (a.shift(1) >= b.shift(1))).fillna(False).astype(bool)


def barssince(condition: pd.Series) -> pd.Series:
    flags = condition.astype('boolean').fillna(False).astype(bool).tolist()
    out: list[float] = []
    last_true_idx: int | None = None
    for i, flag in enumerate(flags):
        if flag:
            last_true_idx = i
            out.append(0.0)
        else:
            out.append(np.nan if last_true_idx is None else float(i - last_true_idx))
    return pd.Series(out, index=condition.index, dtype='float64')
