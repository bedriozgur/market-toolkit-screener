from __future__ import annotations

import numpy as np
import pandas as pd


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.astype("float64")


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean().astype("float64")


def _pine_rsi(series: pd.Series, length: int) -> pd.Series:
    close = series.astype("float64")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = pd.Series(np.nan, index=close.index, dtype="float64")
    avg_loss = pd.Series(np.nan, index=close.index, dtype="float64")

    if len(close) < length:
        return pd.Series(np.nan, index=close.index, dtype="float64")

    gain_seed = gain.iloc[1:length + 1] if len(gain) >= length + 1 else gain.iloc[:length]
    loss_seed = loss.iloc[1:length + 1] if len(loss) >= length + 1 else loss.iloc[:length]

    seed_pos = length
    if seed_pos >= len(close):
        return pd.Series(np.nan, index=close.index, dtype="float64")

    avg_gain.iloc[seed_pos] = gain_seed.mean()
    avg_loss.iloc[seed_pos] = loss_seed.mean()

    for i in range(seed_pos + 1, len(close)):
        avg_gain.iloc[i] = ((avg_gain.iloc[i - 1] * (length - 1)) + gain.iloc[i]) / length
        avg_loss.iloc[i] = ((avg_loss.iloc[i - 1] * (length - 1)) + loss.iloc[i]) / length

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    both_zero = (avg_gain == 0.0) & (avg_loss == 0.0)
    rsi = rsi.where(~both_zero, 50.0)
    return rsi.astype("float64")


def _pine_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    high = high.astype("float64")
    low = low.astype("float64")
    close = close.astype("float64")
    volume = volume.astype("float64")

    hlc3 = (high + low + close) / 3.0
    raw_money_flow = hlc3 * volume

    positive_flow = pd.Series(0.0, index=close.index, dtype="float64")
    negative_flow = pd.Series(0.0, index=close.index, dtype="float64")

    for i in range(1, len(close)):
        if hlc3.iloc[i] > hlc3.iloc[i - 1]:
            positive_flow.iloc[i] = raw_money_flow.iloc[i]
        elif hlc3.iloc[i] < hlc3.iloc[i - 1]:
            negative_flow.iloc[i] = raw_money_flow.iloc[i]

    pos_sum = positive_flow.rolling(length, min_periods=length).sum()
    neg_sum = negative_flow.rolling(length, min_periods=length).sum()

    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    mfi = 100.0 - (100.0 / (1.0 + ratio))
    mfi = mfi.where(neg_sum != 0.0, 100.0)
    both_zero = (pos_sum == 0.0) & (neg_sum == 0.0)
    mfi = mfi.where(~both_zero, 50.0)
    return mfi.astype("float64")


def _pine_crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return ((a > b) & (a.shift(1) <= b.shift(1))).fillna(False).astype(bool)


def _pine_crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return ((a < b) & (a.shift(1) >= b.shift(1))).fillna(False).astype(bool)


def _barssince(condition: pd.Series) -> pd.Series:
    condition = condition.fillna(False).astype(bool)
    out = np.full(len(condition), np.nan, dtype="float64")
    last_true_idx: int | None = None
    for i, flag in enumerate(condition.to_numpy()):
        if flag:
            last_true_idx = i
            out[i] = 0.0
        elif last_true_idx is not None:
            out[i] = float(i - last_true_idx)
    return pd.Series(out, index=condition.index, dtype="float64")


def _forward_fill_when_event(source: pd.Series, event_flag: pd.Series) -> pd.Series:
    mask = event_flag.fillna(False).astype(bool)
    return source.where(mask).ffill()


def compute_alphatrend(
    df: pd.DataFrame,
    *,
    coeff: float = 1.0,
    period: int = 14,
    no_volume_data: bool = False,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    date_col: str = "date",
) -> pd.DataFrame:
    required = [high_col, low_col, close_col]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    out = df.copy()
    high = out[high_col].astype("float64")
    low = out[low_col].astype("float64")
    close = out[close_col].astype("float64")
    volume = out[volume_col].astype("float64") if volume_col in out.columns else pd.Series(0.0, index=out.index, dtype="float64")

    has_volume = bool(volume.notna().any() and float(volume.fillna(0.0).sum()) > 0.0)
    use_rsi_branch = bool(no_volume_data or not has_volume)

    tr = _true_range(high, low, close)
    atr_sma = _sma(tr, period)
    rsi_value = _pine_rsi(close, period)
    mfi_value = _pine_mfi(high, low, close, volume, period)
    momentum_series = rsi_value if use_rsi_branch else mfi_value

    up_t = low - atr_sma * coeff
    down_t = high + atr_sma * coeff
    at = np.full(len(out), np.nan, dtype="float64")

    for i in range(len(out)):
        if pd.isna(atr_sma.iloc[i]):
            continue
        prev_at = 0.0 if i == 0 or pd.isna(at[i - 1]) else float(at[i - 1])
        momentum_ok = bool(momentum_series.iloc[i] >= 50.0) if pd.notna(momentum_series.iloc[i]) else False
        if momentum_ok:
            at[i] = prev_at if up_t.iloc[i] < prev_at else up_t.iloc[i]
        else:
            at[i] = prev_at if down_t.iloc[i] > prev_at else down_t.iloc[i]

    at_series = pd.Series(at, index=out.index, dtype="float64")
    at_lag2 = at_series.shift(2)

    buy_raw = _pine_crossover(at_series, at_lag2)
    sell_raw = _pine_crossunder(at_series, at_lag2)

    k1 = _barssince(buy_raw)
    k2 = _barssince(sell_raw)
    o1 = _barssince(buy_raw.shift(1, fill_value=False))
    o2 = _barssince(sell_raw.shift(1, fill_value=False))

    buy_valid = (buy_raw & (o1 > k2)).fillna(False).astype(bool)
    sell_valid = (sell_raw & (o2 > k1)).fillna(False).astype(bool)

    state_series = pd.Series(
        np.where(at_series > at_lag2, 1.0, np.where(at_series < at_lag2, -1.0, np.nan)),
        index=out.index,
        dtype="float64",
    ).ffill()

    valid_signal_label = pd.Series(pd.NA, index=out.index, dtype="object")
    valid_signal_label.loc[buy_valid] = "BUY"
    valid_signal_label.loc[sell_valid] = "SELL"
    last_valid_signal = valid_signal_label.ffill()

    direction = pd.Series(np.nan, index=out.index, dtype="float64")
    prev_direction = 0.0
    for i in range(len(out)):
        if buy_valid.iloc[i]:
            prev_direction = 1.0
        elif sell_valid.iloc[i]:
            prev_direction = -1.0
        direction.iloc[i] = prev_direction if prev_direction != 0.0 else np.nan

    direction_prev = direction.shift(1)
    direction_change = direction.ne(direction_prev) & direction.notna() & direction_prev.notna()
    bars_since_direction_change = _barssince(direction_change)
    direction_label = direction.map({1.0: "BUY", -1.0: "SELL"})

    bars_since_flip = _barssince(buy_valid | sell_valid)
    flip_date = _forward_fill_when_event(out[date_col], buy_valid | sell_valid) if date_col in out.columns else pd.Series(pd.NaT, index=out.index)
    flip_close = _forward_fill_when_event(close, buy_valid | sell_valid)

    close_to_flip_pct = ((close / flip_close) - 1.0) * 100.0
    move_since_flip = pd.Series(np.nan, index=out.index, dtype="float64")
    buy_mask = last_valid_signal == "BUY"
    sell_mask = last_valid_signal == "SELL"
    move_since_flip.loc[buy_mask] = ((close.loc[buy_mask] / flip_close.loc[buy_mask]) - 1.0) * 100.0
    move_since_flip.loc[sell_mask] = ((flip_close.loc[sell_mask] / close.loc[sell_mask]) - 1.0) * 100.0

    prev_state = state_series.shift(1)
    prev_state_label = prev_state.map({1.0: "BUY", -1.0: "SELL"})
    state_label = state_series.map({1.0: "BUY", -1.0: "SELL"})

    out["AT"] = at_series
    out["AT_lag2"] = at_lag2
    out["AT_tr"] = tr
    out["AT_atr_sma"] = atr_sma
    out["AT_rsi"] = rsi_value
    out["AT_mfi"] = mfi_value
    out["AT_upT"] = up_t
    out["AT_downT"] = down_t
    out["AT_k1"] = k1
    out["AT_k2"] = k2
    out["AT_o1"] = o1
    out["AT_o2"] = o2
    out["buy_raw"] = buy_raw
    out["sell_raw"] = sell_raw
    out["buy_valid"] = buy_valid
    out["sell_valid"] = sell_valid
    out["state"] = state_series
    out["trend"] = last_valid_signal
    out["bars_since_flip"] = bars_since_flip

    out["alphatrend"] = at_series
    out["alphatrend_lag2"] = at_lag2
    out["alphatrend_atr_sma"] = atr_sma
    out["alphatrend_buy_signal_raw"] = buy_raw
    out["alphatrend_sell_signal_raw"] = sell_raw
    out["alphatrend_buy_label"] = buy_valid
    out["alphatrend_sell_label"] = sell_valid
    out["alphatrend_valid_signal_flag"] = (buy_valid | sell_valid).astype(bool)
    out["alphatrend_valid_signal_label"] = valid_signal_label
    out["alphatrend_last_valid_signal"] = last_valid_signal
    out["alphatrend_bars_since_valid_signal"] = bars_since_flip
    out["alphatrend_compare_state"] = state_series
    out["alphatrend_compare_state_label"] = state_label
    out["alphatrend_state"] = state_series
    out["alphatrend_state_label"] = state_label
    out["alphatrend_prev_state"] = prev_state
    out["alphatrend_prev_state_label"] = prev_state_label
    out["alphatrend_flip_age_bars"] = bars_since_flip
    out["alphatrend_flip_date"] = flip_date
    out["alphatrend_flip_close"] = flip_close
    out["alphatrend_last_valid_signal_date"] = flip_date
    out["alphatrend_last_valid_signal_close"] = flip_close
    out["alphatrend_buy_flag"] = buy_valid.astype(int)
    out["alphatrend_sell_flag"] = sell_valid.astype(int)
    out["alphatrend_direction"] = direction
    out["alphatrend_direction_label"] = direction_label
    out["alphatrend_bars_since_direction_change"] = bars_since_direction_change
    out["close_to_flip_pct"] = close_to_flip_pct
    out["move_since_flip_pct"] = move_since_flip

    return out


compute = compute_alphatrend
