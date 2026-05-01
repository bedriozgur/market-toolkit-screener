from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.trend.alphatrend import compute_alphatrend


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def roc(series: pd.Series, length: int) -> pd.Series:
    return (series / series.shift(length) - 1.0) * 100.0


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def atr_rma(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 14) -> pd.Series:
    typical_price = (high + low + close) / 3.0
    money_flow = typical_price * volume
    direction = typical_price.diff()
    positive_flow = money_flow.where(direction > 0, 0.0)
    negative_flow = money_flow.where(direction < 0, 0.0).abs()
    positive_sum = positive_flow.rolling(length, min_periods=length).sum()
    negative_sum = negative_flow.rolling(length, min_periods=length).sum()
    ratio = positive_sum / negative_sum.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + ratio))


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index, dtype="float64")
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index, dtype="float64")
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

    atr = rma(tr, length)
    plus_di = 100.0 * rma(plus_dm, length) / atr.replace(0, np.nan)
    minus_di = 100.0 * rma(minus_dm, length) / atr.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_value = rma(dx, length)
    return adx_value, plus_di, minus_di


def aroon(high: pd.Series, low: pd.Series, length: int = 25) -> tuple[pd.Series, pd.Series, pd.Series]:
    aroon_up = pd.Series(np.nan, index=high.index, dtype="float64")
    aroon_down = pd.Series(np.nan, index=high.index, dtype="float64")
    for i in range(length - 1, len(high)):
        window_high = high.iloc[i - length + 1 : i + 1]
        window_low = low.iloc[i - length + 1 : i + 1]
        days_since_high = length - 1 - int(window_high.values.argmax())
        days_since_low = length - 1 - int(window_low.values.argmin())
        aroon_up.iloc[i] = ((length - days_since_high) / length) * 100.0
        aroon_down.iloc[i] = ((length - days_since_low) / length) * 100.0
    aroon_osc = aroon_up - aroon_down
    return aroon_up, aroon_down, aroon_osc


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 10, multiplier: float = 3.0) -> tuple[pd.Series, pd.Series]:
    atr = atr_rma(high, low, close, length)
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = pd.Series(np.nan, index=close.index, dtype="float64")
    final_lower = pd.Series(np.nan, index=close.index, dtype="float64")
    trend = pd.Series(np.nan, index=close.index, dtype="float64")

    for i in range(len(close)):
        if pd.isna(atr.iloc[i]):
            continue
        if i == 0 or pd.isna(final_upper.iloc[i - 1]):
            final_upper.iloc[i] = basic_upper.iloc[i]
            final_lower.iloc[i] = basic_lower.iloc[i]
            trend.iloc[i] = 1.0 if close.iloc[i] >= basic_lower.iloc[i] else -1.0
            continue

        final_upper.iloc[i] = basic_upper.iloc[i] if (basic_upper.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]) else final_upper.iloc[i - 1]
        final_lower.iloc[i] = basic_lower.iloc[i] if (basic_lower.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]) else final_lower.iloc[i - 1]

        if trend.iloc[i - 1] == -1.0 and close.iloc[i] > final_upper.iloc[i - 1]:
            trend.iloc[i] = 1.0
        elif trend.iloc[i - 1] == 1.0 and close.iloc[i] < final_lower.iloc[i - 1]:
            trend.iloc[i] = -1.0
        else:
            trend.iloc[i] = trend.iloc[i - 1]
            if trend.iloc[i] == 1.0 and final_lower.iloc[i] < final_lower.iloc[i - 1]:
                final_lower.iloc[i] = final_lower.iloc[i - 1]
            if trend.iloc[i] == -1.0 and final_upper.iloc[i] > final_upper.iloc[i - 1]:
                final_upper.iloc[i] = final_upper.iloc[i - 1]

    st_line = pd.Series(np.where(trend == 1.0, final_lower, final_upper), index=close.index, dtype="float64")
    return st_line, trend


def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series) -> dict[str, pd.Series]:
    tenkan = (high.rolling(9, min_periods=9).max() + low.rolling(9, min_periods=9).min()) / 2.0
    kijun = (high.rolling(26, min_periods=26).max() + low.rolling(26, min_periods=26).min()) / 2.0
    senkou_a = ((tenkan + kijun) / 2.0).shift(26)
    senkou_b = ((high.rolling(52, min_periods=52).max() + low.rolling(52, min_periods=52).min()) / 2.0).shift(26)
    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
        "cloud_top": cloud_top,
        "cloud_bottom": cloud_bottom,
        "price_above_cloud": close > cloud_top,
        "kumo_green": senkou_a > senkou_b,
        "tenkan_above_kijun": tenkan > kijun,
        "chikou_above_price_26": close > close.shift(26),
    }


def _bars_since_event(event_series: pd.Series) -> pd.Series:
    flags = event_series.astype("boolean").fillna(False).astype(bool).tolist()
    out: list[float] = []
    last_event_idx: int | None = None
    for i, flag in enumerate(flags):
        if flag:
            last_event_idx = i
            out.append(0.0)
        else:
            out.append(np.nan if last_event_idx is None else float(i - last_event_idx))
    return pd.Series(out, index=event_series.index, dtype="float64")


def build_features(
    df: pd.DataFrame,
    alphatrend_multiplier: float = 1.0,
    alphatrend_common_period: int = 14,
    alphatrend_no_volume_data: bool = False,
    benchmark_close: pd.Series | None = None,
) -> pd.DataFrame:
    """Build the feature frame consumed by the screeners from normalized OHLCV input."""
    out = df.copy()

    out["ema_10"] = ema(out["close"], 10)
    out["ema_20"] = ema(out["close"], 20)
    out["ema_50"] = ema(out["close"], 50)
    out["ema_100"] = ema(out["close"], 100)
    out["ema_200"] = ema(out["close"], 200)
    out["ema_20_slope_5"] = (out["ema_20"] / out["ema_20"].shift(5) - 1.0) * 100.0

    out["sma_20"] = sma(out["close"], 20)
    out["sma_50"] = sma(out["close"], 50)
    out["sma_200"] = sma(out["close"], 200)

    out["roc_5"] = roc(out["close"], 5)
    out["roc_10"] = roc(out["close"], 10)
    out["roc_20"] = roc(out["close"], 20)
    out["roc_63"] = roc(out["close"], 63)
    out["rsi_14"] = rsi(out["close"], 14)

    out["macd_line"] = ema(out["close"], 12) - ema(out["close"], 26)
    out["macd_signal"] = ema(out["macd_line"], 9)
    out["macd_hist"] = out["macd_line"] - out["macd_signal"]
    out["macd_hist_delta"] = out["macd_hist"] - out["macd_hist"].shift(1)

    out["atr_14"] = atr_rma(out["high"], out["low"], out["close"], 14)
    out["atr_pct_14"] = (out["atr_14"] / out["close"]) * 100.0
    out["mfi_14"] = mfi(out["high"], out["low"], out["close"], out["volume"], 14)
    out["obv"] = (np.sign(out["close"].diff()).fillna(0.0) * out["volume"]).cumsum()
    out["obv_sma_20"] = out["obv"].rolling(20, min_periods=20).mean()
    out["obv_trend_20"] = (out["obv"] / out["obv_sma_20"] - 1.0) * 100.0
    out["adx_14"], out["plus_di_14"], out["minus_di_14"] = adx(out["high"], out["low"], out["close"], 14)
    out["aroon_up_25"], out["aroon_down_25"], out["aroon_osc_25"] = aroon(out["high"], out["low"], 25)
    st_line, st_trend = supertrend(out["high"], out["low"], out["close"], length=10, multiplier=3.0)
    out["supertrend_line"] = st_line
    out["supertrend_trend"] = st_trend
    out["supertrend_bullish"] = out["supertrend_trend"] == 1.0
    out["supertrend_prev_trend"] = out["supertrend_trend"].shift(1)
    out["supertrend_flip_flag"] = out["supertrend_prev_trend"].notna() & (out["supertrend_trend"] != out["supertrend_prev_trend"])
    out["supertrend_flip_age_bars"] = _bars_since_event(out["supertrend_flip_flag"])
    out["supertrend_state_label"] = np.where(out["supertrend_trend"] == 1.0, "BUY", "SELL")

    ichi = ichimoku(out["high"], out["low"], out["close"])
    for key, value in ichi.items():
        out[f"ichimoku_{key}"] = value

    bb_mid_20 = sma(out["close"], 20)
    bb_std_20 = out["close"].rolling(20, min_periods=20).std(ddof=0)
    out["bb_mid_20"] = bb_mid_20
    out["bb_upper_20"] = bb_mid_20 + 2.0 * bb_std_20
    out["bb_lower_20"] = bb_mid_20 - 2.0 * bb_std_20
    bb_width = (out["bb_upper_20"] - out["bb_lower_20"]).replace(0, np.nan)
    out["bb_percent_b"] = (out["close"] - out["bb_lower_20"]) / bb_width

    out["range_pct"] = ((out["high"] - out["low"]) / out["close"]) * 100.0
    out["rolling_high_20"] = out["high"].rolling(20, min_periods=20).max()
    out["rolling_high_50"] = out["high"].rolling(50, min_periods=50).max()
    out["rolling_high_252"] = out["high"].rolling(252, min_periods=252).max()
    out["rolling_low_20"] = out["low"].rolling(20, min_periods=20).min()
    out["rolling_low_50"] = out["low"].rolling(50, min_periods=50).min()
    out["rolling_low_252"] = out["low"].rolling(252, min_periods=252).min()
    out["rolling_range_20"] = ((out["rolling_high_20"] - out["rolling_low_20"]) / out["close"]) * 100.0
    out["rolling_range_20_sma_60"] = out["rolling_range_20"].rolling(60, min_periods=20).mean()
    out["dist_to_20d_high_pct"] = ((out["rolling_high_20"] - out["close"]) / out["close"]) * 100.0
    out["dist_to_50d_high_pct"] = ((out["rolling_high_50"] - out["close"]) / out["close"]) * 100.0
    out["dist_to_52w_high_pct"] = ((out["rolling_high_252"] - out["close"]) / out["close"]) * 100.0
    out["dist_to_52w_low_pct"] = ((out["close"] - out["rolling_low_252"]) / out["close"]) * 100.0

    denom = (out["rolling_high_20"] - out["rolling_low_20"]).replace(0, np.nan)
    out["close_position_in_20d_range"] = (out["close"] - out["rolling_low_20"]) / denom

    out["vol_sma_20"] = out["volume"].rolling(20, min_periods=20).mean()
    out["vol_sma_50"] = out["volume"].rolling(50, min_periods=50).mean()
    out["rel_volume_20"] = out["volume"] / out["vol_sma_20"]
    out["volume_ratio_10_50"] = out["volume"].rolling(10, min_periods=10).mean() / out["vol_sma_50"]
    out["dollar_volume"] = out["close"] * out["volume"]
    out["avg_volume_20"] = out["volume"].rolling(20, min_periods=20).mean()
    out["avg_dollar_volume_20"] = out["dollar_volume"].rolling(20, min_periods=20).mean()

    out["body_pct"] = (out["close"] - out["open"]).abs() / out["close"] * 100.0
    out["upper_wick_pct"] = (out["high"] - out[["open", "close"]].max(axis=1)) / out["close"] * 100.0
    out["lower_wick_pct"] = (out[["open", "close"]].min(axis=1) - out["low"]) / out["close"] * 100.0
    out["bullish_bar_flag"] = (out["close"] > out["open"]).astype(int)
    out["bearish_bar_flag"] = (out["close"] < out["open"]).astype(int)

    out["close_above_ema20_flag"] = (out["close"] > out["ema_20"]).astype(int)
    out["close_above_ema20_count_20"] = out["close_above_ema20_flag"].rolling(20, min_periods=20).sum()
    out["avg_rel_volume_3"] = out["rel_volume_20"].rolling(3, min_periods=3).mean()
    out["history_bars"] = range(1, len(out) + 1)

    if benchmark_close is not None:
        benchmark_close = benchmark_close.sort_index()
        benchmark_aligned = benchmark_close.reindex(pd.Index(out["date"])).ffill().bfill()
        out["benchmark_close"] = benchmark_aligned.to_numpy()
        out["rs_line"] = (out["close"] / out["benchmark_close"]) * 100.0
        out["rs_line_slope_5"] = (out["rs_line"] / out["rs_line"].shift(5) - 1.0) * 100.0
        out["rs_line_slope_20"] = (out["rs_line"] / out["rs_line"].shift(20) - 1.0) * 100.0
        out["rs_line_52w_high"] = out["rs_line"].rolling(252, min_periods=252).max()
        out["rs_line_dist_to_52w_high_pct"] = ((out["rs_line_52w_high"] - out["rs_line"]) / out["rs_line"]) * 100.0
        out["benchmark_roc_63"] = roc(out["benchmark_close"], 63)
        out["beta_60"] = out["close"].pct_change().rolling(60, min_periods=60).cov(out["benchmark_close"].pct_change()) / out["benchmark_close"].pct_change().rolling(60, min_periods=60).var()
        out["beta_adjusted_return_63"] = out["roc_63"] / out["beta_60"].replace(0, np.nan)
        out["excess_return_63"] = out["roc_63"] - out["benchmark_roc_63"]
    else:
        out["benchmark_close"] = np.nan
        out["rs_line"] = np.nan
        out["rs_line_slope_5"] = np.nan
        out["rs_line_slope_20"] = np.nan
        out["rs_line_52w_high"] = np.nan
        out["rs_line_dist_to_52w_high_pct"] = np.nan
        out["benchmark_roc_63"] = np.nan
        out["beta_60"] = np.nan
        out["beta_adjusted_return_63"] = np.nan
        out["excess_return_63"] = np.nan

    out["trend_state_ema20"] = np.where(out["close"] > out["ema_20"], 1, -1)
    out["trend_state_prev_ema20"] = out["trend_state_ema20"].shift(1)
    out["flip_flag_ema20"] = out["trend_state_prev_ema20"].notna() & (out["trend_state_ema20"] != out["trend_state_prev_ema20"])
    out["flip_age_bars_ema20"] = _bars_since_event(out["flip_flag_ema20"])
    out["previous_state_ema20"] = out["trend_state_prev_ema20"]
    out["current_state_ema20"] = out["trend_state_ema20"]

    out = compute_alphatrend(
        out,
        coeff=alphatrend_multiplier,
        period=alphatrend_common_period,
        no_volume_data=alphatrend_no_volume_data,
        open_col="open",
        high_col="high",
        low_col="low",
        close_col="close",
        volume_col="volume",
        date_col="date",
    )

    return out
