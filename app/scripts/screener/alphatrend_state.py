from __future__ import annotations

import pandas as pd


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_alphatrend_state_row(
    symbol: str,
    universe: str,
    interval: str,
    last_row: pd.Series,
    fresh_flip_max_bars: int,
) -> dict:
    state_value = last_row.get("alphatrend_state")
    state_label = last_row.get("alphatrend_state_label")
    flip_age_bars = last_row.get("alphatrend_flip_age_bars")
    flip_date = last_row.get("alphatrend_flip_date")
    flip_close = last_row.get("alphatrend_flip_close")
    current_close = last_row.get("close")
    move_since_flip_pct = last_row.get("move_since_flip_pct")
    close_to_flip_pct = last_row.get("close_to_flip_pct")

    is_fresh_flip = bool(pd.notna(flip_age_bars) and float(flip_age_bars) <= float(fresh_flip_max_bars))

    freshness_score = 0.0
    if pd.notna(flip_age_bars):
        freshness_score = _clamp(
            ((fresh_flip_max_bars - float(flip_age_bars) + 1.0) / (fresh_flip_max_bars + 1.0)) * 35.0,
            0.0,
            35.0,
        )

    volume_score = _clamp((float(last_row.get("rel_volume_20", 0.0)) - 0.8) * 10.0, 0.0, 15.0)

    if state_label == "BUY":
        momentum_core = max(float(last_row.get("roc_5", 0.0)), 0.0) + max(float(last_row.get("roc_20", 0.0)), 0.0)
        trend_context = 0.0
        trend_context += 8.0 if float(last_row.get("close", 0.0)) > float(last_row.get("ema_20", 0.0)) else 0.0
        trend_context += 6.0 if float(last_row.get("close", 0.0)) > float(last_row.get("ema_50", 0.0)) else 0.0
        trend_context += 6.0 if float(last_row.get("close", 0.0)) > float(last_row.get("ema_200", 0.0)) else 0.0
    elif state_label == "SELL":
        momentum_core = max(-float(last_row.get("roc_5", 0.0)), 0.0) + max(-float(last_row.get("roc_20", 0.0)), 0.0)
        trend_context = 0.0
        trend_context += 8.0 if float(last_row.get("close", 0.0)) < float(last_row.get("ema_20", 0.0)) else 0.0
        trend_context += 6.0 if float(last_row.get("close", 0.0)) < float(last_row.get("ema_50", 0.0)) else 0.0
        trend_context += 6.0 if float(last_row.get("close", 0.0)) < float(last_row.get("ema_200", 0.0)) else 0.0
    else:
        momentum_core = 0.0
        trend_context = 0.0

    momentum_score = _clamp(momentum_core, 0.0, 20.0)
    state_score = _clamp(trend_context, 0.0, 20.0)
    move_score = _clamp(float(move_since_flip_pct) if pd.notna(move_since_flip_pct) else 0.0, -20.0, 20.0)

    priority_score = round(_clamp(freshness_score + volume_score + momentum_score + state_score + move_score, 0.0, 100.0), 2)

    return {
        "symbol": symbol,
        "universe": universe,
        "interval": interval,
        "date": last_row.get("date"),
        "current_close": current_close,
        "alphatrend_state": int(state_value) if pd.notna(state_value) else None,
        "alphatrend_state_label": state_label,
        "alphatrend_prev_state": int(last_row.get("alphatrend_prev_state")) if pd.notna(last_row.get("alphatrend_prev_state")) else None,
        "alphatrend_prev_state_label": last_row.get("alphatrend_prev_state_label"),
        "alphatrend_flip_date": flip_date,
        "alphatrend_flip_age_bars": float(flip_age_bars) if pd.notna(flip_age_bars) else None,
        "alphatrend_flip_close": flip_close,
        "close_to_flip_pct": round(float(close_to_flip_pct), 4) if pd.notna(close_to_flip_pct) else None,
        "move_since_flip_pct": round(float(move_since_flip_pct), 4) if pd.notna(move_since_flip_pct) else None,
        "is_fresh_flip": is_fresh_flip,
        "alphatrend_buy_flag": int(last_row.get("alphatrend_buy_flag", 0)),
        "alphatrend_sell_flag": int(last_row.get("alphatrend_sell_flag", 0)),
        "rel_volume_20": round(float(last_row.get("rel_volume_20", 0.0)), 4) if pd.notna(last_row.get("rel_volume_20")) else None,
        "roc_5": round(float(last_row.get("roc_5", 0.0)), 4) if pd.notna(last_row.get("roc_5")) else None,
        "roc_20": round(float(last_row.get("roc_20", 0.0)), 4) if pd.notna(last_row.get("roc_20")) else None,
        "rsi_14": round(float(last_row.get("rsi_14", 0.0)), 4) if pd.notna(last_row.get("rsi_14")) else None,
        "mfi_14": round(float(last_row.get("mfi_14", 0.0)), 4) if pd.notna(last_row.get("mfi_14")) else None,
        "ema_20": round(float(last_row.get("ema_20", 0.0)), 6) if pd.notna(last_row.get("ema_20")) else None,
        "ema_50": round(float(last_row.get("ema_50", 0.0)), 6) if pd.notna(last_row.get("ema_50")) else None,
        "ema_200": round(float(last_row.get("ema_200", 0.0)), 6) if pd.notna(last_row.get("ema_200")) else None,
        "freshness_score": round(freshness_score, 2),
        "volume_score": round(volume_score, 2),
        "momentum_score": round(momentum_score, 2),
        "state_score": round(state_score, 2),
        "move_score": round(move_score, 2),
        "priority_score": priority_score,
    }
