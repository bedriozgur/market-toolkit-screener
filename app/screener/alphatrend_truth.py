from __future__ import annotations

import pandas as pd


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_alphatrend_truth_row(symbol: str, universe: str, interval: str, last_row: pd.Series, fresh_signal_max_bars: int) -> dict:
    bars_since_valid = last_row.get("alphatrend_bars_since_valid_signal")
    is_fresh_signal = bool(pd.notna(bars_since_valid) and float(bars_since_valid) <= float(fresh_signal_max_bars))

    freshness_score = 0.0
    if pd.notna(bars_since_valid):
        freshness_score = _clamp(((fresh_signal_max_bars - float(bars_since_valid) + 1.0) / (fresh_signal_max_bars + 1.0)) * 35.0, 0.0, 35.0)

    volume_score = _clamp((float(last_row.get("rel_volume_20", 0.0)) - 0.8) * 10.0, 0.0, 15.0)

    current_state_label = last_row.get("alphatrend_compare_state_label")
    if current_state_label == "BUY":
        momentum_core = max(float(last_row.get("roc_5", 0.0)), 0.0) + max(float(last_row.get("roc_20", 0.0)), 0.0)
        state_score = 0.0
        state_score += 8.0 if float(last_row.get("close", 0.0)) > float(last_row.get("ema_20", 0.0)) else 0.0
        state_score += 6.0 if float(last_row.get("close", 0.0)) > float(last_row.get("ema_50", 0.0)) else 0.0
        state_score += 6.0 if float(last_row.get("close", 0.0)) > float(last_row.get("ema_200", 0.0)) else 0.0
    elif current_state_label == "SELL":
        momentum_core = max(-float(last_row.get("roc_5", 0.0)), 0.0) + max(-float(last_row.get("roc_20", 0.0)), 0.0)
        state_score = 0.0
        state_score += 8.0 if float(last_row.get("close", 0.0)) < float(last_row.get("ema_20", 0.0)) else 0.0
        state_score += 6.0 if float(last_row.get("close", 0.0)) < float(last_row.get("ema_50", 0.0)) else 0.0
        state_score += 6.0 if float(last_row.get("close", 0.0)) < float(last_row.get("ema_200", 0.0)) else 0.0
    else:
        momentum_core = 0.0
        state_score = 0.0

    momentum_score = _clamp(momentum_core, 0.0, 20.0)
    move_score = _clamp(float(last_row.get("move_since_flip_pct", 0.0)) if pd.notna(last_row.get("move_since_flip_pct")) else 0.0, -20.0, 20.0)
    priority_score = round(_clamp(freshness_score + volume_score + momentum_score + state_score + move_score, 0.0, 100.0), 2)

    return {
        "symbol": symbol,
        "universe": universe,
        "interval": interval,
        "date": last_row.get("date"),
        "current_close": last_row.get("close"),
        "alphatrend_value": round(float(last_row.get("alphatrend")), 6) if pd.notna(last_row.get("alphatrend")) else None,
        "alphatrend_lag2": round(float(last_row.get("alphatrend_lag2")), 6) if pd.notna(last_row.get("alphatrend_lag2")) else None,
        "current_state_label": current_state_label,
        "raw_buy_signal": bool(last_row.get("alphatrend_buy_signal_raw", False)),
        "raw_sell_signal": bool(last_row.get("alphatrend_sell_signal_raw", False)),
        "validated_buy_label": bool(last_row.get("alphatrend_buy_label", False)),
        "validated_sell_label": bool(last_row.get("alphatrend_sell_label", False)),
        "last_valid_signal": last_row.get("alphatrend_last_valid_signal"),
        "last_valid_signal_date": last_row.get("alphatrend_last_valid_signal_date"),
        "bars_since_valid_signal": float(bars_since_valid) if pd.notna(bars_since_valid) else None,
        "flip_date": last_row.get("alphatrend_flip_date"),
        "flip_close": round(float(last_row.get("alphatrend_flip_close")), 6) if pd.notna(last_row.get("alphatrend_flip_close")) else None,
        "close_to_flip_pct": round(float(last_row.get("close_to_flip_pct")), 4) if pd.notna(last_row.get("close_to_flip_pct")) else None,
        "move_since_flip_pct": round(float(last_row.get("move_since_flip_pct")), 4) if pd.notna(last_row.get("move_since_flip_pct")) else None,
        "is_fresh_signal": is_fresh_signal,
        "rel_volume_20": round(float(last_row.get("rel_volume_20", 0.0)), 4) if pd.notna(last_row.get("rel_volume_20")) else None,
        "roc_5": round(float(last_row.get("roc_5", 0.0)), 4) if pd.notna(last_row.get("roc_5")) else None,
        "roc_20": round(float(last_row.get("roc_20", 0.0)), 4) if pd.notna(last_row.get("roc_20")) else None,
        "rsi_14": round(float(last_row.get("rsi_14", 0.0)), 4) if pd.notna(last_row.get("rsi_14")) else None,
        "mfi_14": round(float(last_row.get("mfi_14", 0.0)), 4) if pd.notna(last_row.get("mfi_14")) else None,
        "freshness_score": round(freshness_score, 2),
        "volume_score": round(volume_score, 2),
        "momentum_score": round(momentum_score, 2),
        "state_score": round(state_score, 2),
        "move_score": round(move_score, 2),
        "priority_score": priority_score,
    }
