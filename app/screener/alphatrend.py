from __future__ import annotations

import pandas as pd


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def evaluate_alphatrend(last_row: pd.Series, config: dict) -> dict:
    cfg = config.get("alphatrend", {})
    max_signal_age_bars = int(cfg.get("screener_max_signal_age_bars", 5))
    require_active_buy = bool(cfg.get("screener_require_active_buy", True))

    last_valid_signal = last_row.get("alphatrend_last_valid_signal")
    current_state_label = last_row.get("alphatrend_compare_state_label")
    bars_since_valid_signal = last_row.get("alphatrend_bars_since_valid_signal")

    eligible = bool(
        pd.notna(bars_since_valid_signal)
        and float(bars_since_valid_signal) <= float(max_signal_age_bars)
        and last_valid_signal == "BUY"
        and (current_state_label == "BUY" if require_active_buy else True)
    )

    if not eligible:
        return {
            "screener_name": "alphatrend",
            "eligible": False,
            "score": 0.0,
            "reason_1": "not eligible",
            "reason_2": "alphatrend conditions not met",
            "reason_3": "",
        }

    freshness_score = _clamp(((max_signal_age_bars - float(bars_since_valid_signal) + 1.0) / (max_signal_age_bars + 1.0)) * 35.0, 0.0, 35.0)
    volume_score = _clamp((float(last_row.get("rel_volume_20", 0.0)) - 0.8) * 10.0, 0.0, 15.0)
    momentum_score = _clamp(max(float(last_row.get("roc_5", 0.0)), 0.0) + max(float(last_row.get("roc_20", 0.0)), 0.0), 0.0, 20.0)

    state_score = 0.0
    state_score += 8.0 if float(last_row.get("close", 0.0)) > float(last_row.get("ema_20", 0.0)) else 0.0
    state_score += 6.0 if float(last_row.get("close", 0.0)) > float(last_row.get("ema_50", 0.0)) else 0.0
    state_score += 6.0 if float(last_row.get("close", 0.0)) > float(last_row.get("ema_200", 0.0)) else 0.0

    move_score = _clamp(float(last_row.get("move_since_flip_pct", 0.0)) if pd.notna(last_row.get("move_since_flip_pct")) else 0.0, -20.0, 20.0)
    score = round(_clamp(freshness_score + volume_score + momentum_score + state_score + move_score, 0.0, 100.0), 2)

    return {
        "screener_name": "alphatrend",
        "eligible": True,
        "score": score,
        "reason_1": "validated alphatrend buy",
        "reason_2": f"signal age {int(bars_since_valid_signal)} bars",
        "reason_3": "buy state active",
        "alphatrend_current_state": current_state_label,
        "alphatrend_last_valid_signal": last_valid_signal,
        "alphatrend_signal_age_bars": float(bars_since_valid_signal),
        "freshness_score": round(freshness_score, 2),
        "volume_score": round(volume_score, 2),
        "momentum_score": round(momentum_score, 2),
        "state_score": round(state_score, 2),
        "move_score": round(move_score, 2),
    }
