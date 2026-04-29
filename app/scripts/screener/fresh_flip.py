from __future__ import annotations

import pandas as pd


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def evaluate_fresh_flip(last_row: pd.Series, config: dict | None = None) -> dict:
    screener_cfg = (config or {}).get("fresh_flip", {})
    max_flip_age_bars = int(screener_cfg.get("max_flip_age_bars", 5))
    min_rel_volume_20 = float(screener_cfg.get("min_rel_volume_20", 0.8))
    min_abs_roc_5 = float(screener_cfg.get("min_abs_roc_5", 0.5))

    flip_age = last_row.get("flip_age_bars_ema20")
    current_state = int(last_row.get("current_state_ema20", 0))
    previous_state = last_row.get("previous_state_ema20")

    eligible = bool(
        pd.notna(flip_age)
        and float(flip_age) <= max_flip_age_bars
        and float(last_row.get("rel_volume_20", 0.0)) >= min_rel_volume_20
        and abs(float(last_row.get("roc_5", 0.0))) >= min_abs_roc_5
    )

    if not eligible:
        return {
            "screener_name": "fresh_flip",
            "eligible": False,
            "score": 0.0,
            "reason_1": "not eligible",
            "reason_2": "fresh flip conditions not met",
            "reason_3": "",
        }

    freshness_score = _clamp((max_flip_age_bars - float(flip_age) + 1.0) / (max_flip_age_bars + 1.0) * 40.0, 0.0, 40.0)
    volume_score = _clamp((float(last_row.get("rel_volume_20", 0.0)) - min_rel_volume_20) * 15.0, 0.0, 20.0)
    momentum_score = _clamp(abs(float(last_row.get("roc_5", 0.0))) * 6.0, 0.0, 20.0)

    trend_context_score = 0.0
    if current_state == 1:
        trend_context_score += 8.0 if float(last_row.get("close", 0.0)) > float(last_row.get("ema_50", 0.0)) else 2.0
        trend_context_score += 6.0 if float(last_row.get("roc_20", 0.0)) > 0 else 1.0
        trend_context_score += 6.0 if int(last_row.get("bullish_bar_flag", 0)) == 1 else 1.0
    else:
        trend_context_score += 8.0 if float(last_row.get("close", 0.0)) < float(last_row.get("ema_50", 0.0)) else 2.0
        trend_context_score += 6.0 if float(last_row.get("roc_20", 0.0)) < 0 else 1.0
        trend_context_score += 6.0 if int(last_row.get("bearish_bar_flag", 0)) == 1 else 1.0

    trend_context_score = _clamp(trend_context_score, 0.0, 20.0)
    score = round(_clamp(freshness_score + volume_score + momentum_score + trend_context_score, 0.0, 100.0), 2)

    return {
        "screener_name": "fresh_flip",
        "eligible": True,
        "score": score,
        "reason_1": "fresh bullish flip" if current_state == 1 else "fresh bearish flip",
        "reason_2": f"flip age {int(float(flip_age))} bars",
        "reason_3": "momentum confirmed",
        "flip_age_bars": float(flip_age),
        "previous_state": previous_state,
        "current_state": current_state,
        "freshness_score": round(freshness_score, 2),
        "volume_score": round(volume_score, 2),
        "momentum_score": round(momentum_score, 2),
        "trend_context_score": round(trend_context_score, 2),
    }
