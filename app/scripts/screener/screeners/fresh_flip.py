from __future__ import annotations

import pandas as pd

from scripts.screener.models import ScreenerResult


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def evaluate(last_row: pd.Series, config: dict) -> ScreenerResult:
    screener_cfg = config.get("fresh_flip", {})
    max_flip_age_bars = int(screener_cfg.get("max_flip_age_bars", 5))
    min_rel_volume_20 = float(screener_cfg.get("min_rel_volume_20", 0.8))
    min_abs_roc_5 = float(screener_cfg.get("min_abs_roc_5", 0.5))

    flip_age = last_row.get("flip_age_bars_ema20")
    current_state = last_row.get("current_state_ema20")
    previous_state = last_row.get("previous_state_ema20")

    eligible = bool(
        pd.notna(flip_age)
        and flip_age <= max_flip_age_bars
        and last_row["rel_volume_20"] >= min_rel_volume_20
        and abs(last_row["roc_5"]) >= min_abs_roc_5
    )

    if not eligible:
        return ScreenerResult(
            screener_name="fresh_flip",
            eligible=False,
            score=0.0,
            reason_1="not eligible",
            reason_2="fresh flip conditions not met",
            reason_3="",
            details={},
        )

    freshness_score = _clamp((max_flip_age_bars - float(flip_age) + 1.0) / (max_flip_age_bars + 1.0) * 40.0, 0.0, 40.0)
    volume_score = _clamp((last_row["rel_volume_20"] - min_rel_volume_20) * 15.0, 0.0, 20.0)
    momentum_score = _clamp(abs(last_row["roc_5"]) * 6.0, 0.0, 20.0)

    trend_context_score = 0.0
    if int(current_state) == 1:
        trend_context_score += 8.0 if last_row["close"] > last_row["ema_50"] else 2.0
        trend_context_score += 6.0 if last_row["roc_20"] > 0 else 1.0
        trend_context_score += 6.0 if last_row["bullish_bar_flag"] == 1 else 1.0
        reason_1 = "fresh bullish flip"
    else:
        trend_context_score += 8.0 if last_row["close"] < last_row["ema_50"] else 2.0
        trend_context_score += 6.0 if last_row["roc_20"] < 0 else 1.0
        trend_context_score += 6.0 if last_row["bearish_bar_flag"] == 1 else 1.0
        reason_1 = "fresh bearish flip"

    trend_context_score = _clamp(trend_context_score, 0.0, 20.0)
    score = round(_clamp(freshness_score + volume_score + momentum_score + trend_context_score, 0.0, 100.0), 2)

    return ScreenerResult(
        screener_name="fresh_flip",
        eligible=True,
        score=score,
        reason_1=reason_1,
        reason_2=f"flip age {int(flip_age)} bars",
        reason_3="momentum confirmed",
        details={
            "flip_age_bars": float(flip_age),
            "previous_state": None if pd.isna(previous_state) else int(previous_state),
            "current_state": None if pd.isna(current_state) else int(current_state),
            "freshness_score": round(freshness_score, 2),
            "volume_score": round(volume_score, 2),
            "momentum_score": round(momentum_score, 2),
            "trend_context_score": round(trend_context_score, 2),
        },
    )
