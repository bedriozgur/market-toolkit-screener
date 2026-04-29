from __future__ import annotations

import pandas as pd

from scripts.screener.models import ScreenerResult


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def evaluate(last_row: pd.Series) -> ScreenerResult:
    near_ema20 = abs(last_row["close"] - last_row["ema_20"]) / last_row["ema_20"] <= 0.03
    near_ema50 = abs(last_row["close"] - last_row["ema_50"]) / last_row["ema_50"] <= 0.04
    below_recent_high = 5.0 <= last_row["dist_to_50d_high_pct"] <= 12.0

    eligible = bool(
        last_row["ema_50"] > last_row["ema_200"]
        and last_row["close"] > last_row["ema_200"]
        and below_recent_high
        and (near_ema20 or near_ema50)
        and last_row["roc_20"] > -8.0
    )

    if not eligible:
        return ScreenerResult(
            screener_name="pullback",
            eligible=False,
            score=0.0,
            reason_1="not eligible",
            reason_2="pullback conditions not met",
            reason_3="",
            details={},
        )

    higher_trend_score = 30.0
    dist_ema20_pct = abs(last_row["close"] - last_row["ema_20"]) / last_row["ema_20"] * 100.0
    dist_ema50_pct = abs(last_row["close"] - last_row["ema_50"]) / last_row["ema_50"] * 100.0
    best_dist = min(dist_ema20_pct, dist_ema50_pct)
    pullback_depth_score = _clamp(25.0 - best_dist * 5.0, 0.0, 25.0)

    support_quality_score = 0.0
    support_quality_score += 8.0 if last_row["lower_wick_pct"] > 1.0 else 0.0
    support_quality_score += 8.0 if last_row["bullish_bar_flag"] == 1 else 0.0
    support_quality_score += 4.0 if last_row["rsi_14"] > 40.0 else 0.0
    support_quality_score = _clamp(support_quality_score, 0.0, 20.0)

    volume_behavior_score = 0.0
    volume_behavior_score += 5.0 if last_row["rel_volume_20"] <= 1.2 else 2.0
    volume_behavior_score += 5.0 if last_row["avg_rel_volume_3"] >= 0.9 else 0.0
    volume_behavior_score = _clamp(volume_behavior_score, 0.0, 10.0)

    bounce_potential_score = 0.0
    close_pos = last_row["close_position_in_20d_range"]
    if pd.notna(close_pos):
        bounce_potential_score += _clamp((0.7 - abs(close_pos - 0.5)) * 15.0, 0.0, 8.0)
    bounce_potential_score += 7.0 if last_row["roc_5"] > -2.0 else 2.0
    bounce_potential_score = _clamp(bounce_potential_score, 0.0, 15.0)

    score = round(
        _clamp(
            higher_trend_score + pullback_depth_score + support_quality_score + volume_behavior_score + bounce_potential_score,
            0.0,
            100.0,
        ),
        2,
    )

    first_reason = "pullback to ema20" if near_ema20 else "pullback to ema50"
    return ScreenerResult(
        screener_name="pullback",
        eligible=True,
        score=score,
        reason_1=first_reason,
        reason_2="long-term trend intact",
        reason_3="reversal potential near support",
        details={
            "higher_trend_score": round(higher_trend_score, 2),
            "pullback_depth_score": round(pullback_depth_score, 2),
            "support_quality_score": round(support_quality_score, 2),
            "volume_behavior_score": round(volume_behavior_score, 2),
            "bounce_potential_score": round(bounce_potential_score, 2),
        },
    )
