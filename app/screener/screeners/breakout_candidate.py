from __future__ import annotations

import pandas as pd

from scripts.screener.models import ScreenerResult


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def evaluate(last_row: pd.Series) -> ScreenerResult:
    near_high = last_row["dist_to_20d_high_pct"] <= 3.0 or last_row["dist_to_50d_high_pct"] <= 5.0
    contraction = last_row["rolling_range_20"] < last_row["rolling_range_20_sma_60"]

    eligible = bool(
        last_row["close"] > last_row["ema_20"]
        and last_row["close"] > last_row["ema_50"]
        and near_high
        and contraction
        and last_row["atr_pct_14"] <= 8.0
        and last_row["rel_volume_20"] >= 0.9
    )

    if not eligible:
        return ScreenerResult(
            screener_name="breakout_candidate",
            eligible=False,
            score=0.0,
            reason_1="not eligible",
            reason_2="breakout setup missing",
            reason_3="",
            details={},
        )

    proximity_score = _clamp((5.0 - min(last_row["dist_to_20d_high_pct"], last_row["dist_to_50d_high_pct"])) * 6.0, 0.0, 30.0)
    contraction_gap = last_row["rolling_range_20_sma_60"] - last_row["rolling_range_20"]
    contraction_score = _clamp(contraction_gap * 3.0, 0.0, 25.0)

    trend_context_score = 0.0
    trend_context_score += 10.0 if last_row["close"] > last_row["ema_20"] else 0.0
    trend_context_score += 10.0 if last_row["close"] > last_row["ema_50"] else 0.0

    volume_expansion_score = 0.0
    volume_expansion_score += _clamp((last_row["rel_volume_20"] - 0.9) * 8.0, 0.0, 10.0)
    volume_expansion_score += 5.0 if last_row["avg_rel_volume_3"] > 1.0 else 0.0
    volume_expansion_score = _clamp(volume_expansion_score, 0.0, 15.0)

    bar_quality_score = 0.0
    bar_quality_score += 5.0 if last_row["bullish_bar_flag"] == 1 else 0.0
    bar_quality_score += 5.0 if last_row["close_position_in_20d_range"] >= 0.8 else 0.0

    score = round(
        _clamp(proximity_score + contraction_score + trend_context_score + volume_expansion_score + bar_quality_score, 0.0, 100.0),
        2,
    )

    return ScreenerResult(
        screener_name="breakout_candidate",
        eligible=True,
        score=score,
        reason_1="tight range near highs",
        reason_2="breakout pressure building",
        reason_3="volume supportive",
        details={
            "proximity_score": round(proximity_score, 2),
            "contraction_score": round(contraction_score, 2),
            "trend_context_score": round(trend_context_score, 2),
            "volume_expansion_score": round(volume_expansion_score, 2),
            "bar_quality_score": round(bar_quality_score, 2),
        },
    )
