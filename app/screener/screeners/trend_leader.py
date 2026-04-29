from __future__ import annotations

import pandas as pd

from scripts.screener.models import ScreenerResult


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def evaluate(last_row: pd.Series) -> ScreenerResult:
    eligible = bool(
        last_row["close"] > last_row["ema_20"]
        and last_row["ema_20"] > last_row["ema_50"]
        and last_row["ema_50"] > last_row["ema_200"]
        and last_row["roc_20"] > 0
        and last_row["rel_volume_20"] >= 0.8
    )

    if not eligible:
        return ScreenerResult(
            screener_name="trend_leader",
            eligible=False,
            score=0.0,
            reason_1="not eligible",
            reason_2="trend alignment missing",
            reason_3="",
            details={},
        )

    trend_alignment_score = 0.0
    trend_alignment_score += 15.0 if last_row["close"] > last_row["ema_20"] else 0.0
    trend_alignment_score += 15.0 if last_row["ema_20"] > last_row["ema_50"] else 0.0
    trend_alignment_score += 20.0 if last_row["ema_50"] > last_row["ema_200"] else 0.0

    momentum_raw = 0.4 * last_row["roc_5"] + 0.3 * last_row["roc_10"] + 0.3 * last_row["roc_20"]
    momentum_score = _clamp(momentum_raw * 1.5, 0.0, 20.0)
    structure_score = _clamp((10.0 - last_row["dist_to_50d_high_pct"]) * 1.5, 0.0, 15.0)
    volume_score = _clamp((last_row["rel_volume_20"] - 0.8) * 5.0, 0.0, 10.0)
    stability_score = _clamp(last_row["close_above_ema20_count_20"] / 20.0 * 5.0, 0.0, 5.0)

    score = round(_clamp(trend_alignment_score + momentum_score + structure_score + volume_score + stability_score, 0.0, 100.0), 2)

    reasons = ["trend aligned"]
    if last_row["dist_to_50d_high_pct"] <= 3.0:
        reasons.append("near 50d high")
    if last_row["roc_20"] > 0:
        reasons.append("positive 20d momentum")
    while len(reasons) < 3:
        reasons.append("")

    return ScreenerResult(
        screener_name="trend_leader",
        eligible=True,
        score=score,
        reason_1=reasons[0],
        reason_2=reasons[1],
        reason_3=reasons[2],
        details={
            "trend_alignment_score": round(trend_alignment_score, 2),
            "momentum_score": round(momentum_score, 2),
            "structure_score": round(structure_score, 2),
            "volume_score": round(volume_score, 2),
            "stability_score": round(stability_score, 2),
        },
    )
