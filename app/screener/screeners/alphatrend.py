from __future__ import annotations

import pandas as pd

from scripts.screener.models import ScreenerResult


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def evaluate(last_row: pd.Series, config: dict) -> ScreenerResult:
    screener_cfg = config.get("alphatrend", {})
    max_flip_age_bars = int(screener_cfg.get("max_flip_age_bars", 7))
    min_rel_volume_20 = float(screener_cfg.get("min_rel_volume_20", 0.8))
    min_abs_roc_5 = float(screener_cfg.get("min_abs_roc_5", 0.25))
    require_price_confirmation = bool(screener_cfg.get("require_price_confirmation", True))

    flip_age = last_row.get("alphatrend_flip_age_bars")
    current_state = last_row.get("alphatrend_state")
    previous_state = last_row.get("alphatrend_prev_state")
    alphatrend_value = last_row.get("alphatrend")
    alphatrend_lag2 = last_row.get("alphatrend_lag2")

    if pd.isna(current_state) or pd.isna(alphatrend_value) or pd.isna(alphatrend_lag2):
        return ScreenerResult(
            screener_name="alphatrend",
            eligible=False,
            score=0.0,
            reason_1="not eligible",
            reason_2="alphatrend not ready",
            reason_3="",
            details={},
        )

    if require_price_confirmation:
        if int(current_state) == 1:
            price_confirmed = bool(last_row["close"] >= last_row["ema_20"])
        else:
            price_confirmed = bool(last_row["close"] <= last_row["ema_20"])
    else:
        price_confirmed = True

    eligible = bool(
        pd.notna(flip_age)
        and flip_age <= max_flip_age_bars
        and last_row["rel_volume_20"] >= min_rel_volume_20
        and abs(last_row["roc_5"]) >= min_abs_roc_5
        and price_confirmed
    )

    if not eligible:
        return ScreenerResult(
            screener_name="alphatrend",
            eligible=False,
            score=0.0,
            reason_1="not eligible",
            reason_2="alphatrend conditions not met",
            reason_3="",
            details={},
        )

    freshness_score = _clamp((max_flip_age_bars - float(flip_age) + 1.0) / (max_flip_age_bars + 1.0) * 35.0, 0.0, 35.0)
    volume_score = _clamp((last_row["rel_volume_20"] - min_rel_volume_20) * 14.0, 0.0, 15.0)
    momentum_score = _clamp(abs(last_row["roc_5"]) * 5.0, 0.0, 15.0)

    separation_pct = abs(alphatrend_value - alphatrend_lag2) / last_row["close"] * 100.0
    separation_score = _clamp(separation_pct * 8.0, 0.0, 15.0)

    trend_context_score = 0.0
    if int(current_state) == 1:
        trend_context_score += 7.0 if last_row["close"] > last_row["ema_50"] else 2.0
        trend_context_score += 7.0 if last_row["roc_20"] > 0 else 2.0
        trend_context_score += 6.0 if last_row["bullish_bar_flag"] == 1 else 2.0
        reason_1 = "alphatrend bullish flip"
        reason_3 = "buy state active"
    else:
        trend_context_score += 7.0 if last_row["close"] < last_row["ema_50"] else 2.0
        trend_context_score += 7.0 if last_row["roc_20"] < 0 else 2.0
        trend_context_score += 6.0 if last_row["bearish_bar_flag"] == 1 else 2.0
        reason_1 = "alphatrend bearish flip"
        reason_3 = "sell state active"

    trend_context_score = _clamp(trend_context_score, 0.0, 20.0)
    score = round(_clamp(freshness_score + volume_score + momentum_score + separation_score + trend_context_score, 0.0, 100.0), 2)

    return ScreenerResult(
        screener_name="alphatrend",
        eligible=True,
        score=score,
        reason_1=reason_1,
        reason_2=f"flip age {int(flip_age)} bars",
        reason_3=reason_3,
        details={
            "alphatrend_value": round(float(alphatrend_value), 6),
            "alphatrend_lag2": round(float(alphatrend_lag2), 6),
            "alphatrend_flip_age_bars": float(flip_age),
            "alphatrend_previous_state": None if pd.isna(previous_state) else int(previous_state),
            "alphatrend_current_state": int(current_state),
            "alphatrend_buy_flag": int(last_row.get("alphatrend_buy_flag", 0)),
            "alphatrend_sell_flag": int(last_row.get("alphatrend_sell_flag", 0)),
            "freshness_score": round(freshness_score, 2),
            "volume_score": round(volume_score, 2),
            "momentum_score": round(momentum_score, 2),
            "separation_score": round(separation_score, 2),
            "trend_context_score": round(trend_context_score, 2),
        },
    )
