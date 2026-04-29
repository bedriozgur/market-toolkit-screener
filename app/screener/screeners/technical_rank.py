from __future__ import annotations

import math

import pandas as pd

from scripts.screener.models import ScreenerResult


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _safe_bool(value: object, default: bool = False) -> bool:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return bool(value)
    except Exception:
        return default


def _score_band(value: float, bands: list[tuple[float, float]], fallback: float = 0.0) -> float:
    for threshold, points in bands:
        if value >= threshold:
            return points
    return fallback


def evaluate(last_row: pd.Series, config: dict) -> ScreenerResult:
    screener_cfg = config.get("technical_rank", {})
    min_score = float(screener_cfg.get("min_score", 60.0))
    require_trend_alignment = bool(screener_cfg.get("require_trend_alignment", True))
    hard_filters = screener_cfg.get("hard_filters", {})
    sentiment_cfg = screener_cfg.get("sentiment_boost", {})
    missing_policy = str(hard_filters.get("missing_data_policy", "ignore")).lower()

    close = _safe_float(last_row.get("close"))
    ema_20 = _safe_float(last_row.get("ema_20"))
    ema_50 = _safe_float(last_row.get("ema_50"))
    ema_200 = _safe_float(last_row.get("ema_200"))
    ema_20_slope_5 = _safe_float(last_row.get("ema_20_slope_5"))
    supertrend_bullish = _safe_bool(last_row.get("supertrend_bullish", False))
    ichimoku_price_above_cloud = _safe_bool(last_row.get("ichimoku_price_above_cloud", False))
    ichimoku_kumo_green = _safe_bool(last_row.get("ichimoku_kumo_green", False))
    ichimoku_tenkan_above_kijun = _safe_bool(last_row.get("ichimoku_tenkan_above_kijun", False))
    ichimoku_chikou_above_price_26 = _safe_bool(last_row.get("ichimoku_chikou_above_price_26", False))
    roc_5 = _safe_float(last_row.get("roc_5"))
    roc_10 = _safe_float(last_row.get("roc_10"))
    roc_20 = _safe_float(last_row.get("roc_20"))
    roc_63 = _safe_float(last_row.get("roc_63"))
    rsi_14 = _safe_float(last_row.get("rsi_14"))
    mfi_14 = _safe_float(last_row.get("mfi_14"))
    macd_hist = _safe_float(last_row.get("macd_hist"))
    macd_hist_delta = _safe_float(last_row.get("macd_hist_delta"))
    bb_percent_b = _safe_float(last_row.get("bb_percent_b"))
    dist_to_52w_high_pct = _safe_float(last_row.get("dist_to_52w_high_pct"))
    close_position_in_20d_range = _safe_float(last_row.get("close_position_in_20d_range"))
    rel_volume_20 = _safe_float(last_row.get("rel_volume_20"))
    volume_ratio_10_50 = _safe_float(last_row.get("volume_ratio_10_50"))
    atr_pct_14 = _safe_float(last_row.get("atr_pct_14"))
    bullish_bar_flag = int(_safe_float(last_row.get("bullish_bar_flag")))
    close_above_ema20_count_20 = _safe_float(last_row.get("close_above_ema20_count_20"))
    adx_14 = _safe_float(last_row.get("adx_14"))
    plus_di_14 = _safe_float(last_row.get("plus_di_14"))
    minus_di_14 = _safe_float(last_row.get("minus_di_14"))
    aroon_up_25 = _safe_float(last_row.get("aroon_up_25"))
    aroon_down_25 = _safe_float(last_row.get("aroon_down_25"))
    aroon_osc_25 = _safe_float(last_row.get("aroon_osc_25"))
    obv_trend_20 = _safe_float(last_row.get("obv_trend_20"))
    rs_line = _safe_float(last_row.get("rs_line"), default=float("nan"))
    rs_line_slope_5 = _safe_float(last_row.get("rs_line_slope_5"), default=float("nan"))
    rs_line_slope_20 = _safe_float(last_row.get("rs_line_slope_20"), default=float("nan"))
    rs_line_dist_to_52w_high_pct = _safe_float(last_row.get("rs_line_dist_to_52w_high_pct"), default=float("nan"))
    roc_63_rank_pct = _safe_float(last_row.get("roc_63_rank_pct"), default=float("nan"))
    beta_adjusted_return_rank_pct = _safe_float(last_row.get("beta_adjusted_return_rank_pct"), default=float("nan"))
    peer_group_rank_pct = _safe_float(last_row.get("peer_group_rank_pct"), default=float("nan"))
    beta_adjusted_return_63 = _safe_float(last_row.get("beta_adjusted_return_63"), default=float("nan"))

    alphatrend_state_label = last_row.get("alphatrend_compare_state_label")
    alphatrend_last_valid_signal = last_row.get("alphatrend_last_valid_signal")
    alphatrend_bars_since_valid_signal = _safe_float(last_row.get("alphatrend_bars_since_valid_signal"), default=float("nan"))

    close_above_ema20 = close > ema_20 if pd.notna(ema_20) else False
    close_above_ema50 = close > ema_50 if pd.notna(ema_50) else False
    close_above_ema200 = close > ema_200 if pd.notna(ema_200) else False
    ema_stack_bullish = bool(close_above_ema20 and ema_20 > ema_50 and ema_50 > ema_200)
    trend_green = bool(ema_stack_bullish and ema_20_slope_5 > 0)
    alphatrend_buy_state = isinstance(alphatrend_state_label, str) and alphatrend_state_label == "BUY"
    alphatrend_buy_signal = isinstance(alphatrend_last_valid_signal, str) and alphatrend_last_valid_signal == "BUY"
    trend_bias = bool(
        alphatrend_buy_state
        or supertrend_bullish
        or ema_stack_bullish
        or ichimoku_price_above_cloud
        or (ichimoku_kumo_green and ichimoku_tenkan_above_kijun)
    )

    def _field_present(*names: str) -> tuple[str | None, float]:
        for name in names:
            value = last_row.get(name)
            if value is not None and not pd.isna(value):
                return name, float(value)
        return None, float("nan")

    def _optional_fail(condition: bool, reason: str) -> tuple[bool, str]:
        return (True, reason) if condition else (False, "")

    missing_fields: list[str] = []
    hard_failures: list[str] = []
    hard_filter_rules = [
        ("min_market_cap", ["market_cap", "mkt_cap", "marketcap"], lambda value, threshold: value < threshold, "market cap below threshold"),
        ("max_earnings_days", ["earnings_days_until", "days_to_earnings"], lambda value, threshold: value <= threshold, "earnings too close"),
        ("min_eps_growth_yoy", ["eps_growth_yoy"], lambda value, threshold: value < threshold, "EPS growth below threshold"),
        ("min_revenue_growth_yoy", ["revenue_growth_yoy"], lambda value, threshold: value < threshold, "revenue growth below threshold"),
        ("max_debt_to_equity", ["debt_to_equity"], lambda value, threshold: value > threshold, "debt/equity above threshold"),
    ]
    for cfg_key, field_names, comparator, message in hard_filter_rules:
        threshold = hard_filters.get(cfg_key)
        if threshold is None:
            continue
        _, value = _field_present(*field_names)
        if pd.isna(value):
            missing_fields.append(cfg_key)
            if missing_policy == "reject":
                hard_failures.append(f"{message} (missing data)")
            continue
        if comparator(value, float(threshold)):
            hard_failures.append(message)

    if hard_failures:
        return ScreenerResult(
            screener_name="technical_rank",
            eligible=False,
            score=0.0,
            reason_1="not eligible",
            reason_2="hard filters not met",
            reason_3="; ".join(hard_failures[:2]),
            details={
                "hard_failures": ",".join(hard_failures),
                "missing_hard_filter_fields": ",".join(missing_fields),
            },
        )

    if require_trend_alignment and not trend_bias:
        return ScreenerResult(
            screener_name="technical_rank",
            eligible=False,
            score=0.0,
            reason_1="not eligible",
            reason_2="trend alignment missing",
            reason_3="",
            details={},
        )

    trend_score = 0.0
    trend_score += 8.0 if alphatrend_buy_state else 0.0
    trend_score += 6.0 if supertrend_bullish else 0.0
    trend_score += 4.0 if close_above_ema20 else 0.0
    trend_score += 3.0 if close_above_ema50 else 0.0
    trend_score += 3.0 if close_above_ema200 else 0.0
    trend_score += 2.0 if ema_20_slope_5 > 0 else 0.0
    trend_score += 2.0 if close_above_ema20_count_20 >= 14 else 0.0
    trend_score += 2.0 if ichimoku_price_above_cloud else 0.0
    trend_score += 1.0 if ichimoku_chikou_above_price_26 else 0.0
    trend_score = _clamp(trend_score, 0.0, 30.0)

    trend_strength_score = 0.0
    trend_strength_score += 4.0 if adx_14 >= 25.0 else 2.0 if adx_14 >= 20.0 else 0.0
    trend_strength_score += 2.0 if plus_di_14 > minus_di_14 else 0.0
    trend_strength_score += 2.0 if aroon_up_25 >= 70.0 and aroon_down_25 <= 30.0 else 1.0 if aroon_osc_25 > 0 else 0.0
    trend_strength_score += 2.0 if pd.notna(rs_line_slope_20) and rs_line_slope_20 > 0 else 0.0
    trend_strength_score = _clamp(trend_strength_score, 0.0, 10.0)

    momentum_score = 0.0
    momentum_score += 8.0 if macd_hist > 0 else 0.0
    momentum_score += 4.0 if macd_hist_delta > 0 else 0.0
    momentum_score += _score_band(roc_63_rank_pct, [(0.8, 4.0), (0.6, 3.0), (0.4, 2.0), (0.2, 1.0)])
    momentum_score += _score_band(roc_20, [(15.0, 4.0), (8.0, 3.0), (3.0, 2.0), (0.0, 1.0)])
    momentum_score += 2.0 if 55.0 <= rsi_14 <= 70.0 else 1.0 if 45.0 <= rsi_14 < 55.0 else 0.0
    momentum_score += 1.0 if 50.0 <= mfi_14 <= 70.0 else 0.0
    momentum_score = _clamp(momentum_score, 0.0, 20.0)

    structure_score = 0.0
    structure_score += 10.0 if dist_to_52w_high_pct <= 5.0 else 6.0 if dist_to_52w_high_pct <= 15.0 else 0.0
    structure_score += 5.0 if pd.notna(bb_percent_b) and bb_percent_b >= 0.8 else 3.0 if pd.notna(bb_percent_b) and bb_percent_b >= 0.5 else 0.0
    structure_score += 5.0 if close_position_in_20d_range >= 0.7 else 3.0 if close_position_in_20d_range >= 0.55 else 0.0
    structure_score += 2.0 if ichimoku_kumo_green else 0.0
    structure_score = _clamp(structure_score, 0.0, 15.0)

    volume_score = 0.0
    volume_score += 4.0 if rel_volume_20 >= 1.5 else 2.0 if rel_volume_20 >= 1.0 else 0.0
    volume_score += 4.0 if volume_ratio_10_50 >= 1.25 else 2.0 if volume_ratio_10_50 >= 1.0 else 0.0
    volume_score += 2.0 if obv_trend_20 > 0 else 0.0
    volume_score = _clamp(volume_score, 0.0, 10.0)

    relative_score = 0.0
    if pd.notna(rs_line):
        relative_score += 6.0 if pd.notna(rs_line_dist_to_52w_high_pct) and rs_line_dist_to_52w_high_pct <= 5.0 else 4.0 if pd.notna(rs_line_dist_to_52w_high_pct) and rs_line_dist_to_52w_high_pct <= 15.0 else 0.0
        relative_score += 2.0 if pd.notna(rs_line_slope_20) and rs_line_slope_20 > 0 else 0.0
        relative_score += 1.0 if pd.notna(rs_line_slope_5) and rs_line_slope_5 > 0 else 0.0
    relative_score += 3.0 if pd.notna(roc_63_rank_pct) and roc_63_rank_pct >= 0.75 else 2.0 if pd.notna(roc_63_rank_pct) and roc_63_rank_pct >= 0.5 else 0.0
    relative_score += 2.0 if pd.notna(beta_adjusted_return_rank_pct) and beta_adjusted_return_rank_pct >= 0.75 else 1.0 if pd.notna(beta_adjusted_return_rank_pct) and beta_adjusted_return_rank_pct >= 0.5 else 0.0
    relative_score += 2.0 if pd.notna(peer_group_rank_pct) and peer_group_rank_pct >= 0.75 else 1.0 if pd.notna(peer_group_rank_pct) and peer_group_rank_pct >= 0.5 else 0.0
    relative_score = _clamp(relative_score, 0.0, 15.0)

    sentiment_bonus = 0.0
    if bool(sentiment_cfg.get("enabled", False)):
        sentiment_bonus += 1.5 if _safe_float(last_row.get("short_interest_pct"), default=0.0) >= float(sentiment_cfg.get("min_short_interest_pct", 15.0)) else 0.0
        sentiment_bonus += 1.5 if _safe_float(last_row.get("insider_buying_score"), default=0.0) > 0 else 0.0
        sentiment_bonus += 1.0 if _safe_float(last_row.get("analyst_revision_score"), default=0.0) > 0 else 0.0
        sentiment_bonus += 1.0 if _safe_float(last_row.get("earnings_surprise_score"), default=0.0) > 0 else 0.0
    sentiment_bonus = _clamp(sentiment_bonus, 0.0, 5.0)

    score = round(_clamp(trend_score + trend_strength_score + momentum_score + structure_score + volume_score + relative_score + sentiment_bonus, 0.0, 100.0), 2)

    flags: list[str] = []
    if ema_stack_bullish:
        flags.append("ema_stack_bullish")
    if supertrend_bullish:
        flags.append("supertrend_bullish")
    if trend_green:
        flags.append("trend_green")
    if ichimoku_price_above_cloud:
        flags.append("ichimoku_above_cloud")
    if ichimoku_kumo_green:
        flags.append("ichimoku_kumo_green")
    if ichimoku_chikou_above_price_26:
        flags.append("ichimoku_chikou_confirmed")
    if dist_to_52w_high_pct <= 5.0:
        flags.append("near_52w_high")
    if bb_percent_b >= 0.8:
        flags.append("bb_upper_band")
    if rel_volume_20 >= 1.5:
        flags.append("volume_confirmed")
    if volume_ratio_10_50 >= 1.25:
        flags.append("volume_expanding")
    if obv_trend_20 > 0:
        flags.append("obv_trending")
    if macd_hist > 0 and macd_hist_delta > 0:
        flags.append("momentum_expanding")
    if alphatrend_buy_signal:
        flags.append("alphatrend_buy")
    if pd.notna(rs_line) and pd.notna(rs_line_dist_to_52w_high_pct) and rs_line_dist_to_52w_high_pct <= 5.0:
        flags.append("rs_leader")
    if pd.notna(beta_adjusted_return_rank_pct) and beta_adjusted_return_rank_pct >= 0.75:
        flags.append("beta_adj_leader")
    if pd.notna(peer_group_rank_pct) and peer_group_rank_pct >= 0.75:
        flags.append("peer_group_leader")

    if score < min_score:
        return ScreenerResult(
            screener_name="technical_rank",
            eligible=False,
            score=round(score, 2),
            reason_1="not eligible",
            reason_2=f"score below threshold ({score:.2f} < {min_score:.2f})",
            reason_3="",
            details={
                "trend_score": round(trend_score, 2),
                "trend_strength_score": round(trend_strength_score, 2),
                "momentum_score": round(momentum_score, 2),
                "structure_score": round(structure_score, 2),
                "volume_score": round(volume_score, 2),
                "relative_score": round(relative_score, 2),
                "sentiment_bonus": round(sentiment_bonus, 2),
                "beta_adjusted_return_63": round(beta_adjusted_return_63, 4) if pd.notna(beta_adjusted_return_63) else None,
                "flags": ",".join(flags),
                "hard_failures": "",
            },
        )

    reasons = []
    if ema_stack_bullish:
        reasons.append("ema stack bullish")
    elif trend_green:
        reasons.append("trend strengthening")
    else:
        reasons.append("trend supported")

    if dist_to_52w_high_pct <= 5.0:
        reasons.append("near 52w high")
    elif roc_20 > 0:
        reasons.append("positive 20d momentum")
    else:
        reasons.append("momentum holding")

    if rel_volume_20 >= 1.0 or volume_ratio_10_50 >= 1.0:
        reasons.append("volume supportive")
    else:
        reasons.append("quiet volume")

    while len(reasons) < 3:
        reasons.append("")

    return ScreenerResult(
        screener_name="technical_rank",
        eligible=True,
        score=score,
        reason_1=reasons[0],
        reason_2=reasons[1],
        reason_3=reasons[2],
        details={
            "trend_score": round(trend_score, 2),
            "trend_strength_score": round(trend_strength_score, 2),
            "momentum_score": round(momentum_score, 2),
            "structure_score": round(structure_score, 2),
            "volume_score": round(volume_score, 2),
            "relative_score": round(relative_score, 2),
            "sentiment_bonus": round(sentiment_bonus, 2),
            "beta_adjusted_return_63": round(beta_adjusted_return_63, 4) if pd.notna(beta_adjusted_return_63) else None,
            "flags": ",".join(flags),
            "hard_failures": "",
        },
    )
