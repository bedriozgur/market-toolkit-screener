from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: object, default: int | None = None) -> int | None:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def _pick_field(last_row: pd.Series, names: list[str], default: object = None) -> object:
    for name in names:
        value = last_row.get(name)
        if value is not None and not pd.isna(value):
            return value
    return default


def _safe_text(value: object, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    return str(value)


def _normalize_sentiment(value: float) -> float:
    if math.isnan(value):
        return float("nan")
    if -1.5 <= value <= 1.5:
        return _clamp(value, -1.0, 1.0)
    if 0.0 <= value <= 100.0:
        return _clamp((value - 50.0) / 50.0, -1.0, 1.0)
    if -100.0 <= value <= 100.0:
        return _clamp(value / 100.0, -1.0, 1.0)
    return _clamp(value / 1000.0, -1.0, 1.0)


def _band_score(value: float, bands: list[tuple[float, float]], fallback: float = 0.0) -> float:
    if math.isnan(value):
        return 0.0
    for threshold, points in bands:
        if value >= threshold:
            return points
    return fallback


def _fresh_flip_components(last_row: pd.Series) -> tuple[float, list[str], dict[str, Any]]:
    fresh_flip_cfg = {
        "alphatrend_max_bars": 5,
        "supertrend_max_bars": 5,
        "ema20_max_bars": 5,
        "alphatrend_points": 8.0,
        "supertrend_points": 5.0,
        "ema20_points": 4.0,
    }

    fresh_flips: list[str] = []
    details: dict[str, Any] = {}
    total = 0.0

    alphatrend_age = _safe_float(last_row.get("alphatrend_bars_since_valid_signal"))
    alphatrend_signal = _safe_text(last_row.get("alphatrend_last_valid_signal"))
    alphatrend_state = _safe_text(last_row.get("alphatrend_compare_state_label"))
    if not math.isnan(alphatrend_age) and alphatrend_age <= fresh_flip_cfg["alphatrend_max_bars"] and alphatrend_signal in {"BUY", "SELL"}:
        freshness = _clamp((fresh_flip_cfg["alphatrend_max_bars"] - alphatrend_age + 1.0) / (fresh_flip_cfg["alphatrend_max_bars"] + 1.0), 0.0, 1.0)
        points = fresh_flip_cfg["alphatrend_points"] * freshness
        if alphatrend_signal == "BUY" and alphatrend_state == "BUY":
            points += 1.5
        elif alphatrend_signal == "SELL" and alphatrend_state == "SELL":
            points += 1.5
        total += points
        fresh_flips.append(f"alphatrend:{alphatrend_signal}:{int(alphatrend_age)}")
        details["alphatrend_flip_age_bars"] = float(alphatrend_age)
        details["alphatrend_flip_signal"] = alphatrend_signal

    supertrend_age = _safe_float(last_row.get("supertrend_flip_age_bars"))
    supertrend_state = _safe_text(last_row.get("supertrend_state_label"))
    supertrend_bullish = bool(last_row.get("supertrend_bullish", False))
    if not math.isnan(supertrend_age) and supertrend_age <= fresh_flip_cfg["supertrend_max_bars"]:
        freshness = _clamp((fresh_flip_cfg["supertrend_max_bars"] - supertrend_age + 1.0) / (fresh_flip_cfg["supertrend_max_bars"] + 1.0), 0.0, 1.0)
        points = fresh_flip_cfg["supertrend_points"] * freshness
        if (supertrend_state == "BUY" and supertrend_bullish) or (supertrend_state == "SELL" and not supertrend_bullish):
            points += 1.0
        total += points
        fresh_flips.append(f"supertrend:{supertrend_state}:{int(supertrend_age)}")
        details["supertrend_flip_age_bars"] = float(supertrend_age)
        details["supertrend_state_label"] = supertrend_state

    ema20_age = _safe_float(last_row.get("flip_age_bars_ema20"))
    ema20_state = int(_safe_float(last_row.get("current_state_ema20"), default=float("nan"))) if not math.isnan(_safe_float(last_row.get("current_state_ema20"))) else None
    if not math.isnan(ema20_age) and ema20_age <= fresh_flip_cfg["ema20_max_bars"]:
        freshness = _clamp((fresh_flip_cfg["ema20_max_bars"] - ema20_age + 1.0) / (fresh_flip_cfg["ema20_max_bars"] + 1.0), 0.0, 1.0)
        points = fresh_flip_cfg["ema20_points"] * freshness
        if ema20_state == 1 and float(last_row.get("close", 0.0)) > float(last_row.get("ema_50", 0.0)):
            points += 1.0
        elif ema20_state == -1 and float(last_row.get("close", 0.0)) < float(last_row.get("ema_50", 0.0)):
            points += 1.0
        total += points
        fresh_flips.append(f"ema20:{'BUY' if ema20_state == 1 else 'SELL'}:{int(ema20_age)}")
        details["ema20_flip_age_bars"] = float(ema20_age)
        details["ema20_state"] = ema20_state

    return _clamp(total, 0.0, 15.0), fresh_flips, details


def _technical_score(last_row: pd.Series) -> tuple[float, dict[str, Any]]:
    close = _safe_float(last_row.get("close"))
    ema_20 = _safe_float(last_row.get("ema_20"))
    ema_50 = _safe_float(last_row.get("ema_50"))
    ema_200 = _safe_float(last_row.get("ema_200"))
    ema_20_slope_5 = _safe_float(last_row.get("ema_20_slope_5"))
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
    obv_trend_20 = _safe_float(last_row.get("obv_trend_20"))
    rs_line_slope_5 = _safe_float(last_row.get("rs_line_slope_5"))
    rs_line_slope_20 = _safe_float(last_row.get("rs_line_slope_20"))
    rs_line_dist_to_52w_high_pct = _safe_float(last_row.get("rs_line_dist_to_52w_high_pct"))
    roc_63_rank_pct = _safe_float(last_row.get("roc_63_rank_pct"))
    beta_adjusted_return_rank_pct = _safe_float(last_row.get("beta_adjusted_return_rank_pct"))
    peer_group_rank_pct = _safe_float(last_row.get("peer_group_rank_pct"))

    alphatrend_state = _safe_text(last_row.get("alphatrend_compare_state_label"))
    alphatrend_signal = _safe_text(last_row.get("alphatrend_last_valid_signal"))
    supertrend_bullish = bool(last_row.get("supertrend_bullish", False))
    ichimoku_price_above_cloud = bool(last_row.get("ichimoku_price_above_cloud", False))
    ichimoku_kumo_green = bool(last_row.get("ichimoku_kumo_green", False))
    ichimoku_tenkan_above_kijun = bool(last_row.get("ichimoku_tenkan_above_kijun", False))
    ichimoku_chikou_above_price_26 = bool(last_row.get("ichimoku_chikou_above_price_26", False))

    trend_score = 0.0
    if alphatrend_state in {"BUY", "SELL"}:
        trend_score += 7.0
    if alphatrend_signal == "BUY" and close >= ema_20:
        trend_score += 3.0
    elif alphatrend_signal == "SELL" and close <= ema_20:
        trend_score += 3.0
    trend_score += 6.0 if supertrend_bullish else 0.0
    trend_score += 4.0 if close > ema_20 else 0.0
    trend_score += 4.0 if close > ema_50 else 0.0
    trend_score += 4.0 if close > ema_200 else 0.0
    trend_score += 3.0 if ema_20_slope_5 > 0 else 0.0
    trend_score += 2.0 if ichimoku_price_above_cloud else 0.0
    trend_score += 2.0 if ichimoku_kumo_green else 0.0
    trend_score += 2.0 if ichimoku_tenkan_above_kijun else 0.0
    trend_score += 1.0 if ichimoku_chikou_above_price_26 else 0.0
    trend_score = _clamp(trend_score, 0.0, 30.0)

    momentum_score = 0.0
    momentum_score += _band_score(roc_5, [(15.0, 4.0), (8.0, 3.0), (3.0, 2.0), (0.0, 1.0)])
    momentum_score += _band_score(roc_10, [(12.0, 3.0), (6.0, 2.0), (0.0, 1.0)])
    momentum_score += _band_score(roc_20, [(15.0, 4.0), (8.0, 3.0), (3.0, 2.0), (0.0, 1.0)])
    momentum_score += _band_score(roc_63, [(25.0, 2.0), (10.0, 1.0)])
    momentum_score += 4.0 if macd_hist > 0 else 0.0
    momentum_score += 2.0 if macd_hist_delta > 0 else 0.0
    momentum_score += 2.0 if 55.0 <= rsi_14 <= 70.0 else 1.0 if 45.0 <= rsi_14 < 55.0 else 0.0
    momentum_score += 1.0 if 45.0 <= mfi_14 <= 70.0 else 0.0
    momentum_score = _clamp(momentum_score, 0.0, 20.0)

    structure_score = 0.0
    structure_score += 8.0 if dist_to_52w_high_pct <= 5.0 else 5.0 if dist_to_52w_high_pct <= 15.0 else 0.0
    structure_score += 4.0 if pd.notna(bb_percent_b) and bb_percent_b >= 0.8 else 2.0 if pd.notna(bb_percent_b) and bb_percent_b >= 0.5 else 0.0
    structure_score += 3.0 if close_position_in_20d_range >= 0.7 else 1.5 if close_position_in_20d_range >= 0.55 else 0.0
    structure_score += 2.0 if float(last_row.get("dist_to_50d_high_pct", 999.0)) <= 5.0 else 0.0
    structure_score = _clamp(structure_score, 0.0, 15.0)

    volume_score = 0.0
    volume_score += 4.0 if rel_volume_20 >= 1.5 else 2.0 if rel_volume_20 >= 1.0 else 0.0
    volume_score += 4.0 if volume_ratio_10_50 >= 1.25 else 2.0 if volume_ratio_10_50 >= 1.0 else 0.0
    volume_score += 2.0 if obv_trend_20 > 0 else 0.0
    volume_score = _clamp(volume_score, 0.0, 10.0)

    relative_score = 0.0
    if pd.notna(_pick_field(last_row, ["rs_line"])):
        relative_score += 4.0 if pd.notna(rs_line_dist_to_52w_high_pct) and rs_line_dist_to_52w_high_pct <= 5.0 else 2.0 if pd.notna(rs_line_dist_to_52w_high_pct) and rs_line_dist_to_52w_high_pct <= 15.0 else 0.0
        relative_score += 2.0 if pd.notna(rs_line_slope_20) and rs_line_slope_20 > 0 else 0.0
        relative_score += 1.0 if pd.notna(rs_line_slope_5) and rs_line_slope_5 > 0 else 0.0
    relative_score += 2.0 if pd.notna(roc_63_rank_pct) and roc_63_rank_pct >= 0.75 else 1.0 if pd.notna(roc_63_rank_pct) and roc_63_rank_pct >= 0.5 else 0.0
    relative_score += 2.0 if pd.notna(beta_adjusted_return_rank_pct) and beta_adjusted_return_rank_pct >= 0.75 else 1.0 if pd.notna(beta_adjusted_return_rank_pct) and beta_adjusted_return_rank_pct >= 0.5 else 0.0
    relative_score += 2.0 if pd.notna(peer_group_rank_pct) and peer_group_rank_pct >= 0.75 else 1.0 if pd.notna(peer_group_rank_pct) and peer_group_rank_pct >= 0.5 else 0.0
    relative_score = _clamp(relative_score, 0.0, 10.0)

    score = _clamp(trend_score + momentum_score + structure_score + volume_score + relative_score, 0.0, 85.0)
    return score, {
        "trend_score": round(trend_score, 2),
        "momentum_score": round(momentum_score, 2),
        "structure_score": round(structure_score, 2),
        "volume_score": round(volume_score, 2),
        "relative_score": round(relative_score, 2),
    }


def _fundamental_score(last_row: pd.Series) -> tuple[float, dict[str, Any]]:
    market_cap = _safe_float(_pick_field(last_row, ["market_cap", "mkt_cap", "marketcap"]))
    revenue_growth = _safe_float(_pick_field(last_row, ["revenue_growth_yoy", "sales_growth_yoy", "revenue_growth"]))
    eps_growth = _safe_float(_pick_field(last_row, ["eps_growth_yoy", "earnings_growth_yoy", "eps_growth"]))
    operating_margin = _safe_float(_pick_field(last_row, ["operating_margin", "operating_margin_pct", "op_margin"]))
    gross_margin = _safe_float(_pick_field(last_row, ["gross_margin", "gross_margin_pct"]))
    roe = _safe_float(_pick_field(last_row, ["roe", "return_on_equity"]))
    debt_to_equity = _safe_float(_pick_field(last_row, ["debt_to_equity", "debt_equity"]))
    current_ratio = _safe_float(_pick_field(last_row, ["current_ratio", "curr_ratio"]))

    score = 0.0
    if not math.isnan(market_cap):
        score += 3.0 if market_cap >= 10_000_000_000 else 2.0 if market_cap >= 1_000_000_000 else 1.0 if market_cap >= 250_000_000 else 0.0
    score += 5.0 if revenue_growth >= 20.0 else 3.0 if revenue_growth >= 10.0 else 1.0 if revenue_growth >= 5.0 else 0.0
    score += 5.0 if eps_growth >= 20.0 else 3.0 if eps_growth >= 10.0 else 1.0 if eps_growth >= 5.0 else 0.0
    score += 3.0 if operating_margin >= 15.0 else 2.0 if operating_margin >= 8.0 else 0.0
    score += 2.0 if gross_margin >= 35.0 else 1.0 if gross_margin >= 20.0 else 0.0
    score += 4.0 if roe >= 20.0 else 2.0 if roe >= 12.0 else 0.0
    score += 4.0 if 0.0 <= debt_to_equity <= 0.5 else 2.0 if 0.5 < debt_to_equity <= 1.0 else 0.0
    score += 2.0 if current_ratio >= 1.5 else 1.0 if current_ratio >= 1.0 else 0.0
    score = _clamp(score, 0.0, 20.0)
    return score, {
        "market_cap": None if math.isnan(market_cap) else market_cap,
        "revenue_growth_yoy": None if math.isnan(revenue_growth) else revenue_growth,
        "eps_growth_yoy": None if math.isnan(eps_growth) else eps_growth,
        "operating_margin": None if math.isnan(operating_margin) else operating_margin,
        "gross_margin": None if math.isnan(gross_margin) else gross_margin,
        "roe": None if math.isnan(roe) else roe,
        "debt_to_equity": None if math.isnan(debt_to_equity) else debt_to_equity,
        "current_ratio": None if math.isnan(current_ratio) else current_ratio,
    }


def _sentiment_score(last_row: pd.Series) -> tuple[float, dict[str, Any]]:
    reddit_value = _safe_float(_pick_field(last_row, ["reddit_sentiment_score", "reddit_sentiment", "reddit_score", "reddit_bullishness", "social_reddit_sentiment"]))
    twitter_value = _safe_float(_pick_field(last_row, ["twitter_sentiment_score", "twitter_sentiment", "twitter_score", "twitter_bullishness", "social_twitter_sentiment"]))
    combined_value = _safe_float(_pick_field(last_row, ["sentiment_score", "social_sentiment_score", "social_sentiment"]))

    score = 0.0
    reddit_norm = _normalize_sentiment(reddit_value)
    twitter_norm = _normalize_sentiment(twitter_value)
    combined_norm = _normalize_sentiment(combined_value)

    if not math.isnan(reddit_norm):
        score += reddit_norm * 5.0
    if not math.isnan(twitter_norm):
        score += twitter_norm * 5.0
    if not math.isnan(combined_norm):
        score += combined_norm * 3.0

    reddit_mentions = _safe_float(_pick_field(last_row, ["reddit_mentions", "reddit_volume", "reddit_post_count"]))
    twitter_mentions = _safe_float(_pick_field(last_row, ["twitter_mentions", "twitter_volume", "tweet_count"]))
    buzz_multiplier = 1.0
    if not math.isnan(reddit_mentions) and reddit_mentions > 0:
        buzz_multiplier += min(math.log1p(reddit_mentions) / 10.0, 0.5)
    if not math.isnan(twitter_mentions) and twitter_mentions > 0:
        buzz_multiplier += min(math.log1p(twitter_mentions) / 10.0, 0.5)
    score *= buzz_multiplier

    score = _clamp(score, -10.0, 10.0)
    return score, {
        "reddit_sentiment_score": None if math.isnan(reddit_value) else reddit_value,
        "twitter_sentiment_score": None if math.isnan(twitter_value) else twitter_value,
        "sentiment_score": None if math.isnan(combined_value) else combined_value,
        "reddit_mentions": None if math.isnan(reddit_mentions) else reddit_mentions,
        "twitter_mentions": None if math.isnan(twitter_mentions) else twitter_mentions,
    }


def _external_market_score(last_row: pd.Series) -> tuple[float, dict[str, Any]]:
    yf_analyst = _safe_float(last_row.get("yf_analyst_recommendation_score"))
    yf_price_target = _safe_float(last_row.get("yf_analyst_price_target_upside_score"))
    yf_news = _safe_float(last_row.get("yf_news_sentiment_score"))
    yf_upgrades = _safe_float(last_row.get("yf_upgrades_downgrades_score"))
    yf_earnings_growth = _safe_float(last_row.get("yf_earnings_growth_estimate"))
    yf_revenue_growth = _safe_float(last_row.get("yf_revenue_growth_estimate"))
    tv_recommendation = _safe_float(last_row.get("tv_technical_recommendation_score"))

    score = 0.0
    score += yf_analyst
    score += yf_price_target
    score += yf_news
    score += yf_upgrades
    score += _clamp(yf_earnings_growth / 5.0, -2.0, 2.0) if not math.isnan(yf_earnings_growth) else 0.0
    score += _clamp(yf_revenue_growth / 5.0, -2.0, 2.0) if not math.isnan(yf_revenue_growth) else 0.0
    score += tv_recommendation
    score = _clamp(score, -15.0, 15.0)
    return score, {
        "yf_analyst_recommendation_score": None if math.isnan(yf_analyst) else yf_analyst,
        "yf_analyst_price_target_upside_score": None if math.isnan(yf_price_target) else yf_price_target,
        "yf_news_sentiment_score": None if math.isnan(yf_news) else yf_news,
        "yf_upgrades_downgrades_score": None if math.isnan(yf_upgrades) else yf_upgrades,
        "yf_earnings_growth_estimate": None if math.isnan(yf_earnings_growth) else yf_earnings_growth,
        "yf_revenue_growth_estimate": None if math.isnan(yf_revenue_growth) else yf_revenue_growth,
        "tv_technical_recommendation_score": None if math.isnan(tv_recommendation) else tv_recommendation,
        "tv_technical_recommendation": _safe_text(last_row.get("tv_technical_recommendation")),
        "tv_oscillator_recommendation": _safe_text(last_row.get("tv_oscillator_recommendation")),
        "tv_moving_average_recommendation": _safe_text(last_row.get("tv_moving_average_recommendation")),
    }


def build_universe_rank_row(symbol: str, universe: str, interval: str, last_row: pd.Series) -> dict[str, Any]:
    technical_score, technical_details = _technical_score(last_row)
    fresh_flip_score, fresh_flip_signals, fresh_flip_details = _fresh_flip_components(last_row)
    fundamental_score, fundamental_details = _fundamental_score(last_row)
    sentiment_score, sentiment_details = _sentiment_score(last_row)
    external_score, external_details = _external_market_score(last_row)

    total_points = _clamp(technical_score + fresh_flip_score + fundamental_score + sentiment_score + external_score, 0.0, 100.0)

    row: dict[str, Any] = {
        "symbol": symbol,
        "universe": universe,
        "interval": interval,
        "date": last_row.get("date"),
        "close": _safe_float(last_row.get("close"), default=float("nan")),
        "score": round(total_points, 2),
        "total_points": round(total_points, 2),
        "technical_score": round(technical_score, 2),
        "fresh_flip_score": round(fresh_flip_score, 2),
        "fundamental_score": round(fundamental_score, 2),
        "sentiment_score": round(sentiment_score, 2),
        "external_score": round(external_score, 2),
        "fresh_flip_count": len(fresh_flip_signals),
        "fresh_flip_signals": " | ".join(fresh_flip_signals),
        "fresh_flip_summary": "; ".join(fresh_flip_signals),
        "alphatrend_last_valid_signal": last_row.get("alphatrend_last_valid_signal"),
        "alphatrend_compare_state_label": last_row.get("alphatrend_compare_state_label"),
        "alphatrend_bars_since_valid_signal": last_row.get("alphatrend_bars_since_valid_signal"),
        "alphatrend_flip_age_bars": last_row.get("alphatrend_flip_age_bars"),
        "supertrend_trend": last_row.get("supertrend_trend"),
        "supertrend_bullish": last_row.get("supertrend_bullish"),
        "supertrend_flip_age_bars": last_row.get("supertrend_flip_age_bars"),
        "flip_age_bars_ema20": last_row.get("flip_age_bars_ema20"),
        "current_state_ema20": last_row.get("current_state_ema20"),
        "market_cap": fundamental_details["market_cap"],
        "revenue_growth_yoy": fundamental_details["revenue_growth_yoy"],
        "eps_growth_yoy": fundamental_details["eps_growth_yoy"],
        "operating_margin": fundamental_details["operating_margin"],
        "gross_margin": fundamental_details["gross_margin"],
        "roe": fundamental_details["roe"],
        "debt_to_equity": fundamental_details["debt_to_equity"],
        "current_ratio": fundamental_details["current_ratio"],
        "reddit_sentiment_score": sentiment_details["reddit_sentiment_score"],
        "twitter_sentiment_score": sentiment_details["twitter_sentiment_score"],
        "social_sentiment_score": sentiment_details["sentiment_score"],
    }
    row.update(technical_details)
    row.update(fresh_flip_details)
    row.update(fundamental_details)
    row.update(sentiment_details)
    row.update(external_details)
    return row
