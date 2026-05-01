from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.workspace_config import get_workspace_root


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_text(value: object, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    return str(value)


def _cache_root() -> Path:
    return get_workspace_root() / "cache" / "market_data_enrichment"


def _cache_path(symbol: str, interval: str, provider: str) -> Path:
    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    return _cache_root() / provider / interval / f"{safe_symbol}.json"


def _load_cached(path: Path, ttl_hours: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fetched_at = payload.get("fetched_at")
        if not fetched_at:
            return None
        fetched_dt = datetime.fromisoformat(fetched_at)
        age_hours = (datetime.now(timezone.utc) - fetched_dt.astimezone(timezone.utc)).total_seconds() / 3600.0
        if age_hours > ttl_hours:
            return None
        data = payload.get("data")
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_cache(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _default_tradingview_profile(universe: str) -> dict[str, str]:
    universe_key = (universe or "").strip().lower()
    if universe_key.startswith("bist"):
        return {"screener": "turkey", "exchange": "BIST"}
    return {"screener": "america", "exchange": "NASDAQ"}


def _resolve_tradingview_config(universe: str, cfg: dict[str, Any]) -> dict[str, Any]:
    resolved = _default_tradingview_profile(universe)
    tv_cfg = dict(cfg.get("tradingview_ta", {}))
    screener = _safe_text(tv_cfg.get("screener")).strip()
    exchange = _safe_text(tv_cfg.get("exchange")).strip()
    if screener:
        resolved["screener"] = screener
    if exchange:
        resolved["exchange"] = exchange
    if tv_cfg.get("symbol_override"):
        resolved["symbol_override"] = _safe_text(tv_cfg.get("symbol_override")).strip()
    if tv_cfg.get("interval"):
        resolved["interval"] = _safe_text(tv_cfg.get("interval")).strip()
    if tv_cfg.get("timeout") is not None:
        resolved["timeout"] = tv_cfg.get("timeout")
    return resolved


def _score_recommendation_counts(counts: dict[str, Any]) -> tuple[float, str]:
    strong_buy = _safe_float(counts.get("strongBuy"), default=0.0)
    buy = _safe_float(counts.get("buy"), default=0.0)
    hold = _safe_float(counts.get("hold"), default=0.0)
    sell = _safe_float(counts.get("sell"), default=0.0)
    strong_sell = _safe_float(counts.get("strongSell"), default=0.0)
    total = strong_buy + buy + hold + sell + strong_sell
    if total <= 0:
        return 0.0, ""

    bias = ((strong_buy * 1.0) + (buy * 0.5) - (sell * 0.5) - (strong_sell * 1.0)) / total
    bias = _clamp(bias, -1.0, 1.0)
    label = _safe_text(counts.get("RECOMMENDATION") or counts.get("recommendation") or "").upper()
    return bias * 10.0, label


def _score_price_targets(current: float, targets: dict[str, Any]) -> float:
    if math.isnan(current) or current <= 0:
        return 0.0
    mean_target = _safe_float(targets.get("mean"), default=float("nan"))
    high_target = _safe_float(targets.get("high"), default=float("nan"))
    low_target = _safe_float(targets.get("low"), default=float("nan"))

    upside = 0.0
    if not math.isnan(mean_target):
        upside = max(upside, ((mean_target / current) - 1.0) * 100.0)
    if not math.isnan(high_target):
        upside = max(upside, ((high_target / current) - 1.0) * 100.0)
    if not math.isnan(low_target):
        downside = ((low_target / current) - 1.0) * 100.0
        upside = max(upside, downside * 0.15)

    return _clamp(upside / 4.0, -5.0, 5.0)


def _score_news(items: list[dict[str, Any]]) -> tuple[float, int]:
    positive_words = {
        "beat",
        "beats",
        "bullish",
        "upgrade",
        "upgrades",
        "growth",
        "surge",
        "surges",
        "rally",
        "record",
        "strong",
        "expands",
        "expansion",
        "outperform",
        "buy",
        "buyback",
        "profit",
        "profits",
    }
    negative_words = {
        "miss",
        "misses",
        "bearish",
        "downgrade",
        "downgrades",
        "loss",
        "losses",
        "lawsuit",
        "probe",
        "cuts",
        "cut",
        "weak",
        "warning",
        "decline",
        "drop",
        "sell",
    }

    score = 0.0
    count = 0
    for item in items:
        title = _safe_text(item.get("title") or item.get("content") or item.get("summary") or "")
        if not title:
            continue
        count += 1
        lower = title.lower()
        pos_hits = sum(1 for word in positive_words if word in lower)
        neg_hits = sum(1 for word in negative_words if word in lower)
        score += pos_hits - neg_hits

    score = _clamp(score / max(count, 1), -3.0, 3.0)
    return score * 2.0, count


def _fetch_yfinance(symbol: str) -> dict[str, Any]:
    import yfinance as yf  # lazy optional dependency

    ticker = yf.Ticker(symbol)
    current_price = float("nan")
    info: dict[str, Any] = {}
    try:
        info = dict(ticker.info or {})
        current_price = _safe_float(info.get("currentPrice"), default=float("nan"))
        if math.isnan(current_price):
            current_price = _safe_float(info.get("regularMarketPrice"), default=float("nan"))
    except Exception:
        info = {}

    recommendation_summary = {}
    try:
        recommendation_summary = dict(ticker.get_recommendations_summary(as_dict=True) or {})
    except Exception:
        try:
            summary_df = ticker.recommendations_summary
            if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
                recommendation_summary = summary_df.iloc[0].to_dict()
        except Exception:
            recommendation_summary = {}

    analyst_recommendation_score, analyst_recommendation_label = _score_recommendation_counts(recommendation_summary)

    price_targets = {}
    try:
        price_targets = dict(ticker.get_analyst_price_targets() or {})
    except Exception:
        price_targets = {}

    news_items: list[dict[str, Any]] = []
    try:
        news_items = list(ticker.get_news(count=10, tab="news") or [])
    except Exception:
        try:
            news_items = list(ticker.news or [])
        except Exception:
            news_items = []
    news_sentiment_score, news_item_count = _score_news(news_items)

    upgrades_score = 0.0
    try:
        upgrades = ticker.get_upgrades_downgrades(as_dict=True)
        if isinstance(upgrades, pd.DataFrame) and not upgrades.empty:
            recent = upgrades.tail(10)
            for _, row in recent.iterrows():
                to_grade = _safe_text(row.get("toGrade")).lower()
                from_grade = _safe_text(row.get("fromGrade")).lower()
                action = _safe_text(row.get("action")).lower()
                if action == "up":
                    upgrades_score += 0.5
                elif action == "down":
                    upgrades_score -= 0.5
                if "buy" in to_grade and "sell" in from_grade:
                    upgrades_score += 1.0
                if "sell" in to_grade and "buy" in from_grade:
                    upgrades_score -= 1.0
    except Exception:
        upgrades_score = 0.0
    upgrades_score = _clamp(upgrades_score, -5.0, 5.0)

    earnings_growth = float("nan")
    revenue_growth = float("nan")
    try:
        earnings_estimate = ticker.get_earnings_estimate(as_dict=True)
        if isinstance(earnings_estimate, pd.DataFrame) and "+1y" in earnings_estimate.index:
            earnings_growth = _safe_float(earnings_estimate.loc["+1y"].get("growth"), default=float("nan"))
    except Exception:
        pass
    try:
        revenue_estimate = ticker.get_revenue_estimate(as_dict=True)
        if isinstance(revenue_estimate, pd.DataFrame) and "+1y" in revenue_estimate.index:
            revenue_growth = _safe_float(revenue_estimate.loc["+1y"].get("growth"), default=float("nan"))
    except Exception:
        pass

    analyst_price_target_score = _score_price_targets(current_price, price_targets)
    earnings_estimate_score = _clamp(earnings_growth / 10.0, -3.0, 3.0) if not math.isnan(earnings_growth) else 0.0
    revenue_estimate_score = _clamp(revenue_growth / 10.0, -3.0, 3.0) if not math.isnan(revenue_growth) else 0.0

    return {
        "source": "yfinance",
        "yf_ticker_name": info.get("shortName") or info.get("longName"),
        "yf_exchange": info.get("exchange"),
        "yf_currency": info.get("currency"),
        "yf_current_price": current_price if not math.isnan(current_price) else None,
        "yf_analyst_recommendation_label": analyst_recommendation_label,
        "yf_analyst_recommendation_score": round(analyst_recommendation_score, 2),
        "yf_analyst_price_target_upside_score": round(analyst_price_target_score, 2),
        "yf_analyst_price_target_mean": price_targets.get("mean"),
        "yf_analyst_price_target_median": price_targets.get("median"),
        "yf_analyst_price_target_high": price_targets.get("high"),
        "yf_analyst_price_target_low": price_targets.get("low"),
        "yf_news_sentiment_score": round(news_sentiment_score, 2),
        "yf_news_item_count": news_item_count,
        "yf_upgrades_downgrades_score": round(upgrades_score, 2),
        "yf_earnings_growth_estimate": None if math.isnan(earnings_growth) else earnings_growth,
        "yf_revenue_growth_estimate": None if math.isnan(revenue_growth) else revenue_growth,
    }


def _score_tv_recommendation(label: str, buy: int, neutral: int, sell: int) -> float:
    total = buy + neutral + sell
    if total <= 0:
        return 0.0
    bias = (buy - sell) / total
    label_bonus = {
        "STRONG_BUY": 1.0,
        "BUY": 0.7,
        "NEUTRAL": 0.0,
        "SELL": -0.7,
        "STRONG_SELL": -1.0,
    }.get(label.upper(), 0.0)
    return _clamp((bias + label_bonus) * 5.0, -5.0, 5.0)


def _fetch_tradingview(symbol: str, universe: str, config: dict[str, Any]) -> dict[str, Any]:
    from tradingview_ta import TA_Handler, Interval  # lazy optional dependency

    tv_cfg = _resolve_tradingview_config(universe, config)
    screener = str(tv_cfg.get("screener", "america"))
    exchange = str(tv_cfg.get("exchange", "")).strip()
    tv_symbol = str(tv_cfg.get("symbol_override") or symbol).strip()
    if exchange.upper() == "BIST" and tv_symbol.upper().endswith(".IS"):
        tv_symbol = tv_symbol[:-3]
    interval_name = str(tv_cfg.get("interval", "1d")).lower()
    interval_map = {
        "1m": Interval.INTERVAL_1_MINUTE,
        "5m": Interval.INTERVAL_5_MINUTES,
        "15m": Interval.INTERVAL_15_MINUTES,
        "30m": Interval.INTERVAL_30_MINUTES,
        "1h": Interval.INTERVAL_1_HOUR,
        "2h": Interval.INTERVAL_2_HOURS,
        "4h": Interval.INTERVAL_4_HOURS,
        "1d": Interval.INTERVAL_1_DAY,
        "1wk": Interval.INTERVAL_1_WEEK,
        "1mo": Interval.INTERVAL_1_MONTH,
    }
    interval = interval_map.get(interval_name, Interval.INTERVAL_1_DAY)

    handler = TA_Handler(
        symbol=tv_symbol,
        screener=screener,
        exchange=exchange or "NASDAQ",
        interval=interval,
        timeout=int(tv_cfg.get("timeout", 10)),
    )
    analysis = handler.get_analysis()
    summary = dict(getattr(analysis, "summary", {}) or {})
    oscillators = dict(getattr(analysis, "oscillators", {}) or {})
    moving_averages = dict(getattr(analysis, "moving_averages", {}) or {})

    tv_recommendation = _safe_text(summary.get("RECOMMENDATION")).upper()
    tv_oscillator_recommendation = _safe_text(oscillators.get("RECOMMENDATION")).upper()
    tv_moving_average_recommendation = _safe_text(moving_averages.get("RECOMMENDATION")).upper()
    tv_score = _score_tv_recommendation(
        tv_recommendation,
        int(_safe_float(summary.get("BUY"), default=0.0)),
        int(_safe_float(summary.get("NEUTRAL"), default=0.0)),
        int(_safe_float(summary.get("SELL"), default=0.0)),
    )

    return {
        "source": "tradingview_ta",
        "tv_technical_recommendation": tv_recommendation,
        "tv_oscillator_recommendation": tv_oscillator_recommendation,
        "tv_moving_average_recommendation": tv_moving_average_recommendation,
        "tv_buy_count": int(_safe_float(summary.get("BUY"), default=0.0)),
        "tv_neutral_count": int(_safe_float(summary.get("NEUTRAL"), default=0.0)),
        "tv_sell_count": int(_safe_float(summary.get("SELL"), default=0.0)),
        "tv_technical_recommendation_score": round(tv_score, 2),
    }


def load_market_data_enrichment(
    *,
    symbol: str,
    interval: str,
    universe: str,
    config: dict[str, Any] | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    cfg = (config or {}).get("market_data_enrichment", {})
    if not bool(cfg.get("enabled", False)):
        return {}

    ttl_hours = int(cfg.get("cache_ttl_hours", 24))
    provider_flags = {
        "yfinance": bool(cfg.get("use_yfinance", True)),
        "tradingview": bool(cfg.get("use_tradingview_ta", False)),
    }

    merged: dict[str, Any] = {}
    cache_bits: list[dict[str, Any]] = []

    if provider_flags["yfinance"]:
        cache_path = _cache_path(symbol, interval, "yfinance")
        cached = None if force_refresh else _load_cached(cache_path, ttl_hours)
        if cached is None:
            try:
                cached = _fetch_yfinance(symbol)
                _write_cache(cache_path, cached)
            except Exception:
                cached = _load_cached(cache_path, ttl_hours)
        if isinstance(cached, dict):
            merged.update(cached)
            cache_bits.append(cached)

    if provider_flags["tradingview"]:
        cache_path = _cache_path(symbol, interval, "tradingview")
        cached = None if force_refresh else _load_cached(cache_path, ttl_hours)
        if cached is None:
            try:
                cached = _fetch_tradingview(symbol, universe, cfg)
                _write_cache(cache_path, cached)
            except Exception:
                cached = _load_cached(cache_path, ttl_hours)
        if isinstance(cached, dict):
            merged.update(cached)
            cache_bits.append(cached)

    if not merged:
        return {}

    merged["market_data_enrichment_providers"] = ",".join(bit.get("source", "") for bit in cache_bits if bit.get("source"))
    merged["market_data_enrichment_fetched"] = True
    return merged
