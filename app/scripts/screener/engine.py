from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.screener.alphatrend import evaluate_alphatrend
from scripts.screener.alphatrend_truth import build_alphatrend_truth_row
from scripts.screener.feature_builder import build_features
from scripts.screener.filters import passes_liquidity_filter
from scripts.screener.fresh_flip import evaluate_fresh_flip
from scripts.screener.pullback import evaluate_pullback
from scripts.screener.trend_leader import evaluate_trend_leader
from scripts.screener.breakout_candidate import evaluate_breakout_candidate
from scripts.screener.market_data_enrichment import load_market_data_enrichment
from scripts.screener.technical_rank import evaluate_technical_rank
from scripts.screener.universe_rank import build_universe_rank_row


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")


def rank_descending(df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def _make_base_row(symbol: str, universe: str, interval: str, last_row: pd.Series) -> dict:
    return {
        "symbol": symbol,
        "universe": universe,
        "interval": interval,
        "peer_group": last_row.get("peer_group"),
        "date": last_row["date"],
        "open": last_row["open"],
        "high": last_row["high"],
        "low": last_row["low"],
        "close": last_row["close"],
        "volume": last_row["volume"],
        "avg_volume_20": last_row["avg_volume_20"],
        "avg_dollar_volume_20": last_row["avg_dollar_volume_20"],
        "rel_volume_20": last_row["rel_volume_20"],
        "ema_20": last_row["ema_20"],
        "ema_50": last_row["ema_50"],
        "ema_200": last_row["ema_200"],
        "ema_20_slope_5": last_row.get("ema_20_slope_5"),
        "roc_5": last_row["roc_5"],
        "roc_10": last_row["roc_10"],
        "roc_20": last_row["roc_20"],
        "rsi_14": last_row["rsi_14"],
        "mfi_14": last_row.get("mfi_14"),
        "adx_14": last_row.get("adx_14"),
        "aroon_up_25": last_row.get("aroon_up_25"),
        "aroon_down_25": last_row.get("aroon_down_25"),
        "aroon_osc_25": last_row.get("aroon_osc_25"),
        "supertrend_line": last_row.get("supertrend_line"),
        "supertrend_trend": last_row.get("supertrend_trend"),
        "supertrend_bullish": last_row.get("supertrend_bullish"),
        "supertrend_flip_age_bars": last_row.get("supertrend_flip_age_bars"),
        "supertrend_state_label": last_row.get("supertrend_state_label"),
        "ichimoku_price_above_cloud": last_row.get("ichimoku_price_above_cloud"),
        "ichimoku_kumo_green": last_row.get("ichimoku_kumo_green"),
        "ichimoku_tenkan_above_kijun": last_row.get("ichimoku_tenkan_above_kijun"),
        "ichimoku_chikou_above_price_26": last_row.get("ichimoku_chikou_above_price_26"),
        "obv_trend_20": last_row.get("obv_trend_20"),
        "macd_hist": last_row.get("macd_hist"),
        "macd_hist_delta": last_row.get("macd_hist_delta"),
        "atr_pct_14": last_row["atr_pct_14"],
        "bb_percent_b": last_row.get("bb_percent_b"),
        "roc_63": last_row.get("roc_63"),
        "rolling_high_20": last_row["rolling_high_20"],
        "rolling_high_50": last_row["rolling_high_50"],
        "rolling_high_252": last_row.get("rolling_high_252"),
        "dist_to_20d_high_pct": last_row["dist_to_20d_high_pct"],
        "dist_to_50d_high_pct": last_row["dist_to_50d_high_pct"],
        "dist_to_52w_high_pct": last_row.get("dist_to_52w_high_pct"),
        "close_position_in_20d_range": last_row["close_position_in_20d_range"],
        "volume_ratio_10_50": last_row.get("volume_ratio_10_50"),
        "rs_line": last_row.get("rs_line"),
        "rs_line_slope_5": last_row.get("rs_line_slope_5"),
        "rs_line_slope_20": last_row.get("rs_line_slope_20"),
        "rs_line_dist_to_52w_high_pct": last_row.get("rs_line_dist_to_52w_high_pct"),
        "beta_adjusted_return_63": last_row.get("beta_adjusted_return_63"),
        "roc_63_rank_pct": last_row.get("roc_63_rank_pct"),
        "beta_adjusted_return_rank_pct": last_row.get("beta_adjusted_return_rank_pct"),
        "peer_group_rank_pct": last_row.get("peer_group_rank_pct"),
        "alphatrend": last_row.get("alphatrend"),
        "alphatrend_lag2": last_row.get("alphatrend_lag2"),
        "alphatrend_compare_state_label": last_row.get("alphatrend_compare_state_label"),
        "alphatrend_buy_signal_raw": last_row.get("alphatrend_buy_signal_raw"),
        "alphatrend_sell_signal_raw": last_row.get("alphatrend_sell_signal_raw"),
        "alphatrend_buy_label": last_row.get("alphatrend_buy_label"),
        "alphatrend_sell_label": last_row.get("alphatrend_sell_label"),
        "alphatrend_last_valid_signal": last_row.get("alphatrend_last_valid_signal"),
        "alphatrend_last_valid_signal_date": last_row.get("alphatrend_last_valid_signal_date"),
        "alphatrend_bars_since_valid_signal": last_row.get("alphatrend_bars_since_valid_signal"),
        "alphatrend_flip_close": last_row.get("alphatrend_flip_close"),
        "close_to_flip_pct": last_row.get("close_to_flip_pct"),
        "move_since_flip_pct": last_row.get("move_since_flip_pct"),
    }


def _normalize_result(result: dict | object, screener_name: str) -> dict:
    if isinstance(result, dict):
        out = dict(result)
    else:
        out = {
            "screener_name": getattr(result, "screener_name", screener_name),
            "eligible": getattr(result, "eligible", False),
            "score": float(getattr(result, "score", 0.0)),
            "reason_1": getattr(result, "reason_1", ""),
            "reason_2": getattr(result, "reason_2", ""),
            "reason_3": getattr(result, "reason_3", ""),
        }
        details = getattr(result, "details", None)
        if isinstance(details, dict):
            out.update(details)

    out.setdefault("screener_name", screener_name)
    out.setdefault("eligible", False)
    out.setdefault("score", 0.0)
    out.setdefault("reason_1", "")
    out.setdefault("reason_2", "")
    out.setdefault("reason_3", "")
    return out


def _evaluate_screener(screener_name: str, last_row: pd.Series, config: dict) -> dict:
    if screener_name == "trend_leader":
        return _normalize_result(evaluate_trend_leader(last_row), screener_name)
    if screener_name == "pullback":
        return _normalize_result(evaluate_pullback(last_row), screener_name)
    if screener_name == "breakout_candidate":
        return _normalize_result(evaluate_breakout_candidate(last_row), screener_name)
    if screener_name == "fresh_flip":
        try:
            return _normalize_result(evaluate_fresh_flip(last_row, config), screener_name)
        except TypeError:
            return _normalize_result(evaluate_fresh_flip(last_row), screener_name)
    if screener_name == "alphatrend":
        return _normalize_result(evaluate_alphatrend(last_row, config), screener_name)
    if screener_name == "technical_rank":
        return _normalize_result(evaluate_technical_rank(last_row, config), screener_name)
    raise ValueError(f"Unsupported screener: {screener_name}")


def _write_summary_json(summary: dict, output_dir: Path, universe: str, timestamp: str) -> None:
    ensure_dir(output_dir)
    path = output_dir / f"{universe}_screener_run_{timestamp}_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    logging.info("Wrote summary JSON: %s", path)


def run_screeners(
    data_map: dict[str, pd.DataFrame],
    universe: str,
    interval: str,
    screener_names: list[str],
    config: dict,
    output_dir: Path,
    top_n: int,
    write_summary_json: bool,
    symbols_requested: int,
    symbols_load_failed_or_missing: int,
    benchmark_close: pd.Series | None = None,
    symbol_metadata: dict[str, dict] | None = None,
    enrichment_refresh: bool = False,
) -> dict:
    """Execute one screener batch for a universe/interval and write ranked outputs."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    liquidity_config = config["liquidity"]
    alphatrend_cfg = config.get("alphatrend", {})
    alphatrend_multiplier = float(alphatrend_cfg.get("multiplier", 1.0))
    alphatrend_common_period = int(alphatrend_cfg.get("common_period", 14))
    alphatrend_no_volume_data = bool(alphatrend_cfg.get("no_volume_data", False))
    alphatrend_fresh_signal_max_bars = int(alphatrend_cfg.get("fresh_signal_max_bars", 5))

    symbols_feature_ready = 0
    passed_liquidity = 0
    per_screener_rows: dict[str, list[dict]] = {name: [] for name in screener_names}
    combined_rows: list[dict] = []
    alphatrend_truth_rows: list[dict] = []
    pending_rows: list[dict] = []
    universe_rank_rows: list[dict] = []

    required_cols = ["date", "open", "high", "low", "close", "volume", "ema_20", "ema_50", "ema_200"]

    for symbol, raw_df in data_map.items():
        df = build_features(
            raw_df,
            alphatrend_multiplier=alphatrend_multiplier,
            alphatrend_common_period=alphatrend_common_period,
            alphatrend_no_volume_data=alphatrend_no_volume_data,
            benchmark_close=benchmark_close,
        )

        df = df.dropna(subset=required_cols).reset_index(drop=True)
        if df.empty:
            continue

        symbols_feature_ready += 1
        last_row = df.iloc[-1]

        if not passes_liquidity_filter(last_row, liquidity_config):
            continue

        passed_liquidity += 1
        row = last_row.copy()
        metadata = dict((symbol_metadata or {}).get(symbol, {}))
        if metadata:
            for key, value in metadata.items():
                row[key] = value
        row["peer_group"] = metadata.get("peer_group", universe)
        enrichment = load_market_data_enrichment(
            symbol=symbol,
            interval=interval,
            universe=universe,
            config=config,
            force_refresh=enrichment_refresh,
        )
        if enrichment:
            for key, value in enrichment.items():
                row[key] = value
        pending_rows.append({"symbol": symbol, "row": row})

    def _percentile_rank_desc(series: pd.Series) -> pd.Series:
        values = series.astype("float64")
        valid = values.dropna()
        if valid.empty:
            return pd.Series(np.nan, index=series.index, dtype="float64")
        if len(valid) == 1:
            out = pd.Series(np.nan, index=series.index, dtype="float64")
            out.loc[valid.index] = 1.0
            return out
        ranks = valid.rank(method="average", ascending=False)
        pct = 1.0 - ((ranks - 1.0) / (len(valid) - 1.0))
        out = pd.Series(np.nan, index=series.index, dtype="float64")
        out.loc[valid.index] = pct
        return out

    if pending_rows:
        cross_df = pd.DataFrame(
            {
                "symbol": [item["symbol"] for item in pending_rows],
                "peer_group": [item["row"].get("peer_group", universe) for item in pending_rows],
                "roc_63": [item["row"].get("roc_63") for item in pending_rows],
                "beta_adjusted_return_63": [item["row"].get("beta_adjusted_return_63") for item in pending_rows],
            }
        )
        cross_df["roc_63_rank_pct"] = _percentile_rank_desc(cross_df["roc_63"])
        cross_df["beta_adjusted_return_rank_pct"] = _percentile_rank_desc(cross_df["beta_adjusted_return_63"])
        cross_df["peer_group_rank_pct"] = np.nan
        for peer_group, group_index in cross_df.groupby("peer_group").groups.items():
            group = cross_df.loc[group_index, "beta_adjusted_return_63"]
            cross_df.loc[group_index, "peer_group_rank_pct"] = _percentile_rank_desc(group)

        rank_map = cross_df.set_index("symbol").to_dict(orient="index")

    for item in pending_rows:
        symbol = item["symbol"]
        last_row = item["row"].copy()
        rank_info = rank_map.get(symbol, {})
        for key in ("roc_63_rank_pct", "beta_adjusted_return_rank_pct", "peer_group_rank_pct"):
            last_row[key] = rank_info.get(key)

        universe_rank_rows.append(build_universe_rank_row(symbol=symbol, universe=universe, interval=interval, last_row=last_row))

        alphatrend_truth_rows.append(
            build_alphatrend_truth_row(
                symbol=symbol,
                universe=universe,
                interval=interval,
                last_row=last_row,
                fresh_signal_max_bars=alphatrend_fresh_signal_max_bars,
            )
        )

        matched_screeners = []
        best_score = 0.0
        for screener_name in screener_names:
            result = _evaluate_screener(screener_name, last_row, config)
            if result.get("eligible", False):
                row = _make_base_row(symbol, universe, interval, last_row)
                row.update(result)
                per_screener_rows[screener_name].append(row)
                matched_screeners.append(screener_name)
                best_score = max(best_score, float(result.get("score", 0.0)))

        if matched_screeners:
            combined_rows.append(
                {
                    "symbol": symbol,
                    "universe": universe,
                    "interval": interval,
                    "peer_group": last_row.get("peer_group"),
                    "date": last_row["date"],
                    "close": last_row["close"],
                    "hit_count": len(matched_screeners),
                    "best_score": round(best_score, 2),
                    "matched_screeners": ",".join(matched_screeners),
                    "matched_screeners_compact": " | ".join(matched_screeners),
                }
            )

    eligible_counts: dict[str, int] = {}
    top_symbols: dict[str, list[str]] = {}

    for screener_name, rows in per_screener_rows.items():
        df_out = pd.DataFrame(rows)
        if not df_out.empty:
            df_out = rank_descending(df_out, score_col="score")
            eligible_counts[screener_name] = len(df_out)
            top_symbols[screener_name] = df_out.head(top_n)["symbol"].tolist()
            csv_path = output_dir / f"{universe}_{interval}_{screener_name}_{timestamp}.csv"
            write_dataframe(df_out, csv_path)
            logging.info("Wrote CSV: %s", csv_path)
        else:
            eligible_counts[screener_name] = 0
            top_symbols[screener_name] = []

    combined_df = pd.DataFrame(combined_rows)
    if not combined_df.empty:
        combined_df = combined_df.sort_values(["hit_count", "best_score"], ascending=[False, False]).reset_index(drop=True)
        combined_df["rank"] = range(1, len(combined_df) + 1)
        combined_csv = output_dir / f"{universe}_{interval}_combined_{timestamp}.csv"
        write_dataframe(combined_df, combined_csv)
        logging.info("Wrote CSV: %s", combined_csv)

    universe_rank_df = pd.DataFrame(universe_rank_rows)
    if not universe_rank_df.empty:
        universe_rank_df = rank_descending(universe_rank_df, score_col="score")
        universe_rank_csv = output_dir / f"{universe}_{interval}_universe_rank_{timestamp}.csv"
        write_dataframe(universe_rank_df, universe_rank_csv)
        logging.info("Wrote CSV: %s", universe_rank_csv)

        top10_df = universe_rank_df.head(10).copy()
        top10_csv = output_dir / f"{universe}_{interval}_top10_{timestamp}.csv"
        write_dataframe(top10_df, top10_csv)
        logging.info("Wrote CSV: %s", top10_csv)

    alphatrend_truth_df = pd.DataFrame(alphatrend_truth_rows)
    if not alphatrend_truth_df.empty:
        alphatrend_truth_df = alphatrend_truth_df.sort_values(
            ["last_valid_signal", "is_fresh_signal", "priority_score", "symbol"],
            ascending=[True, False, False, True],
        ).reset_index(drop=True)
        alphatrend_truth_df["rank"] = range(1, len(alphatrend_truth_df) + 1)
        truth_csv = output_dir / f"{universe}_{interval}_alphatrend_truth_table_{timestamp}.csv"
        write_dataframe(alphatrend_truth_df, truth_csv)
        logging.info("Wrote CSV: %s", truth_csv)

    summary = {
        "universe": universe,
        "interval": interval,
        "symbols_requested": symbols_requested,
        "symbols_loaded": len(data_map),
        "symbols_load_failed_or_missing": symbols_load_failed_or_missing,
        "symbols_feature_ready": symbols_feature_ready,
        "passed_liquidity": passed_liquidity,
        "alphatrend_truth_rows": len(alphatrend_truth_rows),
        "eligible_counts": eligible_counts,
        "top_symbols": top_symbols,
        "universe_rank_rows": len(universe_rank_df),
        "universe_rank_top10": universe_rank_df.head(10)["symbol"].tolist() if not universe_rank_df.empty else [],
        "universe_rank_top10_rows": (
            universe_rank_df.head(10)[
                [
                    "symbol",
                    "score",
                    "technical_score",
                    "fresh_flip_score",
                    "fundamental_score",
                    "sentiment_score",
                    "external_score",
                    "fresh_flip_signals",
                ]
            ].to_dict(orient="records")
            if not universe_rank_df.empty
            else []
        ),
        "enrichment_refresh": enrichment_refresh,
    }

    if write_summary_json:
        _write_summary_json(summary=summary, output_dir=output_dir, universe=universe, timestamp=timestamp)

    return {"summary": summary}
