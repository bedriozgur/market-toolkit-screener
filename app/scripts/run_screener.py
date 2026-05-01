#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = PROJECT_ROOT / "app"
UNIVERSES_DIR = APP_ROOT / "universes"

from workspace_config import get_workspace_root

from runtime_bootstrap import bootstrap_python

bootstrap_python()

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.screener.engine import run_screeners

RUNNER_VERSION = "3.4.1"
WORKSPACE_ROOT = get_workspace_root()
TELEGRAM_LIMIT = 3800


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OHLCV screeners.")
    parser.add_argument("--data-root", default=str(WORKSPACE_ROOT / "data"), help="Root data directory.")
    parser.add_argument("--universe", required=True, help="Universe name.")
    parser.add_argument("--intervals", required=True, help="Comma-separated intervals, e.g. 1d or 1d,1wk.")
    parser.add_argument(
        "--screeners",
        default="trend_leader,pullback,breakout_candidate,fresh_flip,alphatrend,technical_rank",
        help="Comma-separated screener names.",
    )
    parser.add_argument("--config", required=True, help="Config JSON path.")
    parser.add_argument("--top-n", type=int, default=30, help="Top N symbols.")
    parser.add_argument("--output-dir", default=str(WORKSPACE_ROOT / "outputs" / "screeners"), help="Output folder.")
    parser.add_argument("--summary-json", action="store_true", help="Write summary JSON.")
    telegram_group = parser.add_mutually_exclusive_group()
    telegram_group.add_argument("--telegram", dest="telegram", action="store_true", help="Send the top ranking to Telegram.")
    telegram_group.add_argument("--no-telegram", dest="telegram", action="store_false", help="Disable Telegram notifications.")
    parser.set_defaults(telegram=None)
    parser.add_argument("--telegram-bot-token", default=os.getenv("TELEGRAM_BOT_TOKEN", ""), help="Telegram bot token.")
    parser.add_argument("--telegram-chat-id", default=os.getenv("TELEGRAM_CHAT_ID", ""), help="Telegram chat id.")
    parser.add_argument(
        "--enrichment-refresh",
        action="store_true",
        help="Bypass cached Yahoo/TradingView enrichment and fetch fresh values.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_universe_symbols(universe: str) -> list[str]:
    universe_path = UNIVERSES_DIR / f"{universe}.txt"
    if not universe_path.exists():
        raise FileNotFoundError(f"Universe file not found: {universe_path}")
    with open(universe_path, "r", encoding="utf-8") as f:
        symbols = [line.strip() for line in f if line.strip()]
    if not symbols:
        raise ValueError(f"Universe file is empty: {universe_path}")
    return symbols


def load_universe_metadata(universe: str) -> dict[str, dict]:
    metadata_path = UNIVERSES_DIR / f"{universe}_metadata.csv"
    if not metadata_path.exists():
        logging.info("Universe metadata file not found: %s", metadata_path)
        return {}

    df = pd.read_csv(metadata_path)
    if "ticker" not in df.columns:
        logging.warning("Universe metadata file missing ticker column: %s", metadata_path)
        return {}

    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        symbol = str(row.get("ticker", "")).strip()
        if not symbol:
            continue
        peer_group = ""
        for field in ("sector", "industry", "country", "exchange"):
            value = row.get(field)
            if pd.notna(value) and str(value).strip():
                peer_group = str(value).strip()
                break
        if not peer_group:
            peer_group = universe
        metadata = row.to_dict()
        metadata.pop("ticker", None)
        metadata["peer_group"] = peer_group
        out[symbol] = {
            "peer_group": peer_group,
            "sector": row.get("sector"),
            "industry": row.get("industry"),
            "country": row.get("country"),
            "exchange": row.get("exchange"),
            **metadata,
        }
    return out


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip().replace("\ufeff", "") for col in out.columns]
    lower_map = {col: col.strip().lower().replace(" ", "_") for col in out.columns}
    out = out.rename(columns=lower_map)
    alias_map = {
        "datetime": "date",
        "timestamp": "date",
        "time": "date",
        "adjclose": "adj_close",
        "adj_close": "adj_close",
        "vol": "volume",
    }
    out = out.rename(columns=alias_map)
    return out


def parse_csv_or_parquet(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension: {path}")

    df = _normalize_columns(df)
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in file {path}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    return df


def try_load_symbol_file(data_root: Path, universe: str, symbol: str, interval: str) -> pd.DataFrame | None:
    base_dir = data_root / universe / "ohlcv"
    csv_path = base_dir / f"{symbol}_{interval}.csv"
    parquet_path = base_dir / f"{symbol}_{interval}.parquet"

    if csv_path.exists():
        return parse_csv_or_parquet(csv_path)
    if parquet_path.exists():
        return parse_csv_or_parquet(parquet_path)

    logging.warning("Price file not found for symbol=%s interval=%s", symbol, interval)
    return None


def load_data_map(data_root: Path, universe: str, symbols: Iterable[str], interval: str) -> tuple[dict[str, pd.DataFrame], int]:
    data_map: dict[str, pd.DataFrame] = {}
    load_failed_or_missing = 0

    for symbol in symbols:
        try:
            df = try_load_symbol_file(data_root, universe, symbol, interval)
            if df is not None and not df.empty:
                data_map[symbol] = df
            else:
                load_failed_or_missing += 1
        except Exception as exc:
            logging.warning("Failed to load symbol=%s interval=%s error=%s", symbol, interval, exc)
            load_failed_or_missing += 1

    return data_map, load_failed_or_missing


def resolve_benchmark_symbol_and_source(universe: str, config: dict) -> tuple[str | None, str | None]:
    technical_cfg = config.get("technical_rank", {})
    benchmark_symbol_by_universe = technical_cfg.get("benchmark_symbol_by_universe", {})
    benchmark_source_by_universe = technical_cfg.get("benchmark_source_by_universe", {})
    benchmark_symbol = benchmark_symbol_by_universe.get(universe)
    benchmark_source = benchmark_source_by_universe.get(universe, universe) if benchmark_symbol else None
    return benchmark_symbol, benchmark_source


def load_benchmark_close_series(data_root: Path, source_universe: str | None, benchmark_symbol: str, interval: str) -> pd.Series | None:
    base_dirs: list[Path] = []
    if source_universe:
        base_dirs.append(data_root / source_universe / "ohlcv")
    base_dirs.append(data_root / benchmark_symbol / "ohlcv")
    base_dirs.append(data_root / "benchmarks" / "ohlcv")

    seen: set[Path] = set()
    for base_dir in base_dirs:
        if base_dir in seen:
            continue
        seen.add(base_dir)
        csv_path = base_dir / f"{benchmark_symbol}_{interval}.csv"
        parquet_path = base_dir / f"{benchmark_symbol}_{interval}.parquet"
        if csv_path.exists():
            return parse_csv_or_parquet(csv_path).set_index("date")["close"].sort_index()
        if parquet_path.exists():
            return parse_csv_or_parquet(parquet_path).set_index("date")["close"].sort_index()

    logging.warning("Benchmark file not found for benchmark=%s interval=%s", benchmark_symbol, interval)
    return None


def split_for_telegram(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    chunks: list[str] = []
    remaining = text.strip()
    if not remaining:
        return [""]

    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, limit)
        if split_at <= 0:
            split_at = limit
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip("\n")
    return chunks


def send_telegram(bot_token: str, chat_id: str, text: str) -> None:
    if not bot_token:
        raise ValueError("Missing Telegram bot token.")
    if not chat_id:
        raise ValueError("Missing Telegram chat id.")
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    body = urlencode(payload).encode("utf-8")
    req = Request(url, data=body, method="POST")
    with urlopen(req, timeout=30) as resp:
        _ = resp.read()


def send_telegram_chunks(bot_token: str, chat_id: str, text: str) -> None:
    chunks = split_for_telegram(text)
    for chunk in chunks:
        if chunk.strip():
            send_telegram(bot_token, chat_id, chunk)


def build_telegram_message(summary: dict, universe: str, interval: str, top_rows: list[dict[str, object]]) -> str:
    lines = [
        f"Universe: {universe}",
        f"Interval: {interval}",
        f"Top 10: {', '.join(summary.get('universe_rank_top10', []))}" if summary.get("universe_rank_top10") else "Top 10: none",
        "",
        "Ranked results:",
    ]
    for idx, row in enumerate(top_rows, start=1):
        symbol = str(row.get("symbol", ""))
        score = row.get("score", "")
        tech = row.get("technical_score", "")
        fresh = row.get("fresh_flip_score", "")
        external = row.get("external_score", "")
        flip = str(row.get("fresh_flip_signals", "")).strip()
        suffix = f" | {flip}" if flip else ""
        lines.append(f"{idx}. {symbol} score={score} tech={tech} flip={fresh} ext={external}{suffix}")
    lines.append("")
    lines.append(f"Loaded: {summary.get('symbols_loaded', 0)} | Liquidity: {summary.get('passed_liquidity', 0)}")
    lines.append(f"Source rank rows: {summary.get('universe_rank_rows', 0)}")
    return "\n".join(lines)


def main() -> None:
    configure_logging()
    args = parse_args()

    config = load_json(args.config)
    symbols = load_universe_symbols(args.universe)
    symbol_metadata = load_universe_metadata(args.universe)
    intervals = [item.strip() for item in args.intervals.split(",") if item.strip()]
    screener_names = [item.strip() for item in args.screeners.split(",") if item.strip()]
    benchmark_symbol, benchmark_source = resolve_benchmark_symbol_and_source(args.universe, config)

    logging.info("Screener runner version: %s", RUNNER_VERSION)
    logging.info("Universe: %s", args.universe)
    logging.info("Symbols in universe: %d", len(symbols))
    logging.info("Intervals: %s", ", ".join(intervals))
    logging.info("Screeners: %s", ", ".join(screener_names))
    logging.info("Config: %s", args.config)
    logging.info("Base data directory: %s", args.data_root)
    logging.info("Output directory: %s", args.output_dir)
    if benchmark_symbol:
        logging.info("Benchmark: %s (source folder: %s)", benchmark_symbol, benchmark_source)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    for interval in intervals:
        logging.info("=== Interval start: %s ===", interval)
        benchmark_close = None
        if benchmark_symbol:
            benchmark_close = load_benchmark_close_series(
                data_root=data_root,
                source_universe=benchmark_source,
                benchmark_symbol=benchmark_symbol,
                interval=interval,
            )
        data_map, load_failed_or_missing = load_data_map(
            data_root=data_root,
            universe=args.universe,
            symbols=symbols,
            interval=interval,
        )

        results = run_screeners(
            data_map=data_map,
            universe=args.universe,
            interval=interval,
            screener_names=screener_names,
            config=config,
            output_dir=output_dir,
            top_n=args.top_n,
            write_summary_json=args.summary_json,
            symbols_requested=len(symbols),
            symbols_load_failed_or_missing=load_failed_or_missing,
            benchmark_close=benchmark_close,
            symbol_metadata=symbol_metadata,
            enrichment_refresh=args.enrichment_refresh,
        )

        summary = results["summary"]
        top_rows = summary.get("universe_rank_top10_rows", []) or []

        for screener_name, eligible_count in summary["eligible_counts"].items():
            logging.info("%s eligible: %d", screener_name, eligible_count)
            top_symbols = summary["top_symbols"].get(screener_name, [])
            if top_symbols:
                logging.info("Top %s: %s", screener_name, ", ".join(top_symbols))

        logging.info("Symbols requested: %d", summary["symbols_requested"])
        logging.info("Symbols loaded: %d", summary["symbols_loaded"])
        logging.info("Symbols load-failed or missing: %d", summary["symbols_load_failed_or_missing"])
        logging.info("Symbols feature-ready: %d", summary["symbols_feature_ready"])
        logging.info("Symbols passed liquidity filter: %d", summary["passed_liquidity"])

        truth_rows = summary.get("alphatrend_truth_rows")
        state_rows = summary.get("alphatrend_state_rows")
        if truth_rows is not None:
            logging.info("AlphaTrend truth rows: %d", truth_rows)
        elif state_rows is not None:
            logging.info("AlphaTrend state rows: %d", state_rows)

        telegram_enabled = args.telegram if args.telegram is not None else bool(args.telegram_bot_token and args.telegram_chat_id)
        if telegram_enabled and args.telegram_bot_token and args.telegram_chat_id:
            message = build_telegram_message(summary, args.universe, interval, top_rows)
            send_telegram_chunks(args.telegram_bot_token, args.telegram_chat_id, message)
            logging.info("Telegram notification sent for %s %s.", args.universe, interval)
        elif telegram_enabled and not (args.telegram_bot_token and args.telegram_chat_id):
            logging.warning("Telegram enabled but TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set.")

        logging.info("=== Interval end: %s ===", interval)

    logging.info("Screener run finished successfully")


if __name__ == "__main__":
    main()
