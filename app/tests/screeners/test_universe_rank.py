from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = PROJECT_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
if str(APP_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(APP_ROOT / "scripts"))


def _build_frame(rows: int = 260) -> tuple[pd.DataFrame, pd.Series]:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    close = pd.Series([100.0 + i * 0.35 for i in range(rows)])
    benchmark_close = pd.Series([100.0 + i * 0.2 for i in range(rows)], index=dates)
    open_ = close * 0.995
    high = close * 1.01
    low = close * 0.99
    volume = pd.Series([200000 + i * 1000 for i in range(rows)], dtype="float64")
    return (
        pd.DataFrame(
            {
                "date": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        ),
        benchmark_close,
    )


class UniverseRankTests(unittest.TestCase):
    def test_build_telegram_message_uses_top_rows(self) -> None:
        from scripts.run_screener import build_telegram_message

        message = build_telegram_message(
            summary={
                "universe_rank_top10": ["AAA", "BBB"],
                "symbols_loaded": 2,
                "passed_liquidity": 2,
                "universe_rank_rows": 2,
            },
            universe="bist100",
            interval="1d",
            top_rows=[
                {
                    "symbol": "AAA",
                    "score": 91.2,
                    "technical_score": 70.0,
                    "fresh_flip_score": 12.0,
                    "external_score": 4.0,
                    "fresh_flip_signals": "alphatrend:BUY:1",
                }
            ],
        )

        self.assertIn("Universe: bist100", message)
        self.assertIn("AAA score=91.2", message)
        self.assertIn("alphatrend:BUY:1", message)

    def test_run_screeners_accepts_enrichment_refresh_flag(self) -> None:
        from scripts.screener.engine import run_screeners

        raw_df, benchmark_close = _build_frame()
        output_dir = Path(self._testMethodName)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = run_screeners(
                data_map={"TEST": raw_df},
                universe="bist100",
                interval="1d",
                screener_names=["technical_rank"],
                config={
                    "liquidity": {
                        "min_price": 1.0,
                        "min_avg_volume_20": 1000.0,
                        "min_avg_dollar_volume_20": 1000.0,
                        "min_history_bars": 50,
                    },
                    "technical_rank": {"min_score": 60, "require_trend_alignment": True},
                    "market_data_enrichment": {
                        "enabled": False,
                    },
                },
                output_dir=output_dir,
                top_n=10,
                write_summary_json=False,
                symbols_requested=1,
                symbols_load_failed_or_missing=0,
                benchmark_close=benchmark_close,
                enrichment_refresh=True,
            )

            self.assertTrue(result["summary"]["enrichment_refresh"])
        finally:
            for path in output_dir.glob("*"):
                path.unlink()
            output_dir.rmdir()

    def test_tradingview_profile_is_selected_from_universe(self) -> None:
        from scripts.screener.market_data_enrichment import _resolve_tradingview_config

        self.assertEqual(
            _resolve_tradingview_config("bist100", {"tradingview_ta": {}})["screener"],
            "turkey",
        )
        self.assertEqual(
            _resolve_tradingview_config("bist100", {"tradingview_ta": {}})["exchange"],
            "BIST",
        )
        self.assertEqual(
            _resolve_tradingview_config("sp500", {"tradingview_ta": {}})["screener"],
            "america",
        )
        self.assertEqual(
            _resolve_tradingview_config("sp500", {"tradingview_ta": {}})["exchange"],
            "NASDAQ",
        )

    def test_build_universe_rank_row_includes_fresh_flips_and_scores(self) -> None:
        from scripts.screener.universe_rank import build_universe_rank_row

        last_row = pd.Series(
            {
                "date": pd.Timestamp("2025-01-02"),
                "close": 123.45,
                "ema_20": 120.0,
                "ema_50": 115.0,
                "ema_200": 100.0,
                "ema_20_slope_5": 4.0,
                "roc_5": 3.5,
                "roc_10": 7.2,
                "roc_20": 10.4,
                "roc_63": 18.0,
                "rsi_14": 62.0,
                "mfi_14": 58.0,
                "macd_hist": 1.4,
                "macd_hist_delta": 0.3,
                "bb_percent_b": 0.82,
                "dist_to_52w_high_pct": 2.0,
                "close_position_in_20d_range": 0.88,
                "rel_volume_20": 1.7,
                "volume_ratio_10_50": 1.3,
                "obv_trend_20": 4.0,
                "rs_line_slope_5": 3.0,
                "rs_line_slope_20": 5.0,
                "rs_line_dist_to_52w_high_pct": 1.5,
                "roc_63_rank_pct": 0.86,
                "beta_adjusted_return_rank_pct": 0.77,
                "peer_group_rank_pct": 0.9,
                "alphatrend_compare_state_label": "BUY",
                "alphatrend_last_valid_signal": "BUY",
                "alphatrend_bars_since_valid_signal": 1,
                "supertrend_bullish": True,
                "supertrend_state_label": "BUY",
                "supertrend_flip_age_bars": 1,
                "flip_age_bars_ema20": 1,
                "current_state_ema20": 1,
                "market_cap": 25_000_000_000,
                "revenue_growth_yoy": 18.0,
                "eps_growth_yoy": 22.0,
                "debt_to_equity": 0.4,
                "reddit_sentiment_score": 0.78,
                "twitter_sentiment_score": 0.64,
                "yf_analyst_recommendation_score": 4.0,
                "yf_analyst_price_target_upside_score": 2.0,
                "yf_news_sentiment_score": 1.5,
                "yf_upgrades_downgrades_score": 1.0,
                "yf_earnings_growth_estimate": 16.0,
                "yf_revenue_growth_estimate": 12.0,
                "tv_technical_recommendation_score": 3.0,
                "tv_technical_recommendation": "BUY",
                "tv_oscillator_recommendation": "BUY",
                "tv_moving_average_recommendation": "BUY",
            }
        )

        row = build_universe_rank_row("TEST", "unit_test", "1d", last_row)

        self.assertGreater(row["score"], 0)
        self.assertGreaterEqual(row["fresh_flip_count"], 3)
        self.assertGreater(row["external_score"], 0)
        self.assertIn("alphatrend", row["fresh_flip_signals"])
        self.assertIn("supertrend", row["fresh_flip_signals"])
        self.assertIn("ema20", row["fresh_flip_signals"])
        self.assertEqual(row["tv_technical_recommendation"], "BUY")

    def test_run_screeners_writes_universe_rank_and_top10(self) -> None:
        from scripts.screener.engine import run_screeners

        raw_df, benchmark_close = _build_frame()

        output_dir = Path(self._testMethodName)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = run_screeners(
                data_map={"TEST": raw_df},
                universe="unit_test",
                interval="1d",
                screener_names=["technical_rank"],
                config={
                    "liquidity": {
                        "min_price": 1.0,
                        "min_avg_volume_20": 1000.0,
                        "min_avg_dollar_volume_20": 1000.0,
                        "min_history_bars": 50,
                    },
                    "technical_rank": {"min_score": 60, "require_trend_alignment": True},
                },
                output_dir=output_dir,
                top_n=10,
                write_summary_json=False,
                symbols_requested=1,
                symbols_load_failed_or_missing=0,
                benchmark_close=benchmark_close,
            )

            summary = result["summary"]
            self.assertEqual(summary["universe_rank_top10"], ["TEST"])
            self.assertEqual(summary["universe_rank_rows"], 1)

            top10_files = sorted(output_dir.glob("*_top10_*.csv"))
            universe_rank_files = sorted(output_dir.glob("*_universe_rank_*.csv"))
            self.assertEqual(len(top10_files), 1)
            self.assertEqual(len(universe_rank_files), 1)

            top10_df = pd.read_csv(top10_files[0])
            self.assertEqual(top10_df.loc[0, "symbol"], "TEST")
            self.assertGreater(top10_df.loc[0, "score"], 0)
        finally:
            for path in output_dir.glob("*"):
                path.unlink()
            output_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
