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


def _build_trending_frame(rows: int = 260) -> tuple[pd.DataFrame, pd.Series]:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    close = pd.Series([100.0 + i * 0.35 for i in range(rows)])
    benchmark_close = pd.Series([100.0 + i * 0.2 for i in range(rows)], index=dates)
    open_ = close * 0.995
    high = close * 1.01
    low = close * 0.99
    volume = pd.Series([200000 + i * 1000 for i in range(rows)], dtype="float64")
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    ), benchmark_close


class TechnicalRankTests(unittest.TestCase):
    def test_evaluate_technical_rank_returns_eligible_for_strong_trend(self) -> None:
        from scripts.screener.feature_builder import build_features
        from scripts.screener.screeners.technical_rank import evaluate

        raw_df, benchmark_close = _build_trending_frame()
        df = build_features(raw_df, benchmark_close=benchmark_close)
        last_row = df.iloc[-1]

        result = evaluate(last_row, {"technical_rank": {"min_score": 60, "require_trend_alignment": True}})

        self.assertTrue(result.eligible)
        self.assertGreaterEqual(result.score, 60)
        self.assertEqual(result.screener_name, "technical_rank")
        self.assertIn("trend_score", result.details)
        self.assertIn("flags", result.details)

    def test_run_screeners_writes_technical_rank_output(self) -> None:
        from scripts.screener.engine import run_screeners

        df, benchmark_close = _build_trending_frame()
        output_dir = Path(self._testMethodName)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = run_screeners(
                data_map={"TEST": df},
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
            self.assertEqual(summary["eligible_counts"]["technical_rank"], 1)
            self.assertEqual(summary["top_symbols"]["technical_rank"], ["TEST"])
        finally:
            for path in output_dir.glob("*"):
                path.unlink()
            output_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
