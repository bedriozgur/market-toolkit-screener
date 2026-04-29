from __future__ import annotations

import pandas as pd


def passes_liquidity_filter(last_row: pd.Series, config: dict) -> bool:
    return bool(
        last_row["close"] >= config["min_price"]
        and last_row["avg_volume_20"] >= config["min_avg_volume_20"]
        and last_row["avg_dollar_volume_20"] >= config["min_avg_dollar_volume_20"]
        and last_row["history_bars"] >= config["min_history_bars"]
    )
