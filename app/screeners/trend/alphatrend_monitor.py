from __future__ import annotations

import pandas as pd


def build_monitor_row(symbol: str, universe: str, interval: str, last_row: pd.Series) -> dict:
    return {
        "symbol": symbol,
        "universe": universe,
        "interval": interval,
        "date": last_row.get("date"),
        "close": float(last_row.get("close")) if pd.notna(last_row.get("close")) else None,
        "AT": float(last_row.get("AT")) if pd.notna(last_row.get("AT")) else None,
        "AT_lag2": float(last_row.get("AT_lag2")) if pd.notna(last_row.get("AT_lag2")) else None,
        "direction": int(last_row.get("alphatrend_direction")) if pd.notna(last_row.get("alphatrend_direction")) else None,
        "direction_label": last_row.get("alphatrend_direction_label"),
        "bars_since_direction_change": float(last_row.get("alphatrend_bars_since_direction_change")) if pd.notna(last_row.get("alphatrend_bars_since_direction_change")) else None,
        "last_valid_signal": last_row.get("alphatrend_last_valid_signal"),
        "bars_since_valid_signal": float(last_row.get("alphatrend_bars_since_valid_signal")) if pd.notna(last_row.get("alphatrend_bars_since_valid_signal")) else None,
        "compare_state_label": last_row.get("alphatrend_compare_state_label"),
        "flip_date": last_row.get("alphatrend_flip_date"),
        "flip_close": float(last_row.get("alphatrend_flip_close")) if pd.notna(last_row.get("alphatrend_flip_close")) else None,
        "close_to_flip_pct": float(last_row.get("close_to_flip_pct")) if pd.notna(last_row.get("close_to_flip_pct")) else None,
        "move_since_flip_pct": float(last_row.get("move_since_flip_pct")) if pd.notna(last_row.get("move_since_flip_pct")) else None,
    }
