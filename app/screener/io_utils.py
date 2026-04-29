from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_config_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_universe_symbols(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")

    symbols: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        symbols.append(item)
    return symbols


def read_ohlcv_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in file: {path}")

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).reset_index(drop=True)
    return out


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
