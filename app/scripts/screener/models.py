from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ScreenerResult:
    screener_name: str
    eligible: bool
    score: float
    reason_1: str
    reason_2: str
    reason_3: str
    details: Dict[str, Any]
