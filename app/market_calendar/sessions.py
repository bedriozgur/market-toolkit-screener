from dataclasses import dataclass
from datetime import time

@dataclass(frozen=True)
class MarketSession:
    market: str
    tz: str
    regular_open: time
    regular_close: time
    halfday_close: time

BIST_SESSION = MarketSession("bist", "Europe/Istanbul", time(10,0), time(18,0), time(13,0))
