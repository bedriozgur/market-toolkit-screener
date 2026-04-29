from datetime import date
from market_calendar.sessions import BIST_SESSION

_BIST_FULL_HOLIDAYS_2026 = {date(2026,1,1), date(2026,3,20)}
_BIST_HALF_DAYS_2026 = {date(2026,3,19)}

def get_bist_timezone() -> str:
    return BIST_SESSION.tz

def is_bist_full_holiday(d: date) -> bool:
    return d in _BIST_FULL_HOLIDAYS_2026

def is_bist_half_day(d: date) -> bool:
    return d in _BIST_HALF_DAYS_2026

def get_bist_session_close(d: date):
    return BIST_SESSION.halfday_close if is_bist_half_day(d) else BIST_SESSION.regular_close
