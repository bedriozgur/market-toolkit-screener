from __future__ import annotations

import sys
from pathlib import Path

REAL_INDICATORS_ROOT = Path(__file__).resolve().parents[2] / "indicators"
REAL_APP_ROOT = REAL_INDICATORS_ROOT.parent

if str(REAL_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(REAL_APP_ROOT))

if str(REAL_INDICATORS_ROOT) not in __path__:
    __path__.append(str(REAL_INDICATORS_ROOT))
