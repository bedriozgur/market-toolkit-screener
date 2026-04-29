from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def bootstrap_python() -> None:
    """
    Re-exec into the project virtualenv when the current interpreter lacks pandas.

    This keeps the same script command usable both locally and from automation,
    while still allowing an explicit interpreter override via the shell or CLI.
    """
    if os.environ.get("MARKET_TOOLKIT_BOOTSTRAPPED") == "1":
        return
    if sys.prefix != sys.base_prefix:
        return
    if importlib.util.find_spec("pandas") is not None:
        return

    project_root = Path(__file__).resolve().parents[2]
    candidates = []

    venv_env = os.environ.get("VIRTUAL_ENV", "").strip()
    if venv_env:
        candidates.append(Path(venv_env) / "bin" / "python")

    candidates.append(project_root / ".venv" / "bin" / "python")

    for candidate in candidates:
        try:
            if not candidate.exists():
                continue
            os.environ["MARKET_TOOLKIT_BOOTSTRAPPED"] = "1"
            os.execv(str(candidate), [str(candidate), *sys.argv])
        except Exception:
            continue
