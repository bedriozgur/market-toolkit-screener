from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_workspace_root() -> Path:
    env_value = os.environ.get("MARKET_TOOLKIT_WORKSPACE", "").strip()
    if env_value:
        return Path(env_value).expanduser()

    local_workspace = PROJECT_ROOT / "workspace"
    if local_workspace.exists():
        return local_workspace

    sibling_workspace = PROJECT_ROOT.parent / "market-toolkit-workspace"
    if sibling_workspace.exists():
        return sibling_workspace

    return local_workspace
