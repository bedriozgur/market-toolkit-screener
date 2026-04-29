from __future__ import annotations

import pandas as pd


def rank_descending(df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(score_col, ascending=False).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out
