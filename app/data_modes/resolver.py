from pathlib import Path
import pandas as pd
from aggregators.intraday_to_daily import build_live_daily_bar
from aggregators.daily_to_weekly import build_live_weekly_bar

def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "date": rename_map[c] = "date"
        elif cl == "open": rename_map[c] = "open"
        elif cl == "high": rename_map[c] = "high"
        elif cl == "low": rename_map[c] = "low"
        elif cl == "close": rename_map[c] = "close"
        elif cl == "volume": rename_map[c] = "volume"
    return df.rename(columns=rename_map)

def get_ohlcv(*, root: str | Path, symbol: str, universe: str, timeframe: str, mode: str = "closed", intraday_priority=("15m","30m","1h")) -> pd.DataFrame:
    root = Path(root)
    data_dir = root / "data" / universe / "ohlcv"
    if mode == "closed":
        return _read_csv(data_dir / f"{symbol}_{timeframe}.csv")
    if mode != "live":
        raise ValueError("mode must be 'closed' or 'live'")
    if timeframe == "1d":
        closed_df = _read_csv(data_dir / f"{symbol}_1d.csv")
        for lower_tf in intraday_priority:
            path = data_dir / f"{symbol}_{lower_tf}.csv"
            if path.exists():
                intraday_df = _read_csv(path)
                live_bar = build_live_daily_bar(intraday_df)
                if not live_bar.empty:
                    closed_norm = pd.to_datetime(closed_df["date"]).dt.normalize()
                    live_date = pd.to_datetime(live_bar["date"]).dt.normalize().iloc[0]
                    closed_df = closed_df[closed_norm != live_date].copy()
                    return pd.concat([closed_df, live_bar], ignore_index=True)
        return closed_df
    if timeframe == "1wk":
        closed_df = _read_csv(data_dir / f"{symbol}_1wk.csv")
        daily_path = data_dir / f"{symbol}_1d.csv"
        if daily_path.exists():
            daily_df = _read_csv(daily_path)
            live_bar = build_live_weekly_bar(daily_df)
            if not live_bar.empty:
                closed_weeks = pd.to_datetime(closed_df["date"]).dt.isocalendar()
                live_week = pd.to_datetime(live_bar["date"]).dt.isocalendar().iloc[0]
                mask = ~((closed_weeks.year == live_week.year) & (closed_weeks.week == live_week.week))
                closed_df = closed_df[mask].copy()
                return pd.concat([closed_df, live_bar], ignore_index=True)
        return closed_df
    raise NotImplementedError(f"live mode starter currently supports only 1d and 1wk, got {timeframe}")
