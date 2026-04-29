import pandas as pd

def build_live_weekly_bar(df_daily: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if df_daily.empty:
        return pd.DataFrame(columns=[date_col, "open", "high", "low", "close", "volume"])
    out = df_daily.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    last_dt = out[date_col].iloc[-1]
    iso = last_dt.isocalendar()
    week_df = out[(out[date_col].dt.isocalendar().year == iso.year) & (out[date_col].dt.isocalendar().week == iso.week)].copy()
    if week_df.empty:
        return pd.DataFrame(columns=[date_col, "open", "high", "low", "close", "volume"])
    return pd.DataFrame([{
        date_col: pd.to_datetime(week_df[date_col].iloc[0]).normalize(),
        "open": float(week_df["open"].iloc[0]),
        "high": float(week_df["high"].max()),
        "low": float(week_df["low"].min()),
        "close": float(week_df["close"].iloc[-1]),
        "volume": float(week_df["volume"].sum()) if "volume" in week_df.columns else 0.0,
    }])
