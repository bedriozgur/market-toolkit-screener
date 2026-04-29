import pandas as pd

def build_live_daily_bar(df_intraday: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if df_intraday.empty:
        return pd.DataFrame(columns=[date_col, "open", "high", "low", "close", "volume"])
    out = df_intraday.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    day = out[date_col].dt.normalize().iloc[-1]
    day_df = out[out[date_col].dt.normalize() == day].copy()
    if day_df.empty:
        return pd.DataFrame(columns=[date_col, "open", "high", "low", "close", "volume"])
    return pd.DataFrame([{
        date_col: day,
        "open": float(day_df["open"].iloc[0]),
        "high": float(day_df["high"].max()),
        "low": float(day_df["low"].min()),
        "close": float(day_df["close"].iloc[-1]),
        "volume": float(day_df["volume"].sum()) if "volume" in day_df.columns else 0.0,
    }])
