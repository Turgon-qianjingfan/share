from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

REQUIRED = ["date", "symbol", "open", "high", "low", "close", "volume"]

@dataclass(frozen=True)
class Bar:
    date: pd.Timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少字段: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    df = df.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"])
    if (df[["open","high","low","close"]] <= 0).any().any():
        raise ValueError("OHLC 必须为正数")
    return df.reset_index(drop=True)
