from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

REQUIRED = ["date", "symbol", "open", "high", "low", "close", "volume"]
OPTIONAL_NUMERIC = ["limit_up", "limit_down"]
OPTIONAL_BOOLEAN = ["is_suspended"]


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
    df["symbol"] = df["symbol"].astype(str)
    for c in ["open", "high", "low", "close", "volume", *OPTIONAL_NUMERIC]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="raise")
    if "is_suspended" in df.columns:
        if df["is_suspended"].dtype != bool:
            normalized = df["is_suspended"].astype(str).str.strip().str.lower()
            mapping = {"true": True, "false": False, "1": True, "0": False,
                       "yes": True, "no": False}
            invalid = ~normalized.isin(mapping)
            if invalid.any():
                raise ValueError("is_suspended 只能使用 true/false、1/0、yes/no")
            df["is_suspended"] = normalized.map(mapping).astype(bool)
    prices = df[["open", "high", "low", "close"]]
    if (prices <= 0).any().any():
        raise ValueError("OHLC 必须为正数")
    if (df["high"] < df[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError("high 不能低于 open、low 或 close")
    if (df["low"] > df[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("low 不能高于 open、high 或 close")
    if (df["volume"] < 0).any():
        raise ValueError("volume 不能为负数")
    for col in OPTIONAL_NUMERIC:
        if col in df.columns and (df[col].dropna() <= 0).any():
            raise ValueError(f"{col} 必须为正数或空值")
    df = df.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"])
    return df.reset_index(drop=True)
