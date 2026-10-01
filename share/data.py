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


def limit_to_trading_days(df: pd.DataFrame, trading_days: int = 1000) -> pd.DataFrame:
    """Keep the latest N observations per security, ordered by trading date."""
    if trading_days <= 0:
        raise ValueError("trading_days 必须为正数")
    if df.empty:
        return df.copy()

    out = df.sort_values(["symbol", "date"]).copy()
    out = out.groupby("symbol", group_keys=False).tail(trading_days)
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def load_csv(path: str, trading_days: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少字段: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="raise")

    df = df.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"])
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC 必须为正数")

    df = df.reset_index(drop=True)
    return limit_to_trading_days(df, trading_days) if trading_days is not None else df
