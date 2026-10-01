from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def _adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    prev = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.rolling(window).mean() / atr
    minus_di = 100 * minus_dm.rolling(window).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(window).mean()


def _obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def add_indicators(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 60,
    atr_window: int = 20,
    rsi_window: int = 14,
) -> pd.DataFrame:
    parts = []
    for symbol, g in df.sort_values(["symbol", "date"]).groupby("symbol"):
        g = g.copy()
        c = g["close"]
        g["ma_fast"] = c.rolling(fast).mean()
        g["ma_slow"] = c.rolling(slow).mean()
        g["ema12"] = c.ewm(span=12, adjust=False).mean()
        g["ema26"] = c.ewm(span=26, adjust=False).mean()
        g["macd"] = g["ema12"] - g["ema26"]
        g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
        g["macd_hist"] = g["macd"] - g["macd_signal"]

        g["atr"] = _atr(g, atr_window)
        g["atr_pct"] = g["atr"] / c
        g["rsi"] = _rsi(c, rsi_window)
        g["adx"] = _adx(g, 14)

        mid = c.rolling(20).mean()
        std = c.rolling(20).std()
        g["boll_mid"] = mid
        g["boll_upper"] = mid + 2 * std
        g["boll_lower"] = mid - 2 * std
        g["boll_width"] = (g["boll_upper"] - g["boll_lower"]) / mid.replace(0, np.nan)
        g["boll_pos"] = (c - g["boll_lower"]) / (g["boll_upper"] - g["boll_lower"]).replace(0, np.nan)

        g["daily_ret"] = c.pct_change()
        # A-share daily limit-up proxy by board. This is a research signal,
        # not an exchange rule engine; ST/special-status names are not modeled.
        code = str(symbol).split(".")[0]
        limit_threshold = 0.195 if code.startswith(("300", "301", "688")) else 0.095
        g["limit_up_like"] = g["daily_ret"] >= limit_threshold
        g["limit_up_streak"] = (
            g["limit_up_like"].astype(int)
            .groupby((~g["limit_up_like"]).cumsum())
            .cumsum()
        )
        g["vol_20"] = g["daily_ret"].rolling(20).std() * np.sqrt(252)
        g["ret_5"] = c.pct_change(5)
        g["ret_20"] = c.pct_change(20)
        g["ret_60"] = c.pct_change(60)
        g["ret_120"] = c.pct_change(120)

        g["high_20"] = c.rolling(20).max()
        g["high_60"] = c.rolling(60).max()
        g["high_120"] = c.rolling(120).max()
        g["low_20"] = c.rolling(20).min()
        g["low_60"] = c.rolling(60).min()
        g["drawdown_20"] = c / g["high_20"] - 1
        g["drawdown_60"] = c / g["high_60"] - 1
        g["breakout_20"] = c >= g["high_20"].shift(1)
        g["breakout_60"] = c >= g["high_60"].shift(1)

        g["volume_ma20"] = g["volume"].rolling(20).mean()
        g["volume_ratio"] = g["volume"] / g["volume_ma20"].replace(0, np.nan)
        g["obv"] = _obv(g)
        g["obv_ma20"] = g["obv"].rolling(20).mean()
        g["obv_trend"] = g["obv"] > g["obv_ma20"]
        g["gap"] = g["open"] / c.shift(1) - 1
        g["range_pct"] = (g["high"] - g["low"]) / c

        parts.append(g)

    return pd.concat(parts, ignore_index=True)


def add_relative_strength(df: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Add relative strength versus a market proxy using same-date closes only."""
    b = benchmark[["date", "close"]].drop_duplicates("date").sort_values("date").copy()
    b["benchmark_ret20"] = b["close"].pct_change(20)
    b["benchmark_ret60"] = b["close"].pct_change(60)
    out = df.merge(b[["date", "benchmark_ret20", "benchmark_ret60"]], on="date", how="left")
    out["relative_strength_20"] = out["ret_20"] - out["benchmark_ret20"]
    out["relative_strength_60"] = out["ret_60"] - out["benchmark_ret60"]
    return out
