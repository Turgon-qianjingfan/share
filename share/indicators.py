from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame, fast=20, slow=60, atr_window=20, rsi_window=14) -> pd.DataFrame:
    out = df.copy().sort_values(["symbol", "date"])
    g = out.groupby("symbol", group_keys=False)
    out["ma_fast"] = g["close"].transform(lambda s: s.rolling(fast).mean())
    out["ma_slow"] = g["close"].transform(lambda s: s.rolling(slow).mean())

    prev = g["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev).abs(),
            (out["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.groupby(out["symbol"]).transform(
        lambda s: s.rolling(atr_window).mean()
    )

    delta = g["close"].diff()
    gain = delta.clip(lower=0).groupby(out["symbol"]).transform(
        lambda s: s.rolling(rsi_window).mean()
    )
    loss = (-delta.clip(upper=0)).groupby(out["symbol"]).transform(
        lambda s: s.rolling(rsi_window).mean()
    )

    # Explicitly define RSI edge cases: no losses with positive gains => 100;
    # no gains and no losses => neutral 50. Avoid NaN on persistent uptrends,
    # which previously caused otherwise eligible candidates to be dropped.
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
    rsi = rsi.mask((loss == 0) & (gain == 0), 50.0)
    out["rsi"] = rsi

    out["ret_20"] = g["close"].pct_change(fast)
    out["vol_20"] = g["close"].pct_change().groupby(out["symbol"]).transform(
        lambda s: s.rolling(20).std() * np.sqrt(252)
    )
    out["volume_ma20"] = g["volume"].transform(lambda s: s.rolling(20).mean())
    return out
