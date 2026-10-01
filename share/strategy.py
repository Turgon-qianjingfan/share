from __future__ import annotations
import pandas as pd
from dataclasses import dataclass

@dataclass(frozen=True)
class StrategyConfig:
    max_single_weight: float = 0.10
    max_equity_weight: float = 0.60
    target_positions: int = 6
    min_history: int = 80

def select_candidates(day: pd.DataFrame, cfg: StrategyConfig) -> list[str]:
    d = day.dropna(subset=["ma_fast","ma_slow","atr","rsi","ret_20","vol_20"]).copy()
    d = d[(d["close"] > d["ma_slow"]) & (d["ma_fast"] > d["ma_slow"])]
    d = d[(d["rsi"] >= 45) & (d["rsi"] <= 72)]
    d = d[(d["vol_20"] > 0) & (d["vol_20"] <= 0.55)]
    if d.empty:
        return []
    d["score"] = (
        d["ret_20"].rank(pct=True) * 0.55
        + (-d["vol_20"]).rank(pct=True) * 0.25
        + (-d["atr"] / d["close"]).rank(pct=True) * 0.20
    )
    return d.nlargest(cfg.target_positions, "score")["symbol"].tolist()

def target_weights(candidates: list[str], equity_weight: float, max_single_weight: float) -> dict[str,float]:
    if not candidates or equity_weight <= 0:
        return {}
    w = min(max_single_weight, equity_weight / len(candidates))
    return {s: w for s in candidates}
