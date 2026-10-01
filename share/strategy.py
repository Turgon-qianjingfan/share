from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class StrategyConfig:
    max_single_weight: float = 0.08
    max_equity_weight: float = 0.40
    target_positions: int = 5
    min_history: int = 80
    rsi_low: float = 45
    rsi_high: float = 72
    max_annualized_vol: float = 0.55

def select_candidates(day: pd.DataFrame, cfg: StrategyConfig) -> list[str]:
    d = day.dropna(subset=["ma_fast","ma_slow","atr","rsi","ret_20","vol_20"]).copy()
    if d.empty:
        return []
    d = d[(d["close"] > d["ma_slow"]) & (d["ma_fast"] > d["ma_slow"])]
    d = d[(d["rsi"] >= cfg.rsi_low) & (d["rsi"] <= cfg.rsi_high)]
    d = d[(d["vol_20"] > 0) & (d["vol_20"] <= cfg.max_annualized_vol)]
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
