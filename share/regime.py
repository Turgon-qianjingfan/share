from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class RegimeConfig:
    risk_on_weight: float = 0.50
    neutral_weight: float = 0.35
    defensive_weight: float = 0.20
    crisis_weight: float = 0.00


def market_regime(
    benchmark_row: pd.Series,
    breadth: float,
    breadth_ma20: float | None = None,
    cfg: RegimeConfig | None = None,
) -> tuple[str, float]:
    cfg = cfg or RegimeConfig()
    score = 0
    if benchmark_row["close"] > benchmark_row["ma_slow"]:
        score += 1
    if benchmark_row["ma_fast"] > benchmark_row["ma_slow"]:
        score += 1
    if breadth >= 0.55:
        score += 1
    if breadth_ma20 is not None and breadth >= breadth_ma20:
        score += 1

    drawdown_60 = benchmark_row.get("drawdown_60", 0.0)
    if drawdown_60 <= -0.15:
        return "crisis", cfg.crisis_weight
    if score >= 3:
        return "risk_on", cfg.risk_on_weight
    if score == 2:
        return "neutral", cfg.neutral_weight
    return "defensive", cfg.defensive_weight
