from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class RegimeConfig:
    defensive_weight: float = 0.10
    neutral_weight: float = 0.25
    risk_on_weight: float = 0.40


def market_regime(row: pd.Series) -> tuple[str, float]:
    """Map market trend, volatility and breadth proxy into a dynamic risk budget."""
    score = 0
    if row["close"] > row["ma60"]:
        score += 1
    if row["ma20"] > row["ma60"]:
        score += 1
    if row.get("adx", 0) >= 20:
        score += 1
    if row.get("relative_strength_20", 0) > 0:
        score += 1

    if score >= 3:
        return "risk_on", 0.40
    if score == 2:
        return "neutral", 0.25
    return "defensive", 0.10
