from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .universe import (
    CORE_TIERS,
    TACTICAL_TIERS,
    attach_profiles,
    enrich_ranked_candidates,
    is_tactical_tier,
    select_diversified_candidates,
)


@dataclass(frozen=True)
class StrategyConfig:
    max_single_weight: float = 0.08
    max_tactical_weight: float = 0.04
    max_equity_weight: float = 0.40
    target_positions: int = 5
    min_history: int = 120
    lookback_trading_days: int = 1000

    lookback_fast: int = 20
    lookback_slow: int = 60
    atr_window: int = 20
    rsi_window: int = 14

    entry_rsi_low: float = 42
    entry_rsi_high: float = 78
    max_entry_annualized_vol: float = 0.65
    min_20d_return: float = -0.02
    min_adx: float = 15

    min_holding_days: int = 10
    trend_break_confirm_days: int = 3
    trail_atr_multiple: float = 3.2
    hold_ma_buffer: float = 0.985

    tactical_minimum_entry_score: float = 0.62
    tactical_min_5d_return: float = 0.03
    tactical_min_volume_ratio: float = 1.20
    tactical_min_relative_strength: float = 0.0
    tactical_min_holding_days: int = 3
    tactical_trail_atr_multiple: float = 2.5
    max_tactical_positions: int = 2
    tactical_allocation_ratio: float = 0.20

    max_industry_positions: int = 2
    min_distinct_industries: int = 3
    profile_priority_weight: float = 0.10

    rebalance_threshold: float = 0.025
    technical_weight: float = 0.65
    event_weight: float = 0.20
    industry_weight: float = 0.15
    minimum_entry_score: float = 0.55
    reentry_cooldown_days: int = 15
    severe_event_reentry_days: int = 60


def technical_score(row: pd.Series) -> float:
    score = 0.0
    score += min(max((row["adx"] - 10) / 30, 0), 1) * 0.18
    score += min(max(row["ret_20"], -0.10), 0.25) / 0.35 * 0.20
    score += min(max(row["ret_60"], -0.15), 0.50) / 0.65 * 0.20
    score += (1.0 if row["macd_hist"] > 0 else 0.0) * 0.10
    score += (1.0 if row["obv_trend"] else 0.0) * 0.08
    score += min(max(row["volume_ratio"] - 0.8, 0), 1.5) / 1.5 * 0.06
    score += (1.0 if row["breakout_20"] else 0.0) * 0.08
    score += min(max(row["relative_strength_20"], -0.10), 0.20) / 0.30 * 0.10
    risk_penalty = min(max(row["atr_pct"] - 0.02, 0), 0.08) / 0.08 * 0.18
    return max(0.0, score - risk_penalty)


def is_entry_eligible(row: pd.Series, cfg: StrategyConfig) -> bool:
    required = [
        "ma_fast", "ma_slow", "atr", "rsi", "ret_5", "ret_20", "ret_60",
        "vol_20", "adx", "volume_ratio", "relative_strength_20",
    ]
    if any(pd.isna(row.get(k)) for k in required):
        return False
    return bool(
        row["close"] > row["ma_slow"]
        and row["ma_fast"] > row["ma_slow"]
        and row["rsi"] >= cfg.entry_rsi_low
        and row["rsi"] <= cfg.entry_rsi_high
        and row["ret_20"] >= cfg.min_20d_return
        and row["vol_20"] > 0
        and row["vol_20"] <= cfg.max_entry_annualized_vol
        and row["adx"] >= cfg.min_adx
        and row["relative_strength_20"] > -0.03
    )


def is_tactical_entry_eligible(row: pd.Series, cfg: StrategyConfig) -> bool:
    """Small/micro caps are treated as a short-horizon momentum sleeve."""
    if not is_entry_eligible(row, cfg):
        return False
    return bool(
        (
            bool(row["breakout_20"])
            or float(row["ret_5"]) >= cfg.tactical_min_5d_return
        )
        and float(row["volume_ratio"]) >= cfg.tactical_min_volume_ratio
        and float(row["relative_strength_20"]) >= cfg.tactical_min_relative_strength
    )


def should_exit(
    row: pd.Series,
    *,
    highest_close: float,
    days_held: int,
    below_ma60_streak: int,
    company_score: float,
    severe_negative_event: bool,
    cfg: StrategyConfig,
    min_holding_days: int | None = None,
    trail_atr_multiple: float | None = None,
) -> tuple[bool, str]:
    if severe_negative_event:
        return True, "severe_negative_company_event"

    min_hold = cfg.min_holding_days if min_holding_days is None else min_holding_days
    trail = cfg.trail_atr_multiple if trail_atr_multiple is None else trail_atr_multiple

    if days_held < min_hold:
        return False, ""

    if row["close"] < highest_close - trail * row["atr"]:
        return True, "atr_trailing_stop"

    if (
        row["close"] < row["ma_slow"] * cfg.hold_ma_buffer
        and row["ma_fast"] < row["ma_slow"]
        and below_ma60_streak >= cfg.trend_break_confirm_days
    ):
        return True, "confirmed_trend_break"

    if company_score < -0.65:
        return True, "negative_company_event"

    return False, ""


def rank_candidates(
    day: pd.DataFrame,
    event_scores: dict[str, dict],
    cfg: StrategyConfig,
    stock_profiles: pd.DataFrame | None = None,
    signal_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    rows = []
    enriched_day = attach_profiles(
        day,
        stock_profiles,
        signal_date if signal_date is not None else pd.Timestamp(day["date"].iloc[0]) if "date" in day.columns and not day.empty else pd.Timestamp.now(),
    )
    for _, row in enriched_day.iterrows():
        symbol = row["symbol"]
        if not is_entry_eligible(row, cfg):
            continue

        tactical = is_tactical_tier(row.get("tier", "unknown"))
        if tactical and not is_tactical_entry_eligible(row, cfg):
            continue

        e = event_scores.get(symbol, {})
        ts = technical_score(row)
        cs = max(min(float(e.get("company_score", 0.0)), 1.0), -1.0)
        ins = max(min(float(e.get("industry_score", 0.0)), 1.0), -1.0)
        composite = (
            cfg.technical_weight * ts
            + cfg.event_weight * ((cs + 1) / 2)
            + cfg.industry_weight * ((ins + 1) / 2)
        )
        min_score = cfg.tactical_minimum_entry_score if tactical else cfg.minimum_entry_score
        if composite < min_score:
            continue

        rows.append({
            "symbol": symbol,
            "industry": row.get("industry", ""),
            "tier": row.get("tier", "unknown"),
            "leader_score": row.get("leader_score", 0.0),
            "size_score": row.get("size_score", 0.0),
            "technical_score": ts,
            "company_score": cs,
            "industry_score": ins,
            "composite_score": composite,
            "profile_priority": 0.0,
            "selection_score": composite,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "symbol", "industry", "tier", "leader_score", "size_score",
            "technical_score", "company_score", "industry_score",
            "composite_score", "profile_priority", "selection_score",
        ])

    return enrich_ranked_candidates(pd.DataFrame(rows))


def select_entries(ranked: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    return select_diversified_candidates(
        ranked,
        target_positions=cfg.target_positions,
        max_industry_positions=cfg.max_industry_positions,
        min_distinct_industries=cfg.min_distinct_industries,
        max_tactical_positions=cfg.max_tactical_positions,
    )


def target_weights(
    candidates: list[str],
    equity_weight: float,
    max_single_weight: float,
    *,
    stock_tiers: dict[str, str] | None = None,
    tactical_allocation_ratio: float = 0.20,
    max_tactical_weight: float = 0.04,
) -> dict[str, float]:
    if not candidates or equity_weight <= 0:
        return {}

    stock_tiers = stock_tiers or {}
    tactical = [s for s in candidates if stock_tiers.get(s, "unknown") in TACTICAL_TIERS]
    core = [s for s in candidates if s not in tactical]

    if core and tactical:
        tactical_budget = equity_weight * tactical_allocation_ratio
        core_budget = equity_weight - tactical_budget
    elif core:
        tactical_budget = 0.0
        core_budget = equity_weight
    else:
        # With no core names available, keep the entire book small-cap constrained.
        tactical_budget = min(equity_weight, len(tactical) * max_tactical_weight)
        core_budget = 0.0

    weights: dict[str, float] = {}
    if core:
        per_core = min(max_single_weight, core_budget / len(core))
        weights.update({s: per_core for s in core})

    if tactical:
        per_tactical = min(max_tactical_weight, tactical_budget / len(tactical))
        weights.update({s: per_tactical for s in tactical})

    return {s: w for s, w in weights.items() if w > 0}
