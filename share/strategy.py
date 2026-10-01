from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .universe import (
    TACTICAL_TIERS,
    attach_profiles,
    enrich_ranked_candidates,
    is_tactical_tier,
    select_diversified_candidates,
)


@dataclass(frozen=True)
class StrategyConfig:
    # Balanced book: enough defensive force, but room for opportunistic attacks.
    max_single_weight: float = 0.10
    max_tactical_weight: float = 0.04
    max_equity_weight: float = 0.50
    target_positions: int = 6
    min_history: int = 120
    lookback_trading_days: int = 1000

    lookback_fast: int = 20
    lookback_slow: int = 60
    atr_window: int = 20
    rsi_window: int = 14

    # Core: trend confirmation plus point-in-time leadership preference.
    entry_rsi_low: float = 50
    entry_rsi_high: float = 82
    max_entry_annualized_vol: float = 0.65
    min_20d_return: float = 0.00
    min_adx: float = 15
    min_relative_strength: float = 0.00

    min_holding_days: int = 10
    trend_break_confirm_days: int = 3
    trail_atr_multiple: float = 3.1
    hold_ma_buffer: float = 0.985

    # Tactical event sleeve: chase only unusually strong consecutive limit-up moves.
    tactical_minimum_entry_score: float = 0.70
    tactical_min_5d_return: float = 0.06
    tactical_min_volume_ratio: float = 1.30
    tactical_min_relative_strength: float = 0.03
    tactical_min_money_flow_proxy: float = 0.10
    tactical_min_company_score: float = 0.05
    tactical_min_industry_score: float = -0.05
    tactical_min_limit_up_streak: int = 2
    tactical_min_holding_days: int = 2
    tactical_trail_atr_multiple: float = 2.5
    max_tactical_positions: int = 1
    tactical_allocation_ratio: float = 0.10

    risk_budget_tolerance: float = 0.03

    max_industry_positions: int = 2
    min_distinct_industries: int = 4
    min_defensive_positions: int = 2
    profile_priority_weight: float = 0.20
    leader_priority_bonus: float = 0.05

    rebalance_threshold: float = 0.025
    technical_weight: float = 0.62
    event_weight: float = 0.23
    industry_weight: float = 0.15
    minimum_entry_score: float = 0.57
    reentry_cooldown_days: int = 10
    severe_event_reentry_days: int = 60


def technical_score(row: pd.Series) -> float:
    score = 0.0
    score += min(max((row["adx"] - 10) / 30, 0), 1) * 0.16
    score += min(max(row["ret_20"], -0.10), 0.25) / 0.35 * 0.18
    score += min(max(row["ret_60"], -0.15), 0.50) / 0.65 * 0.20
    score += min(max(row["ret_120"], -0.20), 0.80) / 1.00 * 0.12 if pd.notna(row.get("ret_120")) else 0.0
    score += (1.0 if row["macd_hist"] > 0 else 0.0) * 0.09
    score += (1.0 if row["obv_trend"] else 0.0) * 0.07
    score += min(max(row["volume_ratio"] - 0.8, 0), 1.5) / 1.5 * 0.05
    score += (1.0 if row["breakout_20"] else 0.0) * 0.06
    score += (1.0 if row["breakout_60"] else 0.0) * 0.05
    boll = float(row["boll_pos"]) if pd.notna(row.get("boll_pos")) else 0.5
    score += min(max(boll - 0.60, 0.0), 0.40) / 0.40 * 0.04
    score += min(max(row["relative_strength_20"], -0.10), 0.25) / 0.35 * 0.10
    risk_penalty = min(max(row["atr_pct"] - 0.02, 0), 0.08) / 0.08 * 0.17
    return max(0.0, score - risk_penalty)


def _required_indicators_present(row: pd.Series) -> bool:
    required = [
        "ma_fast", "ma_slow", "atr", "rsi", "ret_5", "ret_20", "ret_60",
        "vol_20", "adx", "volume_ratio", "relative_strength_20", "breakout_20",
    ]
    return not any(pd.isna(row.get(k)) for k in required)


def _trend_entry_eligible(row: pd.Series, cfg: StrategyConfig) -> bool:
    return bool(
        row["close"] > row["ma_slow"]
        and row["ma_fast"] > row["ma_slow"]
        and row["rsi"] >= cfg.entry_rsi_low
        and row["rsi"] <= cfg.entry_rsi_high
        and row["ret_20"] >= cfg.min_20d_return
        and row["vol_20"] > 0
        and row["vol_20"] <= cfg.max_entry_annualized_vol
        and row["adx"] >= cfg.min_adx
        and row["relative_strength_20"] >= cfg.min_relative_strength
    )


def _breakout_entry_eligible(row: pd.Series, cfg: StrategyConfig) -> bool:
    return bool(
        row["close"] > row["ma_fast"]
        and bool(row["breakout_20"])
        and row["ret_5"] >= 0.03
        and row["volume_ratio"] >= 1.20
        and row["relative_strength_20"] >= 0.00
        and row["rsi"] >= 55
        and row["rsi"] <= 92
        and row["vol_20"] > 0
        and row["vol_20"] <= 0.90
        and row["adx"] >= 12
    )


def is_entry_eligible(row: pd.Series, cfg: StrategyConfig) -> bool:
    if not _required_indicators_present(row):
        return False
    return _trend_entry_eligible(row, cfg) or _breakout_entry_eligible(row, cfg)


def is_core_entry_eligible(row: pd.Series, cfg: StrategyConfig) -> bool:
    if not _required_indicators_present(row):
        return False
    return _trend_entry_eligible(row, cfg)


def is_tactical_entry_eligible(
    row: pd.Series,
    cfg: StrategyConfig,
    *,
    company_score: float = 0.0,
    industry_score: float = 0.0,
) -> bool:
    """Selective limit-up chase.

    The historical flow input is a volume/price money-flow proxy. Live decisions
    may additionally use Longbridge's same-day capital-flow feed.
    """
    required = [
        "close", "ma_fast", "atr", "rsi", "ret_5", "ret_20", "vol_20",
        "adx", "volume_ratio", "relative_strength_20", "obv_trend",
        "money_flow_proxy_20", "limit_up_streak",
    ]
    if any(pd.isna(row.get(k)) for k in required):
        return False
    return bool(
        int(row["limit_up_streak"]) >= cfg.tactical_min_limit_up_streak
        and row["ret_5"] >= cfg.tactical_min_5d_return
        and row["volume_ratio"] >= cfg.tactical_min_volume_ratio
        and row["relative_strength_20"] >= cfg.tactical_min_relative_strength
        and row["money_flow_proxy_20"] >= cfg.tactical_min_money_flow_proxy
        and bool(row["obv_trend"])
        and row["rsi"] >= 60
        and row["rsi"] <= 95
        and row["vol_20"] <= 0.90
        and row["adx"] >= 14
        and company_score >= cfg.tactical_min_company_score
        and industry_score >= cfg.tactical_min_industry_score
    )


def should_exit(
    row: pd.Series,
    *,
    highest_close: float,
    days_held: int,
    below_ma60_streak: int,
    company_score: float = 0.0,
    severe_negative_event: bool = False,
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
        signal_date if signal_date is not None else (
            pd.Timestamp(day["date"].iloc[0])
            if "date" in day.columns and not day.empty
            else pd.Timestamp.now()
        ),
    )
    for _, row in enriched_day.iterrows():
        e = event_scores.get(row["symbol"], {})
        cs = max(min(float(e.get("company_score", 0.0)), 1.0), -1.0)
        ins = max(min(float(e.get("industry_score", 0.0)), 1.0), -1.0)

        tactical = is_tactical_entry_eligible(
            row, cfg, company_score=cs, industry_score=ins
        )
        core = (
            not is_tactical_tier(row.get("tier", "unknown"))
            and is_core_entry_eligible(row, cfg)
        )
        if not tactical and not core:
            continue

        mode = "tactical" if tactical else "core"
        ts = technical_score(row)
        composite = (
            cfg.technical_weight * ts
            + cfg.event_weight * ((cs + 1) / 2)
            + cfg.industry_weight * ((ins + 1) / 2)
        )
        min_score = cfg.tactical_minimum_entry_score if tactical else cfg.minimum_entry_score
        if composite < min_score:
            continue

        rows.append({
            "symbol": row["symbol"],
            "industry": row.get("industry", ""),
            "tier": mode,
            "base_tier": row.get("tier", "unknown"),
            "leader_score": row.get("leader_score", 0.0),
            "dynamic_leader_score": row.get("dynamic_leader_score", 0.0),
            "size_score": row.get("size_score", 0.0),
            "technical_score": ts,
            "company_score": cs,
            "industry_score": ins,
            "composite_score": composite,
            "profile_priority": 0.0,
            "selection_score": composite,
            "defensive": row.get("industry", "") in {"银行", "保险", "公共事业", "通信", "食品饮料", "家电", "医药"},
        })

    if not rows:
        return pd.DataFrame(columns=[
            "symbol", "industry", "tier", "base_tier", "leader_score",
            "dynamic_leader_score", "size_score", "technical_score",
            "company_score", "industry_score", "composite_score",
            "profile_priority", "selection_score", "defensive",
        ])

    out = enrich_ranked_candidates(pd.DataFrame(rows), priority_weight=cfg.profile_priority_weight)
    out["selection_score"] += out["tier"].eq("core").astype(float) * (
        out["leader_score"].clip(0, 1) * cfg.leader_priority_bonus
    )
    out["selection_score"] += out["dynamic_leader_score"].clip(0, 1) * 0.03
    out["selection_score"] = out["selection_score"].clip(upper=1.20)
    return out.sort_values(
        ["selection_score", "composite_score", "technical_score"],
        ascending=False,
    ).reset_index(drop=True)


def select_entries(ranked: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    return select_diversified_candidates(
        ranked,
        target_positions=cfg.target_positions,
        max_industry_positions=cfg.max_industry_positions,
        min_distinct_industries=cfg.min_distinct_industries,
        max_tactical_positions=cfg.max_tactical_positions,
        leader_priority_bonus=cfg.leader_priority_bonus,
        priority_weight=cfg.profile_priority_weight,
        min_defensive_positions=cfg.min_defensive_positions,
    )


def target_weights(
    candidates: list[str],
    equity_weight: float,
    max_single_weight: float,
    *,
    stock_tiers: dict[str, str] | None = None,
    tactical_allocation_ratio: float = 0.10,
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
