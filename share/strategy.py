from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    max_single_weight: float = 0.08
    max_equity_weight: float = 0.40
    target_positions: int = 5
    min_history: int = 120

    entry_rsi_low: float = 42
    entry_rsi_high: float = 78
    max_entry_annualized_vol: float = 0.65
    min_20d_return: float = -0.02
    min_adx: float = 15

    min_holding_days: int = 10
    trend_break_confirm_days: int = 3
    trail_atr_multiple: float = 3.2
    hold_ma_buffer: float = 0.985

    rebalance_threshold: float = 0.025
    technical_weight: float = 0.65
    event_weight: float = 0.20
    industry_weight: float = 0.15


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
    required = ["ma_fast","ma_slow","atr","rsi","ret_20","ret_60","vol_20","adx","volume_ratio","relative_strength_20"]
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


def should_exit(row: pd.Series, *, highest_close: float, days_held: int,
                below_ma60_streak: int, company_score: float,
                severe_negative_event: bool, cfg: StrategyConfig) -> tuple[bool, str]:
    if severe_negative_event:
        return True, "severe_negative_company_event"
    if days_held < cfg.min_holding_days:
        return False, ""

    if row["close"] < highest_close - cfg.trail_atr_multiple * row["atr"]:
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


def rank_candidates(day: pd.DataFrame, event_scores: dict[str, dict], cfg: StrategyConfig) -> pd.DataFrame:
    rows = []
    for _, row in day.iterrows():
        symbol = row["symbol"]
        if not is_entry_eligible(row, cfg):
            continue
        e = event_scores.get(symbol, {})
        ts = technical_score(row)
        cs = max(min(e.get("company_score", 0.0), 1.0), -1.0)
        ins = max(min(e.get("industry_score", 0.0), 1.0), -1.0)
        composite = cfg.technical_weight * ts + cfg.event_weight * ((cs + 1) / 2) + cfg.industry_weight * ((ins + 1) / 2)
        rows.append({"symbol": symbol, "technical_score": ts, "company_score": cs,
                     "industry_score": ins, "composite_score": composite})
    if not rows:
        return pd.DataFrame(columns=["symbol", "technical_score", "company_score", "industry_score", "composite_score"])
    return pd.DataFrame(rows).sort_values("composite_score", ascending=False).reset_index(drop=True)


def target_weights(candidates: list[str], equity_weight: float, max_single_weight: float) -> dict[str, float]:
    if not candidates or equity_weight <= 0:
        return {}
    n = min(len(candidates), max(1, int(round(equity_weight / max_single_weight))))
    chosen = candidates[:n]
    w = min(max_single_weight, equity_weight / len(chosen))
    return {s: w for s in chosen}
