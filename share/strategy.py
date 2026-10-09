from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    max_single_weight: float = 0.08
    max_equity_weight: float = 0.40
    target_positions: int = 5
    min_history: int = 80

    # Indicator lookbacks (kept explicit because Backtester passes these through).
    lookback_fast: int = 20
    lookback_slow: int = 60
    atr_window: int = 20

    # Entry filters
    entry_rsi_low: float = 45
    entry_rsi_high: float = 72
    max_entry_annualized_vol: float = 0.60
    min_20d_return: float = 0.00

    # Holding / exit controls
    min_holding_days: int = 10
    trend_break_confirm_days: int = 3
    trail_atr_multiple: float = 3.0
    hold_ma_buffer: float = 0.985

    # Trading friction / turnover control
    rebalance_threshold: float = 0.025  # 2.5 percentage points of portfolio


def entry_candidates(day: pd.DataFrame, cfg: StrategyConfig) -> list[str]:
    """New entries only. High RSI blocks new entries, but does not force exits."""
    d = day.dropna(subset=["ma_fast", "ma_slow", "atr", "rsi", "ret_20", "vol_20"]).copy()
    if d.empty:
        return []

    d = d[
        (d["close"] > d["ma_slow"])
        & (d["ma_fast"] > d["ma_slow"])
        & (d["rsi"] >= cfg.entry_rsi_low)
        & (d["rsi"] <= cfg.entry_rsi_high)
        & (d["ret_20"] >= cfg.min_20d_return)
        & (d["vol_20"] > 0)
        & (d["vol_20"] <= cfg.max_entry_annualized_vol)
    ]

    if d.empty:
        return []

    d["score"] = (
        d["ret_20"].rank(pct=True) * 0.55
        + (-d["vol_20"]).rank(pct=True) * 0.25
        + (-d["atr"] / d["close"]).rank(pct=True) * 0.20
    )
    return d.nlargest(cfg.target_positions, "score")["symbol"].tolist()


def should_exit(
    row: pd.Series,
    *,
    highest_close: float,
    days_held: int,
    below_ma60_streak: int,
    cfg: StrategyConfig,
) -> tuple[bool, str]:
    """Exit only on a real risk event, not because RSI is temporarily high."""
    # Protective stops must never be disabled by the minimum holding period.
    # The minimum period only suppresses ordinary trend-based exits.
    atr_stop = highest_close - cfg.trail_atr_multiple * row["atr"]
    if row["close"] < atr_stop:
        return True, "atr_trailing_stop"

    if days_held < cfg.min_holding_days:
        return False, ""

    trend_break = (
        row["close"] < row["ma_slow"] * cfg.hold_ma_buffer
        and row["ma_fast"] < row["ma_slow"]
        and below_ma60_streak >= cfg.trend_break_confirm_days
    )
    if trend_break:
        return True, "confirmed_trend_break"

    return False, ""


def target_weights(
    candidates: list[str],
    equity_weight: float,
    max_single_weight: float,
) -> dict[str, float]:
    if not candidates or equity_weight <= 0:
        return {}
    w = min(max_single_weight, equity_weight / len(candidates))
    return {s: w for s in candidates}
