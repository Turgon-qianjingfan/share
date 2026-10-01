from __future__ import annotations

from dataclasses import dataclass
import math
import pandas as pd


@dataclass(frozen=True)
class EventConfig:
    company_half_life_days: float = 10.0
    industry_half_life_days: float = 15.0
    max_lookback_days: int = 30
    severe_negative_score: float = -0.75


REQUIRED = ["event_time", "symbol", "direction", "severity", "event_type"]


def validate_events(events: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in events.columns]
    if missing:
        raise ValueError(f"事件数据缺少字段: {missing}")
    out = events.copy()
    out["event_time"] = pd.to_datetime(out["event_time"], utc=True, errors="raise")
    out["direction"] = pd.to_numeric(out["direction"], errors="raise")
    out["severity"] = pd.to_numeric(out["severity"], errors="raise").clip(0, 3)
    return out.sort_values("event_time").reset_index(drop=True)


def _decay(days: float, half_life: float) -> float:
    return math.exp(-math.log(2) * max(days, 0.0) / half_life)


def event_score_for_day(
    events: pd.DataFrame,
    *,
    symbol: str,
    signal_date: pd.Timestamp,
    cfg: EventConfig | None = None,
) -> dict:
    cfg = cfg or EventConfig()
    if events.empty:
        return {"company_score": 0.0, "industry_score": 0.0, "severe_negative": False, "event_count": 0}

    e = events[events["event_time"] <= pd.Timestamp(signal_date, tz="UTC")].copy()
    if e.empty:
        return {"company_score": 0.0, "industry_score": 0.0, "severe_negative": False, "event_count": 0}

    age = (pd.Timestamp(signal_date, tz="UTC") - e["event_time"]).dt.total_seconds() / 86400
    e = e[age <= cfg.max_lookback_days].copy()
    if e.empty:
        return {"company_score": 0.0, "industry_score": 0.0, "severe_negative": False, "event_count": 0}

    e["age_days"] = (pd.Timestamp(signal_date, tz="UTC") - e["event_time"]).dt.total_seconds() / 86400
    e["weight"] = e.apply(
        lambda r: (r["direction"] / 3.0) * r["severity"]
        * _decay(r["age_days"], cfg.company_half_life_days),
        axis=1,
    )

    own = e[e["symbol"] == symbol]
    company_score = float(own["weight"].sum()) if not own.empty else 0.0
    severe_negative = bool((own["weight"] <= cfg.severe_negative_score).any())

    industry_score = float(e.loc[e["event_type"] == "industry", "weight"].sum())
    return {
        "company_score": company_score,
        "industry_score": industry_score,
        "severe_negative": severe_negative,
        "event_count": int(len(e)),
    }


def load_event_csv(path: str) -> pd.DataFrame:
    return validate_events(pd.read_csv(path))
