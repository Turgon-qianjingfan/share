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
    source_quality_floor: float = 0.5


REQUIRED = ["event_time", "symbol", "direction", "severity", "event_type"]
DIRECTION_MAP = {"long": 1.0, "neutral": 0.0, "short": -1.0, "positive": 1.0, "negative": -1.0}
TYPE_WEIGHT = {
    "earnings": 1.2,
    "guidance": 1.2,
    "regulation": 1.15,
    "m_and_a": 1.1,
    "management": 0.8,
    "production": 1.0,
    "supply": 1.1,
    "commodity": 1.0,
    "industry": 1.0,
    "other": 0.7,
}


def validate_events(events: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in events.columns]
    if missing:
        raise ValueError(f"事件数据缺少字段: {missing}")

    out = events.copy()
    out["event_time"] = pd.to_datetime(out["event_time"], utc=True, errors="raise")
    out["direction"] = out["direction"].map(
        lambda x: DIRECTION_MAP.get(str(x).lower(), float(x) if str(x).replace(".", "", 1).replace("-", "", 1).isdigit() else 0.0)
    )
    out["severity"] = pd.to_numeric(out["severity"], errors="raise").clip(0, 3)
    if "source_quality" not in out:
        out["source_quality"] = 1.0
    out["source_quality"] = pd.to_numeric(out["source_quality"], errors="coerce").fillna(1.0).clip(0.5, 1.0)
    if "industry" not in out:
        out["industry"] = ""
    if "title" not in out:
        out["title"] = ""
    return out.sort_values("event_time").reset_index(drop=True)


def _decay(days: float, half_life: float) -> float:
    return math.exp(-math.log(2) * max(days, 0.0) / half_life)


def event_score_for_day(
    events: pd.DataFrame,
    *,
    symbol: str,
    industry: str = "",
    signal_date: pd.Timestamp,
    cfg: EventConfig | None = None,
) -> dict:
    cfg = cfg or EventConfig()
    if events is None or events.empty:
        return {"company_score": 0.0, "industry_score": 0.0, "severe_negative": False, "event_count": 0}

    signal = pd.Timestamp(signal_date)
    if signal.tzinfo is None:
        signal = signal.tz_localize("UTC")
    e = events[events["event_time"] <= signal].copy()
    if e.empty:
        return {"company_score": 0.0, "industry_score": 0.0, "severe_negative": False, "event_count": 0}

    age = (signal - e["event_time"]).dt.total_seconds() / 86400
    e = e[age <= cfg.max_lookback_days].copy()
    if e.empty:
        return {"company_score": 0.0, "industry_score": 0.0, "severe_negative": False, "event_count": 0}

    e["age_days"] = (signal - e["event_time"]).dt.total_seconds() / 86400
    e["type_weight"] = e["event_type"].map(TYPE_WEIGHT).fillna(0.7)
    e["weight"] = (
        e["direction"]
        * (e["severity"] / 3.0)
        * e["source_quality"]
        * e["type_weight"]
    )

    own = e[(e["symbol"] == symbol) & (e["event_type"] != "industry")].copy()
    company_score = float(
        (own["weight"] * own["age_days"].map(lambda d: _decay(d, cfg.company_half_life_days))).sum()
    ) if not own.empty else 0.0

    severe_negative = bool(company_score <= cfg.severe_negative_score)
    industry_events = e[(e["event_type"] == "industry") & (e["industry"] == industry)]
    industry_score = float(
        (industry_events["weight"] * industry_events["age_days"].map(lambda d: _decay(d, cfg.industry_half_life_days))).sum()
    ) if not industry_events.empty else 0.0

    return {
        "company_score": company_score,
        "industry_score": industry_score,
        "severe_negative": severe_negative,
        "event_count": int(len(own) + len(industry_events)),
    }


def load_event_csv(path: str) -> pd.DataFrame:
    return validate_events(pd.read_csv(path))
