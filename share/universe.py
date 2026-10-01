from __future__ import annotations

import pandas as pd

TIER_PRIORITY = {
    "leader": 1.00,
    "large": 0.85,
    "mid": 0.60,
    "small": 0.30,
    "micro": 0.15,
    "unknown": 0.50,
}

CORE_TIERS = {"leader", "large", "mid"}
TACTICAL_TIERS = {"small", "micro"}

REQUIRED_PROFILE_FIELDS = ["symbol", "industry", "tier"]


def validate_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    if profiles is None or profiles.empty:
        return pd.DataFrame(columns=["symbol", "industry", "tier", "leader_score", "size_score", "effective_date"])

    out = profiles.copy()
    missing = [c for c in REQUIRED_PROFILE_FIELDS if c not in out.columns]
    if missing:
        raise ValueError(f"股票画像缺少字段: {missing}")

    out["symbol"] = out["symbol"].astype(str)
    out["industry"] = out["industry"].fillna("").astype(str)
    out["tier"] = out["tier"].fillna("unknown").astype(str).str.lower()
    invalid = sorted(set(out["tier"]) - set(TIER_PRIORITY))
    if invalid:
        raise ValueError(f"未知股票层级: {invalid}")

    if "leader_score" not in out.columns:
        out["leader_score"] = 0.0
    if "size_score" not in out.columns:
        out["size_score"] = 0.0

    out["leader_score"] = pd.to_numeric(out["leader_score"], errors="coerce").fillna(0.0).clip(0, 1)
    out["size_score"] = pd.to_numeric(out["size_score"], errors="coerce").fillna(0.0).clip(0, 1)

    if "effective_date" in out.columns:
        out["effective_date"] = pd.to_datetime(out["effective_date"])
        out = out.sort_values(["symbol", "effective_date"]).drop_duplicates(
            ["symbol", "effective_date"], keep="last"
        )
    else:
        out["effective_date"] = pd.Timestamp.min

    return out.reset_index(drop=True)



def infer_profile_scores(day: pd.DataFrame) -> pd.DataFrame:
    """Infer size/leadership scores from point-in-time market_cap when available.

    This is a transparent heuristic: size is the cross-sectional market-cap
    percentile; leadership combines within-industry market-cap rank and share
    versus the largest company in the same industry. It never uses future rows.
    """
    out = day.copy()
    out["inferred_size_score"] = 0.0
    out["inferred_leader_score"] = 0.0
    if "market_cap" not in out.columns:
        return out

    mc = pd.to_numeric(out["market_cap"], errors="coerce")
    valid = mc.notna() & (mc > 0)
    if not valid.any():
        return out

    out.loc[valid, "inferred_size_score"] = mc[valid].rank(pct=True)

    industry = out.get("industry", pd.Series("", index=out.index)).fillna("").astype(str)
    leader_rank = pd.Series(0.0, index=out.index)
    leader_ratio = pd.Series(0.0, index=out.index)
    for _, idx in out.loc[valid].groupby(industry[valid]).groups.items():
        vals = mc.loc[idx]
        if len(vals) >= 2:
            leader_rank.loc[idx] = vals.rank(pct=True)
            leader_ratio.loc[idx] = vals / vals.max()
    out["inferred_leader_score"] = (0.50 * leader_rank + 0.50 * leader_ratio).clip(0, 1)
    return out

def load_profiles(path: str) -> pd.DataFrame:
    return validate_profiles(pd.read_csv(path))


def profile_snapshot(profiles: pd.DataFrame, signal_date: pd.Timestamp) -> pd.DataFrame:
    profiles = validate_profiles(profiles)
    if profiles.empty:
        return profiles
    dt = pd.Timestamp(signal_date)
    usable = profiles[profiles["effective_date"] <= dt]
    if usable.empty:
        return profiles.iloc[0:0].copy()
    return usable.sort_values(["symbol", "effective_date"]).drop_duplicates(
        "symbol", keep="last"
    )


def attach_profiles(day: pd.DataFrame, profiles: pd.DataFrame | None, signal_date: pd.Timestamp) -> pd.DataFrame:
    out = day.copy()
    out["industry"] = out.get("industry", "")
    out["tier"] = "unknown"
    out["leader_score"] = 0.0
    out["size_score"] = 0.0

    if profiles is not None and not profiles.empty:
        snap = profile_snapshot(profiles, signal_date)
        keep = ["symbol", "industry", "tier", "leader_score", "size_score"]
        snap = snap[keep].copy()
        out = out.merge(snap, on="symbol", how="left", suffixes=("", "_profile"))
        if "industry_profile" in out.columns:
            out["industry"] = out["industry_profile"].where(
                out["industry_profile"].notna() & (out["industry_profile"] != ""), out["industry"]
            )
            out = out.drop(columns=["industry_profile"])

        out["tier"] = out["tier"].fillna("unknown").astype(str).str.lower()
        out["leader_score"] = pd.to_numeric(out["leader_score"], errors="coerce").fillna(0.0).clip(0, 1)
        out["size_score"] = pd.to_numeric(out["size_score"], errors="coerce").fillna(0.0).clip(0, 1)

    inferred = infer_profile_scores(out)

    # A point-in-time market-cap field can supply a fallback when no explicit
    # profile score is present.
    out.loc[out["leader_score"] <= 0, "leader_score"] = inferred["inferred_leader_score"]
    out.loc[out["size_score"] <= 0, "size_score"] = inferred["inferred_size_score"]

    unknown = out["tier"].eq("unknown")
    out.loc[unknown & (out["leader_score"] >= 0.80), "tier"] = "leader"
    out.loc[unknown & (out["size_score"] >= 0.80), "tier"] = "large"
    out.loc[unknown & (out["size_score"] <= 0.20), "tier"] = "small"
    out.loc[unknown, "tier"] = "mid"
    return out


def profile_priority(row: pd.Series) -> float:
    tier_score = TIER_PRIORITY.get(str(row.get("tier", "unknown")).lower(), 0.50)
    leader = float(row.get("leader_score", 0.0) or 0.0)
    size = float(row.get("size_score", 0.0) or 0.0)
    return max(0.0, min(1.0, 0.50 * tier_score + 0.30 * leader + 0.20 * size))


def is_tactical_tier(tier: str) -> bool:
    return str(tier).lower() in TACTICAL_TIERS


def enrich_ranked_candidates(ranked: pd.DataFrame, priority_weight: float = 0.10) -> pd.DataFrame:
    out = ranked.copy()
    if out.empty:
        for c in ["industry", "tier", "leader_score", "size_score", "profile_priority", "selection_score"]:
            if c not in out.columns:
                out[c] = []
        return out

    out["tier"] = out["tier"].fillna("unknown").astype(str).str.lower()
    out["industry"] = out["industry"].fillna("").astype(str)
    out["profile_priority"] = out.apply(profile_priority, axis=1)
    out["selection_score"] = out["composite_score"] + max(0.0, priority_weight) * out["profile_priority"]
    return out.sort_values(
        ["selection_score", "composite_score", "technical_score"],
        ascending=False,
    ).reset_index(drop=True)


def select_diversified_candidates(
    ranked: pd.DataFrame,
    target_positions: int,
    max_industry_positions: int = 2,
    min_distinct_industries: int = 3,
    max_tactical_positions: int = 2,
) -> pd.DataFrame:
    """Select across industries while prioritising leader/large names.

    The selection is deterministic. It first tries to cover distinct industries,
    then fills remaining slots from the strongest core names, and only then
    admits a limited tactical sleeve for small/micro caps.
    """
    if ranked is None or ranked.empty or target_positions <= 0:
        return ranked.iloc[0:0].copy() if ranked is not None else pd.DataFrame()

    out = enrich_ranked_candidates(ranked)
    chosen: list[dict] = []
    industry_counts: dict[str, int] = {}
    tactical_count = 0

    def can_take(row: pd.Series, force_new_industry: bool = False) -> bool:
        industry = str(row.get("industry", ""))
        if industry_counts.get(industry, 0) >= max_industry_positions:
            return False
        tactical = is_tactical_tier(row.get("tier", "unknown"))
        if tactical and tactical_count >= max_tactical_positions:
            return False
        if force_new_industry and industry in industry_counts:
            return False
        return True

    core = out[~out["tier"].isin(TACTICAL_TIERS)]
    tactical = out[out["tier"].isin(TACTICAL_TIERS)]

    # First cover different industries with the best eligible core candidate.
    for _, row in core.iterrows():
        if len(chosen) >= min(target_positions, min_distinct_industries):
            break
        if can_take(row, force_new_industry=True):
            chosen.append(row.to_dict())
            industry_counts[str(row["industry"])] = industry_counts.get(str(row["industry"]), 0) + 1

    # Fill remaining slots with core names.
    for _, row in core.iterrows():
        if len(chosen) >= target_positions:
            break
        if any(x["symbol"] == row["symbol"] for x in chosen):
            continue
        if can_take(row):
            chosen.append(row.to_dict())
            industry_counts[str(row["industry"])] = industry_counts.get(str(row["industry"]), 0) + 1

    # Tactical sleeve is deliberately secondary to core selection.
    for _, row in tactical.iterrows():
        if len(chosen) >= target_positions:
            break
        if any(x["symbol"] == row["symbol"] for x in chosen):
            continue
        if can_take(row):
            chosen.append(row.to_dict())
            industry_counts[str(row["industry"])] = industry_counts.get(str(row["industry"]), 0) + 1
            tactical_count += 1

    return pd.DataFrame(chosen)
