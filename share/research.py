from __future__ import annotations

from itertools import product
import pandas as pd

from .backtest import Backtester
from .strategy import StrategyConfig
from .metrics import summarize, walk_forward_splits


DEFAULT_GRID = {
    "lookback_fast": [15, 20, 30],
    "lookback_slow": [60, 90, 120],
    "entry_rsi_low": [40, 45],
    "entry_rsi_high": [72, 78],
    "min_adx": [15, 20],
    "trail_atr_multiple": [2.8, 3.2, 3.6],
    "technical_weight": [0.60, 0.65, 0.70],
    "event_weight": [0.15, 0.20, 0.25],
    "industry_weight": [0.10, 0.15, 0.20],
    "minimum_entry_score": [0.50, 0.55, 0.60],
    "reentry_cooldown_days": [10, 15, 20],
}


def _objective(metrics: dict, hard_dd: float = 0.08) -> tuple:
    dd = abs(float(metrics.get("max_drawdown", 1.0)))
    breach = max(0.0, dd - hard_dd)
    # Capital preservation first; return and stability break ties.
    return (
        breach,
        -float(metrics.get("final_equity", 0.0)),
        dd,
        -float(metrics.get("sharpe", 0.0)),
    )


def grid_train(
    data: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    industry_map: dict[str, str] | None = None,
    initial_cash: float = 200_000,
    hard_drawdown: float = 0.08,
    grid: dict | None = None,
):
    grid = grid or DEFAULT_GRID
    results = []

    weight_sum_ok = lambda tw, ew, iw: abs(tw + ew + iw - 1.0) < 1e-9

    for fast, slow, rsi_low, rsi_high, min_adx, trail, tw, ew, iw in product(
        grid["lookback_fast"], grid["lookback_slow"],
        grid["entry_rsi_low"], grid["entry_rsi_high"],
        grid["min_adx"], grid["trail_atr_multiple"],
        grid["technical_weight"], grid["event_weight"], grid["industry_weight"],
        grid["minimum_entry_score"], grid["reentry_cooldown_days"]
    ):
        if fast >= slow or rsi_low >= rsi_high or not weight_sum_ok(tw, ew, iw):
            continue

        cfg = StrategyConfig(
            lookback_fast=fast,
            lookback_slow=slow,
            entry_rsi_low=rsi_low,
            entry_rsi_high=rsi_high,
            min_adx=min_adx,
            trail_atr_multiple=trail,
            technical_weight=tw,
            event_weight=ew,
            industry_weight=iw,
            minimum_entry_score=min_score,
            reentry_cooldown_days=int(reentry_days),
        )
        equity, trades = Backtester(
            initial_cash=initial_cash,
            strategy_cfg=cfg,
            hard_drawdown_limit=hard_drawdown,
        ).run(
            data, benchmark=benchmark, events=events, industry_map=industry_map
        )
        m = summarize(equity, initial_cash)
        results.append({
            "fast": fast,
            "slow": slow,
            "entry_rsi_low": rsi_low,
            "entry_rsi_high": rsi_high,
            "min_adx": min_adx,
            "trail_atr_multiple": trail,
            "technical_weight": tw,
            "event_weight": ew,
            "industry_weight": iw,
            "minimum_entry_score": min_score,
            "reentry_cooldown_days": int(reentry_days),
            "trades": len(trades),
            **m,
        })

    result = pd.DataFrame(results)
    if result.empty:
        raise ValueError("训练窗口没有产生有效参数组合")
    result["_objective"] = result.apply(
        lambda row: _objective(row.to_dict(), hard_drawdown), axis=1
    )
    return result.sort_values("_objective").drop(columns="_objective").reset_index(drop=True)


def walk_forward_train(
    data: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    industry_map: dict[str, str] | None = None,
    initial_cash: float = 200_000,
    train_days: int = 504,
    test_days: int = 126,
    purge_days: int = 5,
    hard_drawdown: float = 0.08,
):
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(data["date"]).unique()))
    reports = []

    for window_id, (train_dates, test_dates) in enumerate(
        walk_forward_splits(dates, train_days, test_days, purge_days), 1
    ):
        train = data[data["date"].isin(train_dates)]
        test = data[data["date"].isin(test_dates)]

        train_benchmark = benchmark[benchmark["date"].isin(train_dates)] if benchmark is not None else None
        test_benchmark = benchmark[benchmark["date"].isin(test_dates)] if benchmark is not None else None

        # Events are point-in-time: train may only see events <= its own dates.
        train_events = events[events["event_time"] <= pd.Timestamp(train_dates[-1], tz="UTC")] if events is not None and not events.empty else events
        test_events = events[events["event_time"] <= pd.Timestamp(test_dates[-1], tz="UTC")] if events is not None and not events.empty else events

        ranking = grid_train(
            train,
            benchmark=train_benchmark,
            events=train_events,
            industry_map=industry_map,
            initial_cash=initial_cash,
            hard_drawdown=hard_drawdown,
        )
        best = ranking.iloc[0]

        cfg = StrategyConfig(
            lookback_fast=int(best["fast"]),
            lookback_slow=int(best["slow"]),
            entry_rsi_low=float(best["entry_rsi_low"]),
            entry_rsi_high=float(best["entry_rsi_high"]),
            min_adx=float(best["min_adx"]),
            trail_atr_multiple=float(best["trail_atr_multiple"]),
            technical_weight=float(best["technical_weight"]),
            event_weight=float(best["event_weight"]),
            industry_weight=float(best["industry_weight"]),
        )
        oos_equity, oos_trades = Backtester(
            initial_cash=initial_cash,
            strategy_cfg=cfg,
            hard_drawdown_limit=hard_drawdown,
        ).run(
            test,
            benchmark=test_benchmark,
            events=test_events,
            industry_map=industry_map,
        )
        oos = summarize(oos_equity, initial_cash)

        reports.append({
            "window": window_id,
            "train_start": str(train_dates[0].date()),
            "train_end": str(train_dates[-1].date()),
            "test_start": str(test_dates[0].date()),
            "test_end": str(test_dates[-1].date()),
            "selected_fast": cfg.lookback_fast,
            "selected_slow": cfg.lookback_slow,
            "selected_rsi_low": cfg.entry_rsi_low,
            "selected_rsi_high": cfg.entry_rsi_high,
            "selected_min_adx": cfg.min_adx,
            "selected_trail_atr": cfg.trail_atr_multiple,
            "selected_technical_weight": cfg.technical_weight,
            "selected_event_weight": cfg.event_weight,
            "selected_industry_weight": cfg.industry_weight,
            "selected_minimum_entry_score": cfg.minimum_entry_score,
            "selected_reentry_cooldown_days": cfg.reentry_cooldown_days,
            "oos_trades": len(oos_trades),
            **{f"oos_{k}": v for k, v in oos.items()},
        })

    return pd.DataFrame(reports)
