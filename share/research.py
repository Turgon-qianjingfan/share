from __future__ import annotations

from itertools import product
import pandas as pd

from .backtest import Backtester
from .strategy import StrategyConfig
from .metrics import summarize, walk_forward_splits

DEFAULT_GRID = {
    "lookback_fast": [15, 20, 30],
    "lookback_slow": [60, 90, 120],
    "rsi_low": [40, 45, 50],
    "rsi_high": [68, 72, 75],
    "max_annualized_vol": [0.40, 0.50, 0.60],
}

def _objective(metrics: dict, hard_dd: float = 0.08) -> tuple:
    dd = abs(metrics.get("max_drawdown", 1.0))
    breach = max(0.0, dd - hard_dd)
    return (breach, -metrics.get("final_equity", 0.0), dd, -metrics.get("sharpe", 0.0))

def _make_strategy(base: StrategyConfig, params: dict) -> StrategyConfig:
    return StrategyConfig(
        max_single_weight=base.max_single_weight,
        max_equity_weight=base.max_equity_weight,
        target_positions=base.target_positions,
        min_history=base.min_history,
        rsi_low=float(params["rsi_low"]),
        rsi_high=float(params["rsi_high"]),
        max_annualized_vol=float(params["max_annualized_vol"]),
        lookback_fast=int(params["lookback_fast"]),
        lookback_slow=int(params["lookback_slow"]),
        atr_window=base.atr_window,
    )

def grid_train(
    data: pd.DataFrame,
    initial_cash: float = 200_000,
    hard_drawdown: float = 0.08,
    base: StrategyConfig | None = None,
    grid: dict | None = None,
):
    base = base or StrategyConfig()
    grid = grid or DEFAULT_GRID
    results = []
    for fast, slow, low, high, max_vol in product(
        grid["lookback_fast"], grid["lookback_slow"],
        grid["rsi_low"], grid["rsi_high"], grid["max_annualized_vol"]
    ):
        if fast >= slow or low >= high:
            continue
        cfg = _make_strategy(base, {
            "lookback_fast": fast,
            "lookback_slow": slow,
            "rsi_low": low,
            "rsi_high": high,
            "max_annualized_vol": max_vol,
        })
        bt = Backtester(initial_cash=initial_cash, strategy_cfg=cfg)
        equity, trades = bt.run(data)
        m = summarize(equity, initial_cash)
        results.append({
            "fast": fast,
            "slow": slow,
            "rsi_low": low,
            "rsi_high": high,
            "max_annualized_vol": max_vol,
            "trades": len(trades),
            **m,
        })
    result = pd.DataFrame(results)
    if result.empty:
        raise ValueError("训练窗口没有产生可评估的参数组合")
    result["_objective"] = result.apply(lambda r: _objective(r.to_dict(), hard_drawdown), axis=1)
    return result.sort_values("_objective").drop(columns="_objective").reset_index(drop=True)

def walk_forward_train(
    data: pd.DataFrame,
    initial_cash: float = 200_000,
    train_days: int = 504,
    test_days: int = 126,
    purge_days: int = 5,
    hard_drawdown: float = 0.08,
):
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(data["date"]).unique()))
    splits = walk_forward_splits(dates, train_days, test_days, purge_days)
    reports = []
    for window_id, (train_dates, test_dates) in enumerate(splits, 1):
        train = data[data["date"].isin(train_dates)]
        test = data[data["date"].isin(test_dates)]
        ranking = grid_train(train, initial_cash, hard_drawdown)
        best = ranking.iloc[0]
        cfg = StrategyConfig(
            rsi_low=float(best["rsi_low"]),
            rsi_high=float(best["rsi_high"]),
            max_annualized_vol=float(best["max_annualized_vol"]),
            lookback_fast=int(best["fast"]),
            lookback_slow=int(best["slow"]),
        )
        bt = Backtester(initial_cash=initial_cash, strategy_cfg=cfg)
        oos_equity, oos_trades = bt.run(test)
        oos = summarize(oos_equity, initial_cash)
        reports.append({
            "window": window_id,
            "train_start": str(train_dates[0].date()),
            "train_end": str(train_dates[-1].date()),
            "test_start": str(test_dates[0].date()),
            "test_end": str(test_dates[-1].date()),
            "selected_fast": cfg.lookback_fast,
            "selected_slow": cfg.lookback_slow,
            "selected_rsi_low": cfg.rsi_low,
            "selected_rsi_high": cfg.rsi_high,
            "selected_max_vol": cfg.max_annualized_vol,
            "oos_trades": len(oos_trades),
            **{f"oos_{k}": v for k, v in oos.items()},
        })
    return pd.DataFrame(reports)
