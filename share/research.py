from __future__ import annotations

from itertools import product
import pandas as pd

from .backtest import Backtester
from .strategy import StrategyConfig
from .metrics import summarize, walk_forward_splits

DEFAULT_GRID = {
    "lookback_fast": [15, 20, 30],
    "lookback_slow": [60, 90, 120],
    "entry_rsi_low": [40, 45, 50],
    "entry_rsi_high": [68, 72, 75],
    "max_entry_annualized_vol": [0.45, 0.55, 0.65],
    "trail_atr_multiple": [2.5, 3.0, 3.5],
}

def _objective(metrics: dict, hard_dd: float = 0.08) -> tuple:
    dd = abs(metrics.get("max_drawdown", 1.0))
    breach = max(0.0, dd - hard_dd)
    # Capital preservation first, then terminal capital, then stability.
    return (
        breach,
        -metrics.get("final_equity", 0.0),
        dd,
        -metrics.get("sharpe", 0.0),
    )

def grid_train(data: pd.DataFrame, initial_cash: float = 200_000,
               hard_drawdown: float = 0.08,
               base: StrategyConfig | None = None,
               grid: dict | None = None):
    base = base or StrategyConfig()
    grid = grid or DEFAULT_GRID
    results = []

    for fast, slow, rsi_low, rsi_high, max_vol, trail in product(
        grid["lookback_fast"], grid["lookback_slow"],
        grid["entry_rsi_low"], grid["entry_rsi_high"],
        grid["max_entry_annualized_vol"], grid["trail_atr_multiple"]
    ):
        if fast >= slow or rsi_low >= rsi_high:
            continue

        cfg = StrategyConfig(
            max_single_weight=base.max_single_weight,
            max_equity_weight=base.max_equity_weight,
            target_positions=base.target_positions,
            min_history=base.min_history,
            entry_rsi_low=rsi_low,
            entry_rsi_high=rsi_high,
            max_entry_annualized_vol=max_vol,
            min_20d_return=base.min_20d_return,
            min_holding_days=base.min_holding_days,
            trend_break_confirm_days=base.trend_break_confirm_days,
            trail_atr_multiple=trail,
            hold_ma_buffer=base.hold_ma_buffer,
            rebalance_threshold=base.rebalance_threshold,
            lookback_fast=fast,
            lookback_slow=slow,
            atr_window=base.atr_window,
        )
        equity, trades = Backtester(initial_cash=initial_cash, strategy_cfg=cfg).run(data)
        m = summarize(equity, initial_cash)
        results.append({
            "fast": fast,
            "slow": slow,
            "entry_rsi_low": rsi_low,
            "entry_rsi_high": rsi_high,
            "max_entry_annualized_vol": max_vol,
            "trail_atr_multiple": trail,
            "trades": len(trades),
            **m,
        })

    result = pd.DataFrame(results)
    if result.empty:
        raise ValueError("训练窗口没有产生可评估的参数组合")
    result["_objective"] = result.apply(lambda r: _objective(r.to_dict(), hard_drawdown), axis=1)
    return result.sort_values("_objective").drop(columns="_objective").reset_index(drop=True)

def walk_forward_train(data: pd.DataFrame, initial_cash: float = 200_000,
                       train_days: int = 504, test_days: int = 126,
                       purge_days: int = 5, hard_drawdown: float = 0.08):
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(data["date"]).unique()))
    reports = []
    for window_id, (train_dates, test_dates) in enumerate(
        walk_forward_splits(dates, train_days, test_days, purge_days), 1
    ):
        train = data[data["date"].isin(train_dates)]
        test = data[data["date"].isin(test_dates)]
        ranking = grid_train(train, initial_cash, hard_drawdown)
        best = ranking.iloc[0]

        cfg = StrategyConfig(
            entry_rsi_low=float(best["entry_rsi_low"]),
            entry_rsi_high=float(best["entry_rsi_high"]),
            max_entry_annualized_vol=float(best["max_entry_annualized_vol"]),
            trail_atr_multiple=float(best["trail_atr_multiple"]),
            lookback_fast=int(best["fast"]),
            lookback_slow=int(best["slow"]),
        )
        oos_equity, oos_trades = Backtester(initial_cash=initial_cash, strategy_cfg=cfg).run(test)
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
            "selected_max_vol": cfg.max_entry_annualized_vol,
            "selected_trail_atr": cfg.trail_atr_multiple,
            "oos_trades": len(oos_trades),
            **{f"oos_{k}": v for k, v in oos.items()},
        })
    return pd.DataFrame(reports)
