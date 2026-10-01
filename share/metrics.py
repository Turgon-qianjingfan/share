from __future__ import annotations
import pandas as pd
import numpy as np

def summarize(equity: pd.DataFrame, initial_cash: float) -> dict:
    if equity.empty:
        return {"initial_cash": initial_cash, "final_equity": initial_cash, "total_return": 0.0,
                "max_drawdown": 0.0, "principal_preserved": True}
    s = equity.set_index("date")["equity"].astype(float)
    peak = s.cummax()
    drawdown = s / peak - 1
    daily = s.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    years = max((s.index[-1] - s.index[0]).days / 365.25, 1/365.25)
    cagr = (s.iloc[-1] / initial_cash) ** (1 / years) - 1
    return {"initial_cash": float(initial_cash), "final_equity": float(s.iloc[-1]),
            "total_return": float(s.iloc[-1] / initial_cash - 1), "cagr": float(cagr),
            "max_drawdown": float(drawdown.min()), "sharpe": sharpe,
            "trading_days": int(len(s)), "principal_preserved": bool(s.iloc[-1] >= initial_cash),
            "minimum_equity": float(s.min()),
            "minimum_equity_vs_principal": float(s.min() / initial_cash - 1)}

def walk_forward_splits(dates, train_days=504, test_days=126, purge_days=5):
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
    splits, start = [], 0
    while start + train_days + purge_days + test_days <= len(dates):
        train = dates[start:start + train_days]
        test_start = start + train_days + purge_days
        test = dates[test_start:test_start + test_days]
        splits.append((train, test))
        start += test_days
    return splits
