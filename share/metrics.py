from __future__ import annotations

import numpy as np
import pandas as pd


def summarize(equity: pd.DataFrame, initial_cash: float) -> dict:
    if equity.empty:
        return {
            "initial_cash": float(initial_cash),
            "final_equity": float(initial_cash),
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "trading_days": 0,
            "principal_preserved": True,
            "minimum_equity": float(initial_cash),
            "minimum_equity_vs_principal": 0.0,
        }

    s = equity.set_index("date")["equity"].astype(float).sort_index()
    values = s.to_numpy()

    # The initial capital is a real starting peak. Including it prevents the
    # first losing observations from being omitted from maximum drawdown.
    running_peaks = np.maximum.accumulate(np.r_[float(initial_cash), values])[1:]
    drawdown = values / running_peaks - 1.0
    daily = s.pct_change().dropna()
    daily_std = daily.std()
    sharpe = float(daily.mean() / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0

    years = max((s.index[-1] - s.index[0]).days / 365.25, 1 / 365.25)
    final_equity = float(s.iloc[-1])
    cagr = (final_equity / initial_cash) ** (1 / years) - 1 if initial_cash > 0 else 0.0
    minimum_equity = float(min(initial_cash, s.min()))

    return {
        "initial_cash": float(initial_cash),
        "final_equity": final_equity,
        "total_return": float(final_equity / initial_cash - 1),
        "cagr": float(cagr),
        "max_drawdown": float(drawdown.min()),
        "sharpe": sharpe,
        "trading_days": int(len(s)),
        "principal_preserved": bool(final_equity >= initial_cash),
        "minimum_equity": minimum_equity,
        "minimum_equity_vs_principal": float(minimum_equity / initial_cash - 1),
    }


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
