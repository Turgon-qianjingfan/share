from __future__ import annotations
import pandas as pd
import numpy as np

def summarize(equity: pd.DataFrame, initial_cash: float) -> dict:
    if equity.empty:
        return {"initial_cash": initial_cash, "final_equity": initial_cash, "total_return": 0.0, "max_drawdown": 0.0}
    s = equity.set_index("date")["equity"].astype(float)
    total_return = s.iloc[-1] / initial_cash - 1
    peak = s.cummax()
    drawdown = s / peak - 1
    daily = s.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * (252 ** 0.5)) if daily.std() > 0 else 0.0
    years = max((s.index[-1] - s.index[0]).days / 365.25, 1/365.25)
    cagr = (s.iloc[-1] / initial_cash) ** (1/years) - 1
    return {"initial_cash": initial_cash, "final_equity": float(s.iloc[-1]),
            "total_return": float(total_return), "cagr": float(cagr),
            "max_drawdown": float(drawdown.min()), "sharpe": sharpe,
            "trading_days": int(len(s))}
