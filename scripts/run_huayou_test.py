from __future__ import annotations

import pandas as pd

from share.backtest import Backtester
from share.data import load_csv
from share.metrics import summarize
from share.strategy import StrategyConfig

PATH="data/demo/603799_SH_1000d.csv"
INITIAL=200_000

def buy_and_hold(df: pd.DataFrame, initial_cash: float) -> dict:
    d=df.sort_values("date")
    first=float(d.iloc[0].open)
    last=float(d.iloc[-1].close)
    qty=int(initial_cash / first / 100) * 100
    cash=initial_cash-qty*first
    final=cash+qty*last
    return {
        "final_equity": final,
        "total_return": final/initial_cash-1,
        "max_drawdown_approx": 1.0-(d["close"]/d["close"].cummax()).min(),
        "shares": qty,
    }

def main():
    df=load_csv(PATH)
    bt=Backtester(
        initial_cash=INITIAL,
        strategy_cfg=StrategyConfig(
            max_single_weight=0.08,
            max_equity_weight=0.40,
            target_positions=5,
            lookback_fast=20,
            lookback_slow=60,
            rsi_low=45,
            rsi_high=72,
            max_annualized_vol=0.55,
        )
    )
    equity,trades=bt.run(df)
    m=summarize(equity,INITIAL)
    bh=buy_and_hold(df,INITIAL)

    print("=== HUAYOU COBALT 603799.SH SINGLE-STOCK TEST ===")
    print(f"data_rows={len(df)}")
    print(f"data_range={df.date.min().date()}..{df.date.max().date()}")
    for k,v in m.items(): print(f"strategy_{k}={v}")
    print(f"trade_count={len(trades)}")
    print(f"buy_and_hold={bh}")
    if not trades.empty:
        print("last_trades:")
        print(trades.tail(10).to_string(index=False))

if __name__=="__main__":
    main()
