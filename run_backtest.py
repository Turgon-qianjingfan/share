from __future__ import annotations
import argparse
import pandas as pd
from share.backtest import run_csv

def demo_data():
    dates = pd.bdate_range("2023-01-03", periods=180)
    rows = []
    for j, symbol in enumerate(["600000.SH","600519.SH","000001.SZ","000858.SZ","601318.SH","000333.SZ","002594.SZ","600036.SH"]):
        base = 20 + j * 7
        for i, d in enumerate(dates):
            close = base * (1 + 0.0005*i + 0.02*__import__("math").sin(i/14+j))
            rows.append({"date":d,"symbol":symbol,"open":close*0.998,"high":close*1.01,
                         "low":close*0.99,"close":close,"volume":1_000_000+j*100_000})
    return pd.DataFrame(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv")
    p.add_argument("--initial-cash", type=float, default=200_000)
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    data = demo_data() if args.demo else None
    if not args.demo and not args.csv:
        p.error("请指定 --csv 或 --demo")
    if args.csv:
        from share.backtest import Backtester
        from share.data import load_csv
        equity, trades = Backtester(args.initial_cash).run(load_csv(args.csv))
    else:
        from share.backtest import Backtester
        equity, trades = Backtester(args.initial_cash).run(data)
    from share.metrics import summarize
    print(summarize(equity, args.initial_cash))
    print(f"trades={len(trades)}")
if __name__ == "__main__":
    main()
