from __future__ import annotations

from share.data import load_csv
from share.backtest import Backtester
from share.metrics import summarize

DATA = "data/demo/diversified_10stocks_1000d.csv"

def main():
    data = load_csv(DATA)
    equity, trades = Backtester(initial_cash=200_000).run(data)
    metrics = summarize(equity, 200_000)
    print("=== DIVERSIFIED 10-STOCK V2 TEST ===")
    print(f"data_rows={len(data)}")
    print(f"symbols={data['symbol'].nunique()}")
    print(f"range={data['date'].min().date()}..{data['date'].max().date()}")
    for k, v in metrics.items():
        print(f"{k}={v}")
    print(f"trade_count={len(trades)}")
    if not trades.empty:
        print(trades.to_string(index=False))

if __name__ == "__main__":
    main()
