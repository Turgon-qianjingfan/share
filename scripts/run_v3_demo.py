from __future__ import annotations

import pandas as pd

from share.data import load_csv
from share.backtest import Backtester
from share.metrics import summarize

DATA="data/demo/v3_diversified_10stocks.csv"
BENCHMARK="data/demo/510300_SH_1000d.csv"
EVENTS="data/events/demo_company_events.csv"
INDUSTRIES="data/industry_map.csv"

def main():
    data=load_csv(DATA)
    benchmark=load_csv(BENCHMARK)
    events=pd.read_csv(EVENTS)
    mapping=pd.read_csv(INDUSTRIES)
    industry_map=dict(zip(mapping["symbol"],mapping["industry"]))

    equity,trades=Backtester(initial_cash=200_000).run(
        data, benchmark=benchmark, events=events, industry_map=industry_map
    )
    metrics=summarize(equity,200_000)

    print("=== STRATEGY V3 DIVERSIFIED TEST ===")
    print(f"rows={len(data)} symbols={data['symbol'].nunique()}")
    print(f"range={data['date'].min().date()}..{data['date'].max().date()}")
    for k,v in metrics.items():
        print(f"{k}={v}")
    print(f"trades={len(trades)}")
    if not equity.empty:
        print(f"avg_risk_budget={equity['risk_budget'].mean():.6f}")
        print(f"avg_breadth={equity['breadth'].mean():.6f}")
        print("market_regime_counts=")
        print(equity["market_regime"].value_counts().to_string())

if __name__=="__main__":
    main()
