from __future__ import annotations

import argparse
import pandas as pd

from share.data import load_csv

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",required=True)
    p.add_argument("--start",default="2021-10-01")
    p.add_argument("--end",default="2026-10-01")
    p.add_argument("--min-days",type=int,default=1000)
    args=p.parse_args()

    df=load_csv(args.csv)
    start=pd.Timestamp(args.start); end=pd.Timestamp(args.end)
    actual_start=df["date"].min(); actual_end=df["date"].max()
    errors=[]
    if actual_start > start:
        errors.append(f"整体起始日不足: {actual_start.date()} > {start.date()}")
    if actual_end < end:
        errors.append(f"整体结束日不足: {actual_end.date()} < {end.date()}")
    counts=df.groupby("symbol")["date"].nunique()
    too_short=counts[counts < args.min_days]
    if not too_short.empty:
        errors.append(f"历史长度不足 {args.min_days} 个交易日的股票数: {len(too_short)}")
    print(f"rows={len(df)} symbols={df.symbol.nunique()} range={actual_start.date()}..{actual_end.date()}")
    if errors:
        for e in errors: print("FAIL:",e)
        raise SystemExit(2)
    print("PASS: dataset meets the configured five-year coverage checks")

if __name__ == "__main__":
    main()
