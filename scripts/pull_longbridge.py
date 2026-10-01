from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

def read_symbols(path: str) -> list[str]:
    return [x.strip() for x in Path(path).read_text().splitlines() if x.strip() and not x.startswith("#")]

def fetch(symbol: str, start: str, end: str) -> list[dict]:
    cmd = ["longbridge","kline","history",symbol,"--start",start,"--end",end,"--period","day","--format","json"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{symbol}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--universe",default="data/universe_current.txt")
    p.add_argument("--start",default="2021-10-01")
    p.add_argument("--end",default="2026-10-01")
    p.add_argument("--out",default="data/a_share_5y.csv")
    args=p.parse_args()

    import csv
    rows=[]
    for symbol in read_symbols(args.universe):
        candles=fetch(symbol,args.start,args.end)
        for x in candles:
            rows.append({
                "date": x["time"][:10] if "time" in x else x.get("timestamp","")[:10],
                "symbol": symbol,
                "open": x["open"], "high": x["high"], "low": x["low"],
                "close": x["close"], "volume": x["volume"],
            })
        print(f"{symbol}: {len(candles)} rows")
    if not rows:
        raise RuntimeError("没有取得任何历史数据；不要用空数据生成训练集。")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["date","symbol","open","high","low","close","volume"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")

if __name__ == "__main__":
    main()
