from __future__ import annotations

from pathlib import Path

import pandas as pd

from share.backtest import Backtester
from share.data import load_csv
from share.metrics import summarize
from share.strategy import StrategyConfig

BASES = [
    Path("data/demo/v3_3_lb_16stocks_a.csv"),
    Path("data/demo/v3_3_lb_16stocks_b.csv"),
    Path("data/demo/v3_3_lb_2stocks_tactical.csv"),
]
BENCHMARK_FILE = Path("data/demo/v3_2_14stocks_lb_1000d.csv")
BENCHMARK = "510300.SH"
PROFILES = Path("data/demo/v3_3_34stock_profiles.csv")

def run_case(name, data, benchmark, events=None):
    cfg = StrategyConfig()
    bt = Backtester(initial_cash=200_000, strategy_cfg=cfg)
    equity, trades = bt.run(
        data,
        benchmark=benchmark,
        events=events,
        industry_map=INDUSTRY,
        stock_profiles=None,
    )
    metrics = summarize(equity, 200_000)
    buys = trades[trades["side"] == "BUY"] if not trades.empty else trades
    sell_reasons = trades["reason"].value_counts().to_dict() if not trades.empty else {}
    return name, metrics, equity, trades, buys, sell_reasons

def main():
    data, benchmark = load_expanded()
    events_path = Path("data/events/demo_company_events.csv")
    events = pd.read_csv(events_path) if events_path.exists() else None

    print("=== V3.3 Expanded Longbridge / 1,000-trading-day regression ===")
    print(f"range={data['date'].min().date()}..{data['date'].max().date()}")
    print(f"stocks={data['symbol'].nunique()} bars={len(data)} benchmark_bars={len(benchmark)}")
    print(f"industries={data['industry'].nunique()}")

    cases = [
        ("V3.3 动态龙头+防守底仓+连板事件", events),
        ("V3.3 无新闻（连板新闻门槛关闭前的技术基线）", None),
    ]

    for name, ev in cases:
        if ev is None and "新闻" in name:
            # Run a deliberately comparable no-event variant.
            pass
        result = run_case(name, data, benchmark, ev)
        n, m, eq, trades, buys, reasons = result
        print(f"\n[{n}]")
        for k, v in m.items():
            print(f"{k}={v}")
        if not buys.empty:
            tactical = int((buys["reason"] == "v3_tactical_entry").sum()) if "reason" in buys.columns else 0
            core = int((buys["reason"] == "v3_core_entry").sum()) if "reason" in buys.columns else 0
            print(f"core_buys={core} tactical_buys={tactical}")
            print("buy_symbols=", buys["symbol"].value_counts().head(15).to_dict())
        print("exit_reasons=", reasons)

    close = data.pivot(index="date", columns="symbol", values="close").sort_index()
    first = close.iloc[0]
    valid = first.dropna().index
    per_stock = 0.40 / len(valid)
    units = (200_000 * per_stock) / first[valid]
    bh = (close[valid] * units).sum(axis=1) + 200_000 * 0.60
    bh_df = pd.DataFrame({"date": bh.index, "equity": bh.values, "cash": 120_000.0})
    print("\n[基准：40%等权买入持有 expanded universe]")
    for k, v in summarize(bh_df, 200_000).items():
        print(f"{k}={v}")

if __name__ == "__main__":
    main()def load_expanded():
    frames = [load_csv(p) for p in BASES]
    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"], utc=True).dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw[(raw[["open", "high", "low", "close"]] > 0).all(axis=1)].copy()
    raw = raw.drop_duplicates(["date", "symbol"], keep="first")
    profiles = pd.read_csv(PROFILES)
    industry_map = dict(zip(profiles["symbol"], profiles["industry"]))
    raw["industry"] = raw["symbol"].map(industry_map).fillna("其他")

    bench_all = load_csv(BENCHMARK_FILE)
    benchmark = bench_all[bench_all["symbol"] == BENCHMARK].copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"], utc=True).dt.tz_localize(None)
    data = raw[raw["symbol"] != BENCHMARK].copy()
    return data, benchmark, profiles

def run_case(name, data, benchmark, events=None):
    cfg = StrategyConfig()
    bt = Backtester(initial_cash=200_000, strategy_cfg=cfg)
    equity, trades = bt.run(
        data,
        benchmark=benchmark,
        events=events,
        industry_map=INDUSTRY,
        stock_profiles=None,
    )
    metrics = summarize(equity, 200_000)
    buys = trades[trades["side"] == "BUY"] if not trades.empty else trades
    sell_reasons = trades["reason"].value_counts().to_dict() if not trades.empty else {}
    return name, metrics, equity, trades, buys, sell_reasons

def main():
    data, benchmark = load_expanded()
    events_path = Path("data/events/demo_company_events.csv")
    events = pd.read_csv(events_path) if events_path.exists() else None

    print("=== V3.3 Expanded Longbridge / 1,000-trading-day regression ===")
    print(f"range={data['date'].min().date()}..{data['date'].max().date()}")
    print(f"stocks={data['symbol'].nunique()} bars={len(data)} benchmark_bars={len(benchmark)}")
    print(f"industries={data['industry'].nunique()}")

    cases = [
        ("V3.3 动态龙头+防守底仓+连板事件", events),
        ("V3.3 无新闻（连板新闻门槛关闭前的技术基线）", None),
    ]

    for name, ev in cases:
        if ev is None and "新闻" in name:
            # Run a deliberately comparable no-event variant.
            pass
        result = run_case(name, data, benchmark, ev)
        n, m, eq, trades, buys, reasons = result
        print(f"\n[{n}]")
        for k, v in m.items():
            print(f"{k}={v}")
        if not buys.empty:
            tactical = int((buys["reason"] == "v3_tactical_entry").sum()) if "reason" in buys.columns else 0
            core = int((buys["reason"] == "v3_core_entry").sum()) if "reason" in buys.columns else 0
            print(f"core_buys={core} tactical_buys={tactical}")
            print("buy_symbols=", buys["symbol"].value_counts().head(15).to_dict())
        print("exit_reasons=", reasons)

    close = data.pivot(index="date", columns="symbol", values="close").sort_index()
    first = close.iloc[0]
    valid = first.dropna().index
    per_stock = 0.40 / len(valid)
    units = (200_000 * per_stock) / first[valid]
    bh = (close[valid] * units).sum(axis=1) + 200_000 * 0.60
    bh_df = pd.DataFrame({"date": bh.index, "equity": bh.values, "cash": 120_000.0})
    print("\n[基准：40%等权买入持有 expanded universe]")
    for k, v in summarize(bh_df, 200_000).items():
        print(f"{k}={v}")

if __name__ == "__main__":
    main()
