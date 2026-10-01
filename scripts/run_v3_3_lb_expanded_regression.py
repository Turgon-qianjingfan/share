from __future__ import annotations

from pathlib import Path

import pandas as pd

from share.backtest import Backtester
from share.data import load_csv
from share.metrics import summarize
from share.strategy import StrategyConfig

DATA_FILES = [
    Path("data/demo/v3_3_lb_16stocks_a.csv"),
    Path("data/demo/v3_3_lb_16stocks_b.csv"),
    Path("data/demo/v3_3_lb_2stocks_tactical.csv"),
]
BENCHMARK_FILE = Path("data/demo/v3_2_14stocks_lb_1000d.csv")
BENCHMARK = "510300.SH"
PROFILES = Path("data/demo/v3_3_34stock_profiles.csv")
EVENTS = Path("data/events/demo_company_events.csv")


def load_expanded():
    raw = pd.concat([load_csv(p) for p in DATA_FILES], ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"], utc=True).dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw[(raw[["open", "high", "low", "close"]] > 0).all(axis=1)]
    raw = raw.drop_duplicates(["date", "symbol"], keep="first")

    profiles = pd.read_csv(PROFILES)
    industry_map = dict(zip(profiles["symbol"], profiles["industry"]))
    raw["industry"] = raw["symbol"].map(industry_map).fillna("其他")

    bench_all = load_csv(BENCHMARK_FILE)
    bench_all["date"] = pd.to_datetime(bench_all["date"], utc=True).dt.tz_localize(None)
    benchmark = bench_all[bench_all["symbol"] == BENCHMARK].copy()
    return raw, benchmark, profiles


def run_case(name, data, benchmark, profiles, events):
    cfg = StrategyConfig()
    bt = Backtester(initial_cash=200_000, strategy_cfg=cfg)
    equity, trades = bt.run(
        data,
        benchmark=benchmark,
        events=events,
        stock_profiles=profiles,
    )
    metrics = summarize(equity, 200_000)
    buys = trades[trades["side"] == "BUY"] if not trades.empty else trades
    if not buys.empty:
        tactical = int((buys["reason"] == "v3_tactical_entry").sum())
        core = int((buys["reason"] == "v3_core_entry").sum())
    else:
        tactical = core = 0
    stats = {
        "buy_trades": int(len(buys)),
        "sell_trades": int((trades["side"] == "SELL").sum()) if not trades.empty else 0,
        "core_buys": core,
        "tactical_buys": tactical,
        "exit_reasons": trades["reason"].value_counts().to_dict() if not trades.empty else {},
    }
    return name, equity, trades, metrics, stats


def main():
    data, benchmark, profiles = load_expanded()
    events = pd.read_csv(EVENTS) if EVENTS.exists() else None

    print("=== V3.3 Expanded Longbridge / 34-stock / 1,000-day regression ===")
    print(f"range={data['date'].min().date()}..{data['date'].max().date()}")
    print(f"stocks={data['symbol'].nunique()} bars={len(data)} benchmark_bars={len(benchmark)}")
    print(f"profiles={profiles['tier'].value_counts().to_dict()}")
    print(f"industries={data['industry'].nunique()}")

    for name, ev in [
        ("V3.3 动态龙头+防守底仓+连板事件", events),
        ("V3.3 动态龙头+防守底仓（无新闻）", None),
    ]:
        _, eq, trades, metrics, stats = run_case(name, data, benchmark, profiles, ev)
        print(f"\n[{name}]")
        for k, v in metrics.items():
            print(f"{k}={v}")
        exposure = ((eq["equity"] - eq["cash"]) / eq["equity"]).mean() if not eq.empty else 0.0
        print(f"avg_equity_exposure={exposure:.6f}")
        print(f"trade_stats={stats}")

    close = data.pivot(index="date", columns="symbol", values="close").sort_index()
    first = close.iloc[0].dropna()
    valid = first.index
    per_stock = 0.40 / len(valid)
    units = (200_000 * per_stock) / first
    bh40 = (close[valid] * units).sum(axis=1) + 200_000 * 0.60
    bh40_df = pd.DataFrame({"date": bh40.index, "equity": bh40.values, "cash": 120_000.0})
    print("\n[基准：34股40%等权买入持有]")
    for k, v in summarize(bh40_df, 200_000).items():
        print(f"{k}={v}")

    per_stock = 0.50 / len(valid)
    units = (200_000 * per_stock) / first
    bh50 = (close[valid] * units).sum(axis=1) + 200_000 * 0.50
    bh50_df = pd.DataFrame({"date": bh50.index, "equity": bh50.values, "cash": 100_000.0})
    print("\n[基准：34股50%等权买入持有]")
    for k, v in summarize(bh50_df, 200_000).items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
