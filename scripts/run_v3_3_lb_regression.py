from __future__ import annotations

import pandas as pd

from share.backtest import Backtester
from share.data import load_csv
from share.metrics import summarize
from share.strategy import StrategyConfig

DATA_PATHS = [
    "data/demo/v3_3_lb_16stocks_a.csv",
    "data/demo/v3_3_lb_16stocks_b.csv",
    "data/demo/v3_3_lb_2stocks_tactical.csv",
]
BENCHMARK_FILE = "data/demo/v3_2_14stocks_lb_1000d.csv"
BENCHMARK = "510300.SH"
PROFILES = "data/demo/v3_3_34stock_profiles.csv"
EVENTS = "data/events/demo_company_events.csv"


def load_expanded_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [load_csv(p) for p in DATA_PATHS]
    data = pd.concat(frames, ignore_index=True)
    bench_all = load_csv(BENCHMARK_FILE)
    benchmark = bench_all[bench_all["symbol"] == BENCHMARK].copy()
    return data, benchmark


def run_case(name: str, data: pd.DataFrame, benchmark: pd.DataFrame, profiles, events):
    cfg = StrategyConfig()
    bt = Backtester(initial_cash=200_000, strategy_cfg=cfg)
    equity, trades = bt.run(
        data,
        benchmark=benchmark,
        events=events,
        stock_profiles=profiles,
    )
    metrics = summarize(equity, 200_000)
    return name, equity, trades, metrics


def trade_stats(trades: pd.DataFrame, profiles: pd.DataFrame) -> dict:
    if trades.empty:
        return {"buy_trades": 0, "sell_trades": 0, "tactical_buys": 0, "core_buys": 0, "top_exit_reasons": {}}
    tier = profiles.set_index("symbol")["tier"].to_dict()
    buys = trades[trades["side"] == "BUY"].copy()
    tactical_buys = int(buys["symbol"].map(tier).isin({"tactical", "small", "micro"}).sum())
    return {
        "buy_trades": int((trades["side"] == "BUY").sum()),
        "sell_trades": int((trades["side"] == "SELL").sum()),
        "tactical_buys": tactical_buys,
        "core_buys": int(len(buys) - tactical_buys),
        "top_exit_reasons": trades["reason"].value_counts().to_dict(),
    }


def main():
    data, benchmark = load_expanded_data()
    profiles = pd.read_csv(PROFILES)
    events = pd.read_csv(EVENTS)

    # Use profile industries as the single source of truth for this research run.
    industry_map = dict(zip(profiles["symbol"], profiles["industry"]))
    data["industry"] = data["symbol"].map(industry_map)

    print("=== Longbridge 34-stock / 1,000-trading-day regression ===")
    print(f"range={data['date'].min().date()}..{data['date'].max().date()}")
    print(f"stocks={data['symbol'].nunique()} bars={len(data)} benchmark_bars={len(benchmark)}")
    print(f"profile_tiers={profiles['tier'].value_counts().to_dict()}")

    for name, d, b, p, e in [
        ("V3.3 扩大股票池+龙头动态识别+新闻", data, benchmark, profiles, events),
        ("V3.3 扩大股票池+龙头动态识别（无新闻）", data, benchmark, profiles, None),
        ("V3.3 无固定画像+动态龙头", data, benchmark, None, events),
    ]:
        n, eq, trades, m = run_case(name, d, b, p, e)
        print(f"\n[{n}]")
        for k, v in m.items():
            print(f"{k}={v}")
        if not eq.empty:
            exposure = ((eq["equity"] - eq["cash"]) / eq["equity"]).mean()
            print(f"avg_equity_exposure={exposure:.6f}")
        print("trade_stats=", trade_stats(trades, profiles))

    close = data.pivot(index="date", columns="symbol", values="close").sort_index()
    first = close.iloc[0]
    valid = first.dropna().index
    per_stock = 0.40 / len(valid)
    units = (200_000 * per_stock) / first[valid]
    bh40 = (close[valid] * units).sum(axis=1) + 200_000 * 0.60
    bh40_df = pd.DataFrame({"date": bh40.index, "equity": bh40.values, "cash": 120_000.0})
    print("\n[基准：34股40%等权买入持有]")
    for k, v in summarize(bh40_df, 200_000).items():
        print(f"{k}={v}")

    per_stock50 = 0.50 / len(valid)
    units50 = (200_000 * per_stock50) / first[valid]
    bh50 = (close[valid] * units50).sum(axis=1) + 200_000 * 0.50
    bh50_df = pd.DataFrame({"date": bh50.index, "equity": bh50.values, "cash": 100_000.0})
    print("\n[基准：34股50%等权买入持有]")
    for k, v in summarize(bh50_df, 200_000).items():
        print(f"{k}={v}")

    bclose = benchmark.set_index("date")["close"].sort_index()
    units_b = 200_000 / bclose.iloc[0]
    market = pd.DataFrame({"date": bclose.index, "equity": bclose.values * units_b, "cash": 0.0})
    print("\n[参考：510300.SH 100%买入持有]")
    for k, v in summarize(market, 200_000).items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
