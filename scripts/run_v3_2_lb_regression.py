from __future__ import annotations

import pandas as pd

from share.backtest import Backtester
from share.data import load_csv
from share.metrics import summarize
from share.strategy import StrategyConfig

DATA = "data/demo/v3_2_14stocks_lb_1000d.csv"
BENCHMARK = "510300.SH"
PROFILES = "data/demo/v3_2_test_profiles.csv"
INDUSTRY = {
    "600036.SH": "银行",
    "601318.SH": "保险",
    "600519.SH": "食品饮料",
    "000333.SZ": "家电",
    "600276.SH": "医药",
    "002475.SZ": "电子",
    "002371.SZ": "半导体设备",
    "300750.SZ": "新能源电池",
    "600938.SH": "能源",
    "601899.SH": "有色",
    "600988.SH": "黄金",
    "601100.SH": "工业机械",
    "603799.SH": "有色",
    "603011.SH": "工业机械",
}
EVENTS = "data/events/demo_company_events.csv"


def run_case(name, data, benchmark, profiles, events):
    cfg = StrategyConfig()
    bt = Backtester(initial_cash=200_000, strategy_cfg=cfg)
    equity, trades = bt.run(
        data,
        benchmark=benchmark,
        events=events,
        industry_map=INDUSTRY,
        stock_profiles=profiles,
    )
    metrics = summarize(equity, 200_000)
    return name, equity, trades, metrics


def summarize_trades(trades):
    if trades.empty:
        return {}
    buys = int((trades["side"] == "BUY").sum())
    sells = int((trades["side"] == "SELL").sum())
    reasons = trades["reason"].value_counts().to_dict() if "reason" in trades.columns else {}
    return {"buy_trades": buys, "sell_trades": sells, "top_exit_reasons": reasons}


def main():
    all_data = load_csv(DATA)
    benchmark = all_data[all_data["symbol"] == BENCHMARK].copy()
    data = all_data[all_data["symbol"] != BENCHMARK].copy()
    data["industry"] = data["symbol"].map(INDUSTRY)
    profiles = pd.read_csv(PROFILES)
    events = pd.read_csv(EVENTS)

    cases = [
        ("V3.2 分层+行业+新闻", data, benchmark, profiles, events),
        ("V3.2 分层+行业（无新闻）", data, benchmark, profiles, None),
        ("V3.2 无分层+行业+新闻", data, benchmark, None, events),
        ("基准：40%等权买入持有", data, benchmark, None, None),
    ]

    print("=== Longbridge 14-stock / 1,000-trading-day regression ===")
    print(f"range={data['date'].min().date()}..{data['date'].max().date()}")
    print(f"stocks={data['symbol'].nunique()} bars={len(data)} benchmark_bars={len(benchmark)}")

    for name, d, b, p, e in cases[:3]:
        result = run_case(name, d, b, p, e)
        n, eq, trades, m = result
        print(f"\n[{n}]")
        for k, v in m.items():
            print(f"{k}={v}")
        if not eq.empty:
            exposure = ((eq["equity"] - eq["cash"]) / eq["equity"]).mean()
            print(f"avg_equity_exposure={exposure:.6f}")
            print("tier_counts=")
            if p is None:
                print("all_mid_profile")
            else:
                print(p["tier"].value_counts().to_dict())
            print("industry_counts_selected=")
            if not trades.empty:
                merged = trades[["symbol","side"]].copy()
                buys = merged[merged["side"] == "BUY"]
                print(buys["symbol"].map(INDUSTRY).value_counts().to_dict())
            print("trade_stats=", summarize_trades(trades))

    # Static 40% equity-weighted buy-and-hold comparator using the same 14-stock universe.
    close = data.pivot(index="date", columns="symbol", values="close").sort_index()
    common = close.dropna(how="all").copy()
    first = common.iloc[0]
    valid = first.dropna().index
    per_stock = 0.40 / len(valid)
    units = (200_000 * per_stock) / first[valid]
    bh = (common[valid] * units).sum(axis=1) + 200_000 * 0.60
    bh_df = pd.DataFrame({"date": bh.index, "equity": bh.values, "cash": 120_000.0})
    print("\n[基准：40%等权买入持有]")
    for k, v in summarize(bh_df, 200_000).items():
        print(f"{k}={v}")

    # Market ETF buy-and-hold with full capital is also shown as context, not as the
    # strategy target.
    bclose = benchmark.set_index("date")["close"].sort_index()
    units_b = 200_000 / bclose.iloc[0]
    market = pd.DataFrame({"date": bclose.index, "equity": bclose.values * units_b,
                           "cash": 0.0})
    print("\n[参考：510300.SH 100%买入持有]")
    for k, v in summarize(market, 200_000).items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
