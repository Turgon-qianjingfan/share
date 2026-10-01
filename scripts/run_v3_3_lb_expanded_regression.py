from __future__ import annotations

from pathlib import Path

import pandas as pd

from share.backtest import Backtester
from share.data import load_csv
from share.metrics import summarize
from share.strategy import StrategyConfig

BASE = Path("data/demo/v3_2_14stocks_lb_1000d.csv")
EXTRAS = sorted(Path("data/demo").glob("v3_2_lb_extra_*.csv"))
BENCHMARK = "510300.SH"

INDUSTRY = {
    # Defensive / stable
    "600036.SH": "银行", "601398.SH": "银行", "600030.SH": "券商",
    "601318.SH": "保险", "601688.SH": "券商",
    "600519.SH": "食品饮料", "000858.SZ": "食品饮料", "600887.SH": "食品饮料",
    "000333.SZ": "家电", "603288.SH": "家电", "600690.SH": "家电", "000651.SZ": "家电",
    "600276.SH": "医药", "300760.SZ": "医药", "000963.SZ": "医药",
    "600900.SH": "公共事业", "601985.SH": "公共事业", "600941.SH": "通信", "600050.SH": "通信",
    # Technology / growth
    "002415.SZ": "电子", "002475.SZ": "电子", "002371.SZ": "半导体设备",
    "688012.SH": "半导体设备", "688041.SH": "半导体",
    "300750.SZ": "新能源电池", "002812.SZ": "新能源电池", "601012.SH": "新能源光伏",
    # Energy / resources
    "600938.SH": "能源", "601857.SH": "能源", "600028.SH": "能源", "601088.SH": "煤炭",
    "601899.SH": "有色", "600111.SH": "稀土有色", "600547.SH": "黄金", "603799.SH": "有色",
    # Industrial / cyclical
    "601100.SH": "工业机械", "600031.SH": "工程机械", "000425.SZ": "工程机械",
    "603011.SH": "工业机械", "600010.SH": "钢铁", "600019.SH": "钢铁", "600309.SH": "化工",
    # Automotive / defense
    "600104.SH": "汽车", "601633.SH": "汽车", "000625.SZ": "汽车",
    "600760.SH": "军工", "600893.SH": "军工", "000768.SZ": "军工",
}

def load_expanded():
    frames = [load_csv(BASE)]
    for path in EXTRAS:
        frames.append(load_csv(path))
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.drop_duplicates(["date", "symbol"], keep="first")
    raw["industry"] = raw["symbol"].map(INDUSTRY).fillna("其他")
    benchmark = raw[raw["symbol"] == BENCHMARK].copy()
    data = raw[raw["symbol"] != BENCHMARK].copy()
    return data, benchmark

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
