import pandas as pd

from share.backtest import Backtester
from share.execution import ExecutionConfig, Simulator
from share.portfolio import Portfolio
from share.metrics import walk_forward_splits
from share.strategy import StrategyConfig, should_exit


def test_t1_and_lot_size():
    p = Portfolio(100_000)
    s = Simulator(p, ExecutionConfig(slippage_rate=0, commission_rate=0, stamp_duty_rate=0))
    assert s.buy("2024-01-01", "600000.SH", 150, 150, "test") is True
    p.mark_t1(close_prices={"600000.SH": 150})
    assert s.sell("2024-01-02", "600000.SH", 150, 50, "test") is False
    assert s.sell("2024-01-02", "600000.SH", 150, 100, "test") is True


def test_backtest_runs_with_200k():
    dates = pd.bdate_range("2021-10-01", periods=650)
    rows = []
    for j, symbol in enumerate(["600000.SH", "000001.SZ", "600519.SH"]):
        for i, d in enumerate(dates):
            c = 20 + j + i * 0.02
            rows.append({"date": d, "symbol": symbol, "open": c, "high": c * 1.01,
                         "low": c * 0.99, "close": c, "volume": 1e6})
    equity, trades = Backtester(initial_cash=200_000).run(pd.DataFrame(rows))
    assert not equity.empty
    assert equity["equity"].iloc[-1] > 0


def test_trailing_stop_does_not_exit_before_min_holding():
    cfg = StrategyConfig(min_holding_days=10, trail_atr_multiple=3.0)
    row = pd.Series({"close": 95.0, "ma_fast": 110.0, "ma_slow": 100.0, "atr": 1.0})
    exited, _ = should_exit(row, highest_close=110.0, days_held=5, below_ma60_streak=5, cfg=cfg)
    assert exited is False


def test_confirmed_break_can_exit_after_minimum_holding():
    cfg = StrategyConfig(min_holding_days=10, trend_break_confirm_days=3)
    row = pd.Series({"close": 97.0, "ma_fast": 98.0, "ma_slow": 100.0, "atr": 1.0})
    exited, reason = should_exit(row, highest_close=99.0, days_held=12, below_ma60_streak=3, cfg=cfg)
    assert exited is True
    assert reason == "confirmed_trend_break"


def test_walk_forward_has_purge():
    dates = pd.bdate_range("2021-10-01", periods=900)
    splits = walk_forward_splits(dates, train_days=504, test_days=126, purge_days=5)
    assert splits
    for train, test in splits:
        assert train[-1] < test[0]


def test_event_signal_is_point_in_time():
    from share.events import event_score_for_day, validate_events

    events = validate_events(pd.DataFrame([{
        "event_time": "2026-01-01T00:00:00Z",
        "symbol": "603799.SH",
        "direction": -1,
        "severity": 3,
        "event_type": "earnings",
        "industry": "有色",
        "source_quality": 1.0,
        "title": "negative event",
    }]))
    before = event_score_for_day(events, symbol="603799.SH", industry="有色",
                                 signal_date=pd.Timestamp("2025-12-31"))
    after = event_score_for_day(events, symbol="603799.SH", industry="有色",
                                signal_date=pd.Timestamp("2026-01-02"))
    assert before["company_score"] == 0
    assert after["company_score"] < 0


def test_industry_event_does_not_become_company_event():
    from share.events import event_score_for_day, validate_events

    events = validate_events(pd.DataFrame([{
        "event_time": "2026-01-01T00:00:00Z",
        "symbol": "",
        "direction": 1,
        "severity": 2,
        "event_type": "industry",
        "industry": "黄金",
        "source_quality": 1.0,
        "title": "gold industry event",
    }]))
    out = event_score_for_day(events, symbol="600988.SH", industry="黄金",
                              signal_date=pd.Timestamp("2026-01-02"))
    assert out["company_score"] == 0
    assert out["industry_score"] > 0


def test_empty_candidate_rank_is_safe():
    from share.strategy import rank_candidates, StrategyConfig
    out = rank_candidates(pd.DataFrame(columns=["symbol"]), {}, StrategyConfig())
    assert out.empty
    assert "composite_score" in out.columns


def test_limit_to_latest_1000_trading_days_per_symbol():
    from share.data import limit_to_trading_days
    dates = pd.bdate_range("2020-01-01", periods=1100)
    rows = []
    for symbol in ["A", "B"]:
        for i, d in enumerate(dates):
            rows.append({"date": d, "symbol": symbol, "open": 10 + i * 0.01,
                         "high": 10 + i * 0.01, "low": 10 + i * 0.01,
                         "close": 10 + i * 0.01, "volume": 1000})
    out = limit_to_trading_days(pd.DataFrame(rows), 1000)
    assert out.groupby("symbol").size().to_dict() == {"A": 1000, "B": 1000}
    assert out["date"].min() > dates[0]


def test_selection_prioritises_core_and_diversifies_industries():
    from share.strategy import select_entries, StrategyConfig

    ranked = pd.DataFrame([
        {"symbol": "L1", "industry": "银行", "tier": "leader", "leader_score": 1, "size_score": 1,
         "technical_score": 0.70, "company_score": 0.2, "industry_score": 0.1, "composite_score": 0.70},
        {"symbol": "L2", "industry": "银行", "tier": "leader", "leader_score": 1, "size_score": 1,
         "technical_score": 0.69, "company_score": 0.2, "industry_score": 0.1, "composite_score": 0.69},
        {"symbol": "L3", "industry": "医药", "tier": "large", "leader_score": 0.8, "size_score": 0.9,
         "technical_score": 0.66, "company_score": 0.1, "industry_score": 0.1, "composite_score": 0.66},
        {"symbol": "L4", "industry": "电子", "tier": "large", "leader_score": 0.8, "size_score": 0.8,
         "technical_score": 0.65, "company_score": 0.1, "industry_score": 0.1, "composite_score": 0.65},
        {"symbol": "S1", "industry": "新能源", "tier": "small", "leader_score": 0, "size_score": 0.1,
         "technical_score": 0.90, "company_score": 0.2, "industry_score": 0.2, "composite_score": 0.90},
    ])
    cfg = StrategyConfig(target_positions=4, min_distinct_industries=3,
                         max_industry_positions=2, max_tactical_positions=1)
    out = select_entries(ranked, cfg)
    assert len(out) == 4
    assert out["industry"].nunique() >= 3
    assert out["tier"].eq("small").sum() <= 1
    assert out.iloc[0]["tier"] != "small"


def test_tactical_entry_requires_limit_up_and_flow_confirmation():
    from share.strategy import is_tactical_entry_eligible, StrategyConfig
    base = {"close": 110, "ma_fast": 105, "ma_slow": 100, "atr": 2, "rsi": 75,
            "ret_5": 0.06, "ret_20": 0.10, "ret_60": 0.20, "vol_20": 0.30, "adx": 25,
            "volume_ratio": 1.40, "relative_strength_20": 0.04, "obv_trend": True,
            "money_flow_proxy_20": 0.12, "limit_up_streak": 2}
    assert is_tactical_entry_eligible(pd.Series(base), StrategyConfig(), company_score=0.10, industry_score=0.05) is True
    base["money_flow_proxy_20"] = 0.05
    assert is_tactical_entry_eligible(pd.Series(base), StrategyConfig(), company_score=0.10, industry_score=0.05) is False


def test_market_cap_can_drive_point_in_time_size_and_leadership_scores():
    from share.universe import attach_profiles

    day = pd.DataFrame([
        {"symbol": "A", "industry": "行业甲", "market_cap": 1000},
        {"symbol": "B", "industry": "行业甲", "market_cap": 500},
        {"symbol": "C", "industry": "行业乙", "market_cap": 100},
    ])
    out = attach_profiles(day, None, pd.Timestamp("2026-01-02"))
    a = out[out["symbol"] == "A"].iloc[0]
    b = out[out["symbol"] == "B"].iloc[0]
    c = out[out["symbol"] == "C"].iloc[0]
    assert a["size_score"] > b["size_score"] > c["size_score"]
    assert a["leader_score"] > b["leader_score"]
    assert c["leader_score"] == 0


def test_conservative_core_exposure_cap():
    from share.strategy import target_weights
    cfg = StrategyConfig(max_single_weight=0.10, max_equity_weight=0.40)
    out = target_weights(["A", "B", "C", "D", "E"], 0.40, cfg.max_single_weight)
    assert abs(sum(out.values()) - 0.40) < 1e-9


def test_quality_breakout_entry_can_chase_strong_move():
    from share.strategy import is_entry_eligible, StrategyConfig
    row = pd.Series({
        "close": 110, "ma_fast": 100, "ma_slow": 105, "atr": 2, "rsi": 70,
        "ret_5": 0.04, "ret_20": 0.08, "ret_60": 0.12, "vol_20": 0.35, "adx": 18,
        "volume_ratio": 1.40, "relative_strength_20": 0.06, "breakout_20": True,
    })
    assert is_entry_eligible(row, StrategyConfig()) is True
