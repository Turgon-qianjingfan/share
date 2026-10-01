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
    rows=[]
    for j, symbol in enumerate(["600000.SH", "000001.SZ", "600519.SH"]):
        for i, d in enumerate(dates):
            c=20+j+i*0.02
            rows.append({"date":d,"symbol":symbol,"open":c,"high":c*1.01,"low":c*0.99,"close":c,"volume":1e6})
    equity,trades=Backtester(initial_cash=200_000).run(pd.DataFrame(rows))
    assert not equity.empty
    assert equity["equity"].iloc[-1] > 0

def test_trailing_stop_does_not_exit_before_min_holding():
    cfg=StrategyConfig(min_holding_days=10, trail_atr_multiple=3.0)
    row=pd.Series({"close":95.0,"ma_fast":110.0,"ma_slow":100.0,"atr":1.0})
    exited, _ = should_exit(row, highest_close=110.0, days_held=5, below_ma60_streak=5, cfg=cfg)
    assert exited is False

def test_confirmed_break_can_exit_after_minimum_holding():
    cfg=StrategyConfig(min_holding_days=10, trend_break_confirm_days=3)
    row=pd.Series({"close":97.0,"ma_fast":98.0,"ma_slow":100.0,"atr":1.0})
    exited, reason=should_exit(row, highest_close=110.0, days_held=12, below_ma60_streak=3, cfg=cfg)
    assert exited is True
    assert reason == "confirmed_trend_break"

def test_walk_forward_has_purge():
    dates = pd.bdate_range("2021-10-01", periods=900)
    splits = walk_forward_splits(dates, train_days=504, test_days=126, purge_days=5)
    assert splits
    for train, test in splits:
        assert train[-1] < test[0]


def test_event_signal_is_point_in_time():
    import pandas as pd
    from share.events import event_score_for_day, validate_events

    events=validate_events(pd.DataFrame([{
        "event_time":"2026-01-01T00:00:00Z",
        "symbol":"603799.SH",
        "direction":-1,
        "severity":3,
        "event_type":"earnings",
        "industry":"有色",
        "source_quality":1.0,
        "title":"negative event",
    }]))
    before=event_score_for_day(events, symbol="603799.SH", industry="有色",
                               signal_date=pd.Timestamp("2025-12-31"),)
    after=event_score_for_day(events, symbol="603799.SH", industry="有色",
                              signal_date=pd.Timestamp("2026-01-02"),)
    assert before["company_score"] == 0
    assert after["company_score"] < 0

def test_industry_event_does_not_become_company_event():
    import pandas as pd
    from share.events import event_score_for_day, validate_events

    events=validate_events(pd.DataFrame([{
        "event_time":"2026-01-01T00:00:00Z",
        "symbol":"",
        "direction":1,
        "severity":2,
        "event_type":"industry",
        "industry":"黄金",
        "source_quality":1.0,
        "title":"gold industry event",
    }]))
    out=event_score_for_day(events, symbol="600988.SH", industry="黄金",
                            signal_date=pd.Timestamp("2026-01-02"))
    assert out["company_score"] == 0
    assert out["industry_score"] > 0


def test_empty_candidate_rank_is_safe():
    import pandas as pd
    from share.strategy import rank_candidates, StrategyConfig
    out=rank_candidates(pd.DataFrame(columns=["symbol"]), {}, StrategyConfig())
    assert out.empty
    assert "composite_score" in out.columns
