import pandas as pd
from share.backtest import Backtester
from share.execution import ExecutionConfig
from share.portfolio import Portfolio
from share.execution import Simulator
from share.metrics import walk_forward_splits

def test_t1_and_lot_size():
    p = Portfolio(100_000)
    s = Simulator(p, ExecutionConfig(slippage_rate=0, commission_rate=0, stamp_duty_rate=0))
    assert s.buy("2024-01-01","600000.SH",150,150,"test") is True
    p.mark_t1()
    assert s.sell("2024-01-02","600000.SH",150,50,"test") is False
    assert s.sell("2024-01-02","600000.SH",150,100,"test") is True

def test_backtest_runs_with_200k():
    dates = pd.bdate_range("2021-10-01", periods=650)
    rows=[]
    for j,symbol in enumerate(["600000.SH","000001.SZ","600519.SH"]):
        for i,d in enumerate(dates):
            c=20+j+i*0.02
            rows.append({"date":d,"symbol":symbol,"open":c,"high":c*1.01,"low":c*0.99,"close":c,"volume":1e6})
    equity,trades=Backtester(initial_cash=200_000).run(pd.DataFrame(rows))
    assert not equity.empty
    assert equity["equity"].iloc[-1] > 0
    assert equity["equity"].max() >= 200_000

def test_walk_forward_has_purge():
    dates = pd.bdate_range("2021-10-01", periods=900)
    splits = walk_forward_splits(dates, train_days=504, test_days=126, purge_days=5)
    assert splits
    for train, test in splits:
        assert train[-1] < test[0]
