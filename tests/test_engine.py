import pandas as pd
from share.backtest import Backtester
from share.execution import ExecutionConfig
from share.portfolio import Portfolio
from share.execution import Simulator

def test_t1_and_lot_size():
    p = Portfolio(100_000)
    s = Simulator(p, ExecutionConfig(slippage_rate=0, commission_rate=0, stamp_duty_rate=0))
    assert s.buy("2024-01-01","600000.SH",10,150,"test") is True
    p.mark_t1()
    assert s.sell("2024-01-02","600000.SH",10,50,"test") is False
    assert s.sell("2024-01-02","600000.SH",10,100,"test") is True

def test_backtest_runs():
    dates = pd.bdate_range("2024-01-01", periods=100)
    rows=[]
    for j,symbol in enumerate(["600000.SH","000001.SZ","600519.SH"]):
        for i,d in enumerate(dates):
            c=20+j+i*0.02
            rows.append({"date":d,"symbol":symbol,"open":c,"high":c*1.01,"low":c*0.99,"close":c,"volume":1e6})
    df=pd.DataFrame(rows)
    equity,trades=Backtester().run(df)
    assert not equity.empty
    assert equity["equity"].iloc[-1] > 0
