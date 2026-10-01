from __future__ import annotations
import pandas as pd
from .data import load_csv
from .indicators import add_indicators
from .strategy import StrategyConfig, select_candidates, target_weights
from .portfolio import Portfolio
from .execution import ExecutionConfig, Simulator

class Backtester:
    def __init__(self, initial_cash=1_000_000, strategy_cfg=None, execution_cfg=None):
        self.strategy_cfg = strategy_cfg or StrategyConfig()
        self.execution_cfg = execution_cfg or ExecutionConfig()
        self.initial_cash = initial_cash

    def run(self, data: pd.DataFrame):
        data = add_indicators(data)
        portfolio = Portfolio(self.initial_cash)
        sim = Simulator(portfolio, self.execution_cfg)
        rows = []
        dates = sorted(data["date"].unique())
        for i, date in enumerate(dates[:-1]):
            today = data[data["date"] == date]
            next_day = data[data["date"] == dates[i+1]]
            prices_close = dict(zip(today.symbol, today.close))
            portfolio.mark_t1()
            candidates = select_candidates(today, self.strategy_cfg)
            equity_weight = self.strategy_cfg.max_equity_weight
            targets = target_weights(candidates, equity_weight, self.strategy_cfg.max_single_weight)
            next_prices = dict(zip(next_day.symbol, next_day.open))
            equity_before = portfolio.equity(prices_close)
            # 先减仓：不再符合信号的股票，或超过目标权重。
            for symbol, pos in list(portfolio.positions.items()):
                target = targets.get(symbol, 0.0)
                current = pos.quantity * next_prices.get(symbol, prices_close.get(symbol,0)) / max(equity_before,1)
                if target < current:
                    sim.sell(date=dates[i+1], symbol=symbol,
                             price=next_prices.get(symbol, prices_close.get(symbol,0)),
                             quantity=pos.quantity if target == 0 else max(0, int((current-target)*equity_before / max(next_prices.get(symbol,1),1))),
                             reason="rebalance")
            # 再建仓。
            equity_open = portfolio.cash + sum(p.quantity * next_prices.get(s,0) for s,p in portfolio.positions.items())
            for symbol, weight in targets.items():
                px = next_prices.get(symbol)
                if px is None: continue
                current_value = portfolio.positions.get(symbol).quantity * px if symbol in portfolio.positions else 0
                desired = equity_open * weight
                amount = max(0, desired - current_value)
                sim.buy(date=dates[i+1], symbol=symbol, price=px,
                        quantity=int(amount/px), reason="trend_rebalance")
            prices = dict(zip(next_day.symbol, next_day.close))
            rows.append({"date":dates[i+1],"equity":portfolio.equity(prices),"cash":portfolio.cash,
                         "positions":len(portfolio.positions)})
        return pd.DataFrame(rows), pd.DataFrame(sim.trades)

def run_csv(path, initial_cash=1_000_000):
    return Backtester(initial_cash).run(load_csv(path))
