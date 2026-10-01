from __future__ import annotations

import pandas as pd

from .data import load_csv
from .indicators import add_indicators
from .strategy import StrategyConfig, select_candidates, target_weights
from .portfolio import Portfolio
from .execution import ExecutionConfig, Simulator


class Backtester:
    def __init__(
        self,
        initial_cash: float = 200_000,
        strategy_cfg: StrategyConfig | None = None,
        execution_cfg: ExecutionConfig | None = None,
        max_equity_weight: float = 0.40,
        hard_drawdown_limit: float = 0.08,
        warning_drawdown_limit: float = 0.05,
        recovery_drawdown_limit: float = 0.03,
        cooldown_days: int = 10,
    ):
        self.strategy_cfg = strategy_cfg or StrategyConfig(
            max_single_weight=0.08,
            max_equity_weight=max_equity_weight,
            target_positions=5,
        )
        self.execution_cfg = execution_cfg or ExecutionConfig()
        self.initial_cash = initial_cash
        self.max_equity_weight = max_equity_weight
        self.hard_drawdown_limit = hard_drawdown_limit
        self.warning_drawdown_limit = warning_drawdown_limit
        self.recovery_drawdown_limit = recovery_drawdown_limit
        self.cooldown_days = cooldown_days

    def run(self, data: pd.DataFrame):
        data = add_indicators(data, fast=self.strategy_cfg.lookback_fast, slow=self.strategy_cfg.lookback_slow, atr_window=self.strategy_cfg.atr_window)
        portfolio = Portfolio(self.initial_cash)
        sim = Simulator(portfolio, self.execution_cfg)
        rows = []
        dates = sorted(data["date"].unique())
        peak_equity = self.initial_cash
        cooldown = 0

        for i, date in enumerate(dates[:-1]):
            today = data[data["date"] == date]
            next_day = data[data["date"] == dates[i + 1]]
            prices_close = dict(zip(today.symbol, today.close))
            portfolio.mark_t1()

            equity = portfolio.equity(prices_close)
            peak_equity = max(peak_equity, equity)
            drawdown = equity / peak_equity - 1

            if cooldown > 0:
                cooldown -= 1

            # Capital-preservation circuit breaker:
            # after an 8% peak-to-trough drawdown, liquidate and remain in cash
            # for a cooldown period. This does NOT mathematically guarantee
            # preservation of principal; it limits strategy risk.
            if drawdown <= -self.hard_drawdown_limit:
                for symbol, pos in list(portfolio.positions.items()):
                    px = next_day.loc[next_day.symbol == symbol, "open"]
                    if not px.empty:
                        sim.sell(dates[i + 1], symbol, float(px.iloc[0]), pos.quantity, "hard_drawdown_circuit_breaker")
                cooldown = self.cooldown_days

            risk_off = cooldown > 0 or drawdown <= -self.warning_drawdown_limit
            equity_weight = 0.0 if risk_off else self.max_equity_weight
            if drawdown > -self.recovery_drawdown_limit and cooldown == 0:
                equity_weight = self.max_equity_weight

            candidates = [] if risk_off else select_candidates(today, self.strategy_cfg)
            targets = target_weights(candidates, equity_weight, self.strategy_cfg.max_single_weight)

            next_prices = dict(zip(next_day.symbol, next_day.open))
            equity_before = portfolio.equity(prices_close)

            for symbol, pos in list(portfolio.positions.items()):
                target = targets.get(symbol, 0.0)
                px = next_prices.get(symbol, prices_close.get(symbol, 0))
                current = pos.quantity * px / max(equity_before, 1)
                if target < current:
                    qty = pos.quantity if target == 0 else int((current - target) * equity_before / max(px, 1))
                    sim.sell(dates[i + 1], symbol, px, qty, "capital_preservation_rebalance")

            equity_open = portfolio.cash + sum(
                p.quantity * next_prices.get(s, 0)
                for s, p in portfolio.positions.items()
            )
            for symbol, weight in targets.items():
                px = next_prices.get(symbol)
                if px is None or px <= 0:
                    continue
                current_value = portfolio.positions.get(symbol).quantity * px if symbol in portfolio.positions else 0
                desired = equity_open * weight
                amount = max(0, desired - current_value)
                sim.buy(dates[i + 1], symbol, px, int(amount / px), "trend_rebalance")

            prices = dict(zip(next_day.symbol, next_day.close))
            end_equity = portfolio.equity(prices)
            rows.append({
                "date": dates[i + 1],
                "equity": end_equity,
                "cash": portfolio.cash,
                "positions": len(portfolio.positions),
                "drawdown": end_equity / peak_equity - 1,
                "risk_off": risk_off,
            })

        return pd.DataFrame(rows), pd.DataFrame(sim.trades)


def run_csv(path, initial_cash=200_000):
    return Backtester(initial_cash=initial_cash).run(load_csv(path))
