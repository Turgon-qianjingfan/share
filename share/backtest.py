from __future__ import annotations

import pandas as pd

from .data import load_csv
from .indicators import add_indicators
from .strategy import StrategyConfig, entry_candidates, should_exit, target_weights
from .portfolio import Portfolio
from .execution import ExecutionConfig, Simulator


def risk_controls(
    drawdown: float,
    cooldown_active: bool,
    warning_drawdown_limit: float = 0.05,
    hard_drawdown_limit: float = 0.08,
) -> tuple[bool, bool]:
    """Return (block_new_entries, force_liquidation).

    A warning drawdown blocks new risk but lets existing positions follow their
    normal exit rules. Only the hard threshold (or an active cooldown) forces
    liquidation, keeping the two risk levels meaningfully distinct.
    """
    block_new_entries = cooldown_active or drawdown <= -warning_drawdown_limit
    force_liquidation = cooldown_active or drawdown <= -hard_drawdown_limit
    return block_new_entries, force_liquidation


class Backtester:
    def __init__(
        self,
        initial_cash: float = 200_000,
        strategy_cfg: StrategyConfig | None = None,
        execution_cfg: ExecutionConfig | None = None,
        max_equity_weight: float = 0.40,
        warning_drawdown_limit: float = 0.05,
        hard_drawdown_limit: float = 0.08,
        cooldown_days: int = 10,
        principal_guard_weight: float = 0.20,
    ):
        self.strategy_cfg = strategy_cfg or StrategyConfig()
        self.execution_cfg = execution_cfg or ExecutionConfig()
        self.initial_cash = initial_cash
        self.max_equity_weight = max_equity_weight
        self.warning_drawdown_limit = warning_drawdown_limit
        self.hard_drawdown_limit = hard_drawdown_limit
        self.cooldown_days = cooldown_days
        self.principal_guard_weight = principal_guard_weight

    def run(self, data: pd.DataFrame):
        data = add_indicators(
            data,
            fast=self.strategy_cfg.lookback_fast,
            slow=self.strategy_cfg.lookback_slow,
            atr_window=self.strategy_cfg.atr_window,
        )
        portfolio = Portfolio(self.initial_cash)
        sim = Simulator(portfolio, self.execution_cfg)
        rows = []
        dates = sorted(data["date"].unique())
        peak_equity = self.initial_cash
        cooldown = 0

        for i, date in enumerate(dates[:-1]):
            today = data[data["date"] == date]
            next_day = data[data["date"] == dates[i + 1]]
            prices_close = dict(zip(today["symbol"], today["close"]))

            portfolio.mark_t1(close_prices=prices_close)

            equity = portfolio.equity(prices_close)
            peak_equity = max(peak_equity, equity)
            drawdown = equity / peak_equity - 1

            cooldown_active = cooldown > 0
            if cooldown_active:
                cooldown -= 1

            next_prices = dict(zip(next_day["symbol"], next_day["open"]))

            # Global capital-preservation circuit breaker.
            hard_drawdown_trigger = drawdown <= -self.hard_drawdown_limit
            if hard_drawdown_trigger:
                for symbol, pos in list(portfolio.positions.items()):
                    px = next_prices.get(symbol)
                    if px is not None:
                        sim.sell(
                            dates[i + 1], symbol, px, pos.quantity,
                            "hard_drawdown_circuit_breaker",
                        )
                cooldown = self.cooldown_days
                cooldown_active = True

            risk_off, force_liquidation = risk_controls(
                drawdown,
                cooldown_active,
                warning_drawdown_limit=self.warning_drawdown_limit,
                hard_drawdown_limit=self.hard_drawdown_limit,
            )

            # When the account is below principal, cut normal equity exposure
            # in half. This is a capital-recovery guard, not a guarantee.
            if equity < self.initial_cash and not risk_off:
                allowed_equity_weight = min(self.max_equity_weight, self.principal_guard_weight)
            else:
                allowed_equity_weight = self.max_equity_weight

            entries = [] if risk_off else entry_candidates(today, self.strategy_cfg)
            target_map = target_weights(
                entries,
                allowed_equity_weight,
                self.strategy_cfg.max_single_weight,
            )

            # Update holding-state indicators and process exits.
            for symbol, pos in list(portfolio.positions.items()):
                row = today[today["symbol"] == symbol]
                if row.empty:
                    continue
                current = row.iloc[0]
                if (
                    current["close"] < current["ma_slow"] * self.strategy_cfg.hold_ma_buffer
                    and current["ma_fast"] < current["ma_slow"]
                ):
                    pos.below_ma60_streak += 1
                else:
                    pos.below_ma60_streak = 0

                pos.highest_close = max(pos.highest_close, float(current["close"]))

                px = next_prices.get(symbol)
                if px is None:
                    continue

                # A warning drawdown blocks new entries but lets holdings follow
                # their normal stop/trend rules. Hard drawdown/cooldown forces exit.
                if force_liquidation:
                    sim.sell(
                        dates[i + 1], symbol, px, pos.quantity,
                        "hard_drawdown_or_cooldown",
                    )
                    continue

                exit_now, reason = should_exit(
                    current,
                    highest_close=pos.highest_close,
                    days_held=pos.days_held,
                    below_ma60_streak=pos.below_ma60_streak,
                    cfg=self.strategy_cfg,
                )
                if exit_now:
                    sim.sell(dates[i + 1], symbol, px, pos.quantity, reason)

            # Only add NEW / materially underweight positions. Do not rebalance
            # every day; this is the main turnover control.
            equity_open = portfolio.cash + sum(
                p.quantity * next_prices.get(s, 0.0)
                for s, p in portfolio.positions.items()
            )
            current_total_weight = (
                portfolio.market_value(next_prices) / max(equity_open, 1.0)
            )

            for symbol, target_weight in target_map.items():
                if symbol in portfolio.positions:
                    current_weight = (
                        portfolio.positions[symbol].quantity
                        * next_prices.get(symbol, 0.0)
                        / max(equity_open, 1.0)
                    )
                    if target_weight - current_weight < self.strategy_cfg.rebalance_threshold:
                        continue
                px = next_prices.get(symbol)
                if px is None or px <= 0:
                    continue

                remaining_capacity = max(0.0, allowed_equity_weight - current_total_weight)
                desired_weight = min(target_weight, remaining_capacity)
                if desired_weight <= 0:
                    continue

                desired_value = equity_open * desired_weight
                current_value = portfolio.positions.get(symbol, None)
                current_value = (
                    0.0 if current_value is None
                    else current_value.quantity * px
                )
                amount = max(0.0, desired_value - current_value)
                bought = int(amount / px)
                if bought > 0:
                    before_qty = portfolio.positions.get(symbol).quantity if symbol in portfolio.positions else 0
                    if sim.buy(dates[i + 1], symbol, px, bought, "confirmed_entry"):
                        after_qty = portfolio.positions[symbol].quantity
                        if before_qty == 0:
                            portfolio.positions[symbol].entry_date = str(dates[i + 1])
                            portfolio.positions[symbol].highest_close = px
                        current_total_weight = (
                            portfolio.market_value(next_prices) / max(equity_open, 1.0)
                        )

            prices = dict(zip(next_day["symbol"], next_day["close"]))
            end_equity = portfolio.equity(prices)
            peak_equity = max(peak_equity, end_equity)
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
