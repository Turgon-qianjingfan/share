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
    """Return (block_new_entries, force_liquidation)."""
    block_new_entries = cooldown_active or drawdown <= -warning_drawdown_limit
    force_liquidation = cooldown_active or drawdown <= -hard_drawdown_limit
    return block_new_entries, force_liquidation


def _flag(row, name, default=False):
    if row is None or name not in row.index or pd.isna(row[name]):
        return default
    return bool(row[name])


def _optional_price(row, name):
    if row is None or name not in row.index or pd.isna(row[name]):
        return None
    return float(row[name])


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
        last_close: dict[str, float] = {}

        for i, date in enumerate(dates[:-1]):
            today = data[data["date"] == date]
            next_day = data[data["date"] == dates[i + 1]]
            today_rows = {r["symbol"]: r for _, r in today.iterrows()}
            next_rows = {r["symbol"]: r for _, r in next_day.iterrows()}
            prices_close = dict(last_close)
            prices_close.update(dict(zip(today["symbol"], today["close"])))

            portfolio.mark_t1(close_prices=prices_close)
            equity = portfolio.equity(prices_close)
            peak_equity = max(peak_equity, equity)
            drawdown = equity / peak_equity - 1

            cooldown_active = cooldown > 0
            if cooldown_active:
                cooldown -= 1

            next_open_prices = dict(zip(next_day["symbol"], next_day["open"]))
            # Missing bars do not imply a zero-value asset: carry the last known
            # close for valuation, but never use that carried value to execute.
            next_prices = dict(prices_close)
            next_prices.update(next_open_prices)

            hard_drawdown_trigger = drawdown <= -self.hard_drawdown_limit
            if hard_drawdown_trigger:
                for symbol, pos in list(portfolio.positions.items()):
                    bar = next_rows.get(symbol)
                    px = next_open_prices.get(symbol)
                    if px is not None:
                        sim.sell(
                            dates[i + 1], symbol, px, pos.quantity,
                            "hard_drawdown_circuit_breaker",
                            suspended=_flag(bar, "is_suspended"),
                            limit_down=_optional_price(bar, "limit_down"),
                        )
                cooldown = self.cooldown_days
                cooldown_active = True

            risk_off, force_liquidation = risk_controls(
                drawdown, cooldown_active,
                warning_drawdown_limit=self.warning_drawdown_limit,
                hard_drawdown_limit=self.hard_drawdown_limit,
            )

            if equity < self.initial_cash and not risk_off:
                allowed_equity_weight = min(self.max_equity_weight, self.principal_guard_weight)
            else:
                allowed_equity_weight = self.max_equity_weight

            entries = [] if risk_off else entry_candidates(today, self.strategy_cfg)
            target_map = target_weights(
                entries, allowed_equity_weight, self.strategy_cfg.max_single_weight,
            )

            for symbol, pos in list(portfolio.positions.items()):
                current = today_rows.get(symbol)
                if current is None:
                    # Missing bar/suspension: keep the position and its last mark.
                    continue
                if (
                    current["close"] < current["ma_slow"] * self.strategy_cfg.hold_ma_buffer
                    and current["ma_fast"] < current["ma_slow"]
                ):
                    pos.below_ma60_streak += 1
                else:
                    pos.below_ma60_streak = 0
                pos.highest_close = max(pos.highest_close, float(current["close"]))

                bar = next_rows.get(symbol)
                px = next_open_prices.get(symbol)
                if px is None:
                    continue
                if force_liquidation:
                    sim.sell(
                        dates[i + 1], symbol, px, pos.quantity,
                        "hard_drawdown_or_cooldown",
                        suspended=_flag(bar, "is_suspended"),
                        limit_down=_optional_price(bar, "limit_down"),
                    )
                    continue

                exit_now, reason = should_exit(
                    current, highest_close=pos.highest_close,
                    days_held=pos.days_held,
                    below_ma60_streak=pos.below_ma60_streak,
                    cfg=self.strategy_cfg,
                )
                if exit_now:
                    sim.sell(
                        dates[i + 1], symbol, px, pos.quantity, reason,
                        suspended=_flag(bar, "is_suspended"),
                        limit_down=_optional_price(bar, "limit_down"),
                    )

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
                        * next_prices.get(symbol, 0.0) / max(equity_open, 1.0)
                    )
                    if target_weight - current_weight < self.strategy_cfg.rebalance_threshold:
                        continue
                px = next_open_prices.get(symbol)
                if px is None or px <= 0:
                    continue
                bar = next_rows.get(symbol)
                if _flag(bar, "is_suspended") or (
                    _optional_price(bar, "limit_up") is not None
                    and px >= _optional_price(bar, "limit_up") - max(px * 1e-8, 1e-8)
                ):
                    continue

                remaining_capacity = max(0.0, allowed_equity_weight - current_total_weight)
                desired_weight = min(target_weight, remaining_capacity)
                if desired_weight <= 0:
                    continue
                desired_value = equity_open * desired_weight
                position = portfolio.positions.get(symbol)
                current_value = 0.0 if position is None else position.quantity * px
                amount = max(0.0, desired_value - current_value)
                bought = int(amount / px)
                if bought > 0:
                    before_qty = 0 if position is None else position.quantity
                    if sim.buy(
                        dates[i + 1], symbol, px, bought, "confirmed_entry",
                        suspended=_flag(bar, "is_suspended"),
                        limit_up=_optional_price(bar, "limit_up"),
                    ):
                        if before_qty == 0:
                            portfolio.positions[symbol].entry_date = str(dates[i + 1])
                            portfolio.positions[symbol].highest_close = px
                        current_total_weight = (
                            portfolio.market_value(next_prices) / max(equity_open, 1.0)
                        )

            end_prices = dict(prices_close)
            end_prices.update(dict(zip(next_day["symbol"], next_day["close"])))
            end_equity = portfolio.equity(end_prices)
            peak_equity = max(peak_equity, end_equity)
            rows.append({
                "date": dates[i + 1],
                "equity": end_equity,
                "cash": portfolio.cash,
                "positions": len(portfolio.positions),
                "drawdown": end_equity / peak_equity - 1,
                "risk_off": risk_off,
            })
            last_close = end_prices

        return pd.DataFrame(rows), pd.DataFrame(sim.trades)


def run_csv(path, initial_cash=200_000):
    return Backtester(initial_cash=initial_cash).run(load_csv(path))
