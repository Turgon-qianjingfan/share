from __future__ import annotations

import pandas as pd

from .data import load_csv
from .indicators import add_indicators, add_relative_strength
from .strategy import StrategyConfig, is_entry_eligible, rank_candidates, should_exit, target_weights
from .portfolio import Portfolio
from .execution import ExecutionConfig, Simulator
from .events import EventConfig, event_score_for_day, validate_events
from .regime import RegimeConfig, market_regime


class Backtester:
    def __init__(
        self,
        initial_cash: float = 200_000,
        strategy_cfg: StrategyConfig | None = None,
        execution_cfg: ExecutionConfig | None = None,
        regime_cfg: RegimeConfig | None = None,
        event_cfg: EventConfig | None = None,
        warning_drawdown_limit: float = 0.05,
        hard_drawdown_limit: float = 0.08,
        principal_guard_weight: float = 0.20,
        cooldown_days: int = 10,
    ):
        self.strategy_cfg = strategy_cfg or StrategyConfig()
        self.execution_cfg = execution_cfg or ExecutionConfig()
        self.regime_cfg = regime_cfg or RegimeConfig()
        self.event_cfg = event_cfg or EventConfig()
        self.initial_cash = initial_cash
        self.warning_drawdown_limit = warning_drawdown_limit
        self.hard_drawdown_limit = hard_drawdown_limit
        self.principal_guard_weight = principal_guard_weight
        self.cooldown_days = cooldown_days

    @staticmethod
    def _market_regime_state(regime_candidate, regime_streak, stable_regime, raw_regime):
        if raw_regime == "crisis":
            return raw_regime, 3, raw_regime
        if raw_regime == regime_candidate:
            regime_streak += 1
        else:
            regime_candidate, regime_streak = raw_regime, 1
        if regime_streak >= 3:
            stable_regime = regime_candidate
        return regime_candidate, regime_streak, stable_regime

    def run(
        self,
        data: pd.DataFrame,
        benchmark: pd.DataFrame | None = None,
        events: pd.DataFrame | None = None,
        industry_map: dict[str, str] | None = None,
    ):
        data = add_indicators(
            data,
            fast=self.strategy_cfg.lookback_fast,
            slow=self.strategy_cfg.lookback_slow,
            atr_window=self.strategy_cfg.atr_window,
        )

        if benchmark is None:
            benchmark = data.groupby("date", as_index=False).agg(
                open=("open", "mean"),
                high=("high", "mean"),
                low=("low", "mean"),
                close=("close", "mean"),
                volume=("volume", "sum"),
            )

        benchmark = add_indicators(
            benchmark.assign(symbol="MARKET_PROXY"),
            fast=self.strategy_cfg.lookback_fast,
            slow=self.strategy_cfg.lookback_slow,
            atr_window=self.strategy_cfg.atr_window,
        )
        data = add_relative_strength(data, benchmark)
        events = validate_events(events) if events is not None and not events.empty else pd.DataFrame()
        industry_map = industry_map or {}

        portfolio = Portfolio(self.initial_cash)
        sim = Simulator(portfolio, self.execution_cfg)
        dates = sorted(data["date"].unique())
        benchmark_by_date = {
            row["date"]: row for _, row in benchmark.iterrows()
            if row["date"] in dates
        }

        breadth_history: list[float] = []
        rows = []
        peak_equity = self.initial_cash
        cooldown = 0
        regime_candidate = None
        regime_streak = 0
        stable_regime = "neutral"

        for i, date in enumerate(dates[:-1]):
            sold_today: set[str] = set()
            today = data[data["date"] == date]
            next_day = data[data["date"] == dates[i + 1]]
            prices_close = dict(zip(today["symbol"], today["close"]))
            portfolio.mark_t1(close_prices=prices_close)

            equity = portfolio.equity(prices_close)
            peak_equity = max(peak_equity, equity)
            drawdown = equity / peak_equity - 1
            cooldown = max(0, cooldown - 1)

            valid_breadth = today.dropna(subset=["ma_slow"])
            breadth = (
                float((valid_breadth["close"] > valid_breadth["ma_slow"]).mean())
                if not valid_breadth.empty else 0.5
            )
            breadth_history.append(breadth)
            breadth_ma20 = sum(breadth_history[-20:]) / min(len(breadth_history), 20)

            bm = benchmark_by_date.get(date)
            if bm is None or pd.isna(bm.get("ma_slow")):
                raw_regime = "neutral"
            else:
                raw_regime, _ = market_regime(
                    pd.Series(bm),
                    breadth=breadth,
                    breadth_ma20=breadth_ma20,
                    cfg=self.regime_cfg,
                )
            regime_candidate, regime_streak, stable_regime = self._market_regime_state(
                regime_candidate, regime_streak, stable_regime, raw_regime
            )
            regime_weight = {
                "risk_on": self.regime_cfg.risk_on_weight,
                "neutral": self.regime_cfg.neutral_weight,
                "defensive": self.regime_cfg.defensive_weight,
                "crisis": self.regime_cfg.crisis_weight,
            }[stable_regime]

            if drawdown <= -self.hard_drawdown_limit:
                for symbol, pos in list(portfolio.positions.items()):
                    px = next_day.loc[next_day["symbol"] == symbol, "open"]
                    if not px.empty:
                        sim.sell(
                            dates[i + 1], symbol, float(px.iloc[0]), pos.quantity,
                            "hard_drawdown_circuit_breaker",
                        )
                cooldown = self.cooldown_days

            risk_budget = min(regime_weight, 0.10) if drawdown <= -self.warning_drawdown_limit else regime_weight
            if equity < self.initial_cash:
                risk_budget = min(risk_budget, self.principal_guard_weight)
            if cooldown > 0 or stable_regime == "crisis":
                risk_budget = 0.0

            event_scores = {
                symbol: event_score_for_day(
                    events,
                    symbol=symbol,
                    industry=industry_map.get(symbol, ""),
                    signal_date=pd.Timestamp(date),
                    cfg=self.event_cfg,
                )
                for symbol in today["symbol"].unique()
            }

            ranked = rank_candidates(today, event_scores, self.strategy_cfg)
            entries = ranked["symbol"].tolist()[: self.strategy_cfg.target_positions] if not ranked.empty else []

            for symbol, pos in list(portfolio.positions.items()):
                row_df = today[today["symbol"] == symbol]
                px_series = next_day.loc[next_day["symbol"] == symbol, "open"]
                if row_df.empty or px_series.empty:
                    continue

                row = row_df.iloc[0]
                if row["close"] < row["ma_slow"] * self.strategy_cfg.hold_ma_buffer and row["ma_fast"] < row["ma_slow"]:
                    pos.below_ma60_streak += 1
                else:
                    pos.below_ma60_streak = 0
                pos.highest_close = max(pos.highest_close, float(row["close"]))

                e = event_scores.get(symbol, {})
                exit_now, reason = should_exit(
                    row,
                    highest_close=pos.highest_close,
                    days_held=pos.days_held,
                    below_ma60_streak=pos.below_ma60_streak,
                    company_score=e.get("company_score", 0.0),
                    severe_negative_event=e.get("severe_negative", False),
                    cfg=self.strategy_cfg,
                )
                if exit_now or cooldown > 0:
                    if sim.sell(
                        dates[i + 1], symbol, float(px_series.iloc[0]), pos.quantity,
                        reason or "risk_off",
                    ):
                        sold_today.add(symbol)

            next_prices = dict(zip(next_day["symbol"], next_day["open"]))
            equity_open = portfolio.cash + sum(
                p.quantity * next_prices.get(s, 0.0)
                for s, p in portfolio.positions.items()
            )
            current_total_weight = portfolio.market_value(next_prices) / max(equity_open, 1.0)

            if current_total_weight > risk_budget:
                reduction_ratio = 0.0 if risk_budget <= 0 else risk_budget / current_total_weight
                for symbol, pos in list(portfolio.positions.items()):
                    px = next_prices.get(symbol)
                    if px is None:
                        continue
                    keep_qty = int(
                        (pos.quantity * reduction_ratio) / self.execution_cfg.lot_size
                    ) * self.execution_cfg.lot_size
                    sell_qty = max(0, pos.quantity - keep_qty)
                    if sell_qty > 0:
                        if sim.sell(
                            dates[i + 1], symbol, px, sell_qty,
                            "dynamic_total_risk_budget",
                        ):
                            sold_today.add(symbol)

            equity_open = portfolio.cash + sum(
                p.quantity * next_prices.get(s, 0.0)
                for s, p in portfolio.positions.items()
            )
            targets = target_weights(entries, risk_budget, self.strategy_cfg.max_single_weight)
            for symbol, target_weight in targets.items():
                # No daily top-up. Only open a genuinely new position.
                if symbol in portfolio.positions or symbol in sold_today:
                    continue
                row_df = today[today["symbol"] == symbol]
                px = next_prices.get(symbol)
                if row_df.empty or px is None or px <= 0:
                    continue
                if not is_entry_eligible(row_df.iloc[0], self.strategy_cfg):
                    continue

                desired_value = equity_open * target_weight
                buy_qty = int(desired_value / px)
                if buy_qty <= 0:
                    continue
                sim.buy(dates[i + 1], symbol, px, buy_qty, "v3_new_position_entry")

            prices = dict(zip(next_day["symbol"], next_day["close"]))
            end_equity = portfolio.equity(prices)
            peak_equity = max(peak_equity, end_equity)
            rows.append({
                "date": dates[i + 1],
                "equity": end_equity,
                "cash": portfolio.cash,
                "positions": len(portfolio.positions),
                "drawdown": end_equity / peak_equity - 1,
                "risk_budget": risk_budget,
                "market_regime": stable_regime,
                "breadth": breadth,
                "event_count": int(sum(v.get("event_count", 0) for v in event_scores.values())),
            })

        return pd.DataFrame(rows), pd.DataFrame(sim.trades)


def run_csv(path, initial_cash=200_000, benchmark_path=None, events_path=None, industry_map_path=None):
    data = load_csv(path)
    benchmark = load_csv(benchmark_path) if benchmark_path else None
    events = pd.read_csv(events_path) if events_path else None
    industry_map = None
    if industry_map_path:
        mapping = pd.read_csv(industry_map_path)
        industry_map = dict(zip(mapping["symbol"], mapping["industry"]))
    return Backtester(initial_cash=initial_cash).run(
        data,
        benchmark=benchmark,
        events=events,
        industry_map=industry_map,
    )
