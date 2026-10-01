from __future__ import annotations

import math

import pandas as pd

from .data import load_csv, limit_to_trading_days
from .indicators import add_indicators, add_relative_strength
from .strategy import (
    StrategyConfig,
    is_core_entry_eligible,
    is_tactical_entry_eligible,
    rank_candidates,
    select_entries,
    should_exit,
    target_weights,
)
from .portfolio import Portfolio
from .execution import ExecutionConfig, Simulator
from .events import EventConfig, event_score_for_day, validate_events
from .regime import RegimeConfig, market_regime
from .universe import attach_profiles, is_tactical_tier


class Backtester:
    def __init__(
        self,
        initial_cash: float = 200_000,
        strategy_cfg: StrategyConfig | None = None,
        execution_cfg: ExecutionConfig | None = None,
        regime_cfg: RegimeConfig | None = None,
        event_cfg: EventConfig | None = None,
        warning_drawdown_limit: float = 0.07,
        hard_drawdown_limit: float = 0.10,
        principal_guard_weight: float = 0.30,
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
        stock_profiles: pd.DataFrame | None = None,
    ):
        data = limit_to_trading_days(data, self.strategy_cfg.lookback_trading_days)
        if benchmark is not None:
            benchmark = limit_to_trading_days(benchmark, self.strategy_cfg.lookback_trading_days)

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
        if stock_profiles is not None and not stock_profiles.empty:
            profiled = attach_profiles(data, stock_profiles, pd.Timestamp(data["date"].max()))
            profile_industry_map = dict(zip(profiled["symbol"], profiled["industry"]))
            for symbol, industry in profile_industry_map.items():
                if industry:
                    industry_map.setdefault(symbol, industry)

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
        last_exit_date: dict[str, pd.Timestamp] = {}
        severe_event_exit_date: dict[str, pd.Timestamp] = {}
        held_mode: dict[str, str] = {}

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
                        if sim.sell(
                            dates[i + 1], symbol, float(px.iloc[0]), pos.quantity,
                            "hard_drawdown_circuit_breaker",
                        ):
                            last_exit_date[symbol] = pd.Timestamp(dates[i + 1])
                            held_mode.pop(symbol, None)
                cooldown = self.cooldown_days

            risk_budget = min(
                regime_weight,
                self.strategy_cfg.max_equity_weight,
            )
            if drawdown <= -self.warning_drawdown_limit:
                risk_budget = min(risk_budget, 0.35)
            if equity < self.initial_cash:
                risk_budget = min(risk_budget, self.principal_guard_weight)
            if cooldown > 0 or stable_regime == "crisis":
                risk_budget = 0.0

            profiled_today = attach_profiles(today, stock_profiles, pd.Timestamp(date))
            industry_by_symbol = dict(zip(profiled_today["symbol"], profiled_today["industry"]))
            tier_by_symbol = dict(zip(profiled_today["symbol"], profiled_today["tier"]))
            event_scores = {
                symbol: event_score_for_day(
                    events,
                    symbol=symbol,
                    industry=industry_by_symbol.get(symbol) or industry_map.get(symbol, ""),
                    signal_date=pd.Timestamp(date),
                    cfg=self.event_cfg,
                )
                for symbol in today["symbol"].unique()
            }

            ranked = rank_candidates(
                today,
                event_scores,
                self.strategy_cfg,
                stock_profiles=stock_profiles,
                signal_date=pd.Timestamp(date),
            )
            entries_df = select_entries(ranked, self.strategy_cfg)
            if not entries_df.empty:
                # Regime discipline:
                # - defensive/crisis: do not open new positions;
                # - neutral: core only;
                # - risk_on: core + the highly selective tactical sleeve.
                if stable_regime in {"defensive", "crisis"}:
                    entries_df = entries_df.iloc[0:0].copy()
                elif stable_regime != "risk_on":
                    entries_df = entries_df[~entries_df["tier"].map(is_tactical_tier)].copy()

            entries = entries_df["symbol"].tolist() if not entries_df.empty else []
            if not entries_df.empty:
                for _, selected in entries_df.iterrows():
                    tier_by_symbol[selected["symbol"]] = selected.get("tier", "unknown")

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
                tier = held_mode.get(symbol, tier_by_symbol.get(symbol, "unknown"))
                exit_now, reason = should_exit(
                    row,
                    highest_close=pos.highest_close,
                    days_held=pos.days_held,
                    below_ma60_streak=pos.below_ma60_streak,
                    company_score=e.get("company_score", 0.0),
                    severe_negative_event=e.get("severe_negative", False),
                    cfg=self.strategy_cfg,
                    min_holding_days=(
                        self.strategy_cfg.tactical_min_holding_days
                        if is_tactical_tier(tier)
                        else self.strategy_cfg.min_holding_days
                    ),
                    trail_atr_multiple=(
                        self.strategy_cfg.tactical_trail_atr_multiple
                        if is_tactical_tier(tier)
                        else self.strategy_cfg.trail_atr_multiple
                    ),
                )
                if exit_now or cooldown > 0:
                    if sim.sell(
                        dates[i + 1], symbol, float(px_series.iloc[0]), pos.quantity,
                        reason or "risk_off",
                    ):
                        sold_today.add(symbol)
                        if symbol not in portfolio.positions:
                            last_exit_date[symbol] = pd.Timestamp(dates[i + 1])
                            held_mode.pop(symbol, None)
                            if reason == "severe_negative_company_event":
                                severe_event_exit_date[symbol] = pd.Timestamp(dates[i + 1])

            next_prices = dict(zip(next_day["symbol"], next_day["open"]))
            equity_open = portfolio.cash + sum(
                p.quantity * next_prices.get(s, 0.0)
                for s, p in portfolio.positions.items()
            )
            current_total_weight = portfolio.market_value(next_prices) / max(equity_open, 1.0)

            excess_weight = current_total_weight - risk_budget
            # Keep a small tolerance band so ordinary regime fluctuations do not
            # repeatedly trim positions. When a real reduction is needed, cut
            # tactical/weak positions first and protect leaders for last.
            if risk_budget <= 0 or excess_weight > self.strategy_cfg.risk_budget_tolerance:
                removal_value = max(0.0, equity_open * max(excess_weight, 0.0))
                reduction_rows = []
                for symbol, pos in portfolio.positions.items():
                    px = next_prices.get(symbol)
                    row_df = today[today["symbol"] == symbol]
                    if px is None or row_df.empty:
                        continue
                    row = row_df.iloc[0]
                    tier = held_mode.get(symbol, tier_by_symbol.get(symbol, "unknown"))
                    e = event_scores.get(symbol, {})
                    try:
                        strength = float(row.get("technical_score", 0.0))
                    except (TypeError, ValueError):
                        strength = 0.0
                    # Lower bucket is reduced first. Tactical -> other core -> leader.
                    leader_rank = 2 if str(tier).lower() == "leader" else 1 if str(tier).lower() in {"large", "mid"} else 0
                    reduction_rows.append((
                        leader_rank,
                        strength,
                        float(e.get("company_score", 0.0)),
                        symbol,
                        pos,
                        float(px),
                    ))
                reduction_rows.sort(key=lambda x: (x[0], x[1], x[2]))
                for _, _, _, symbol, pos, px in reduction_rows:
                    if removal_value <= 0:
                        break
                    # Prefer fully closing a weak/small position. For the final
                    # position, sell only enough whole lots to remove the excess.
                    sell_qty = min(
                        pos.quantity,
                        max(
                            self.execution_cfg.lot_size,
                            int(math.ceil(removal_value / px / self.execution_cfg.lot_size))
                            * self.execution_cfg.lot_size,
                        ),
                    )
                    sell_qty = int(sell_qty / self.execution_cfg.lot_size) * self.execution_cfg.lot_size
                    if sell_qty <= 0:
                        continue
                    if sim.sell(
                        dates[i + 1], symbol, px, sell_qty,
                        "dynamic_total_risk_budget",
                    ):
                        sold_today.add(symbol)
                        removal_value = max(0.0, removal_value - sell_qty * px)

            equity_open = portfolio.cash + sum(
                p.quantity * next_prices.get(s, 0.0)
                for s, p in portfolio.positions.items()
            )
            targets = target_weights(
                entries,
                risk_budget,
                self.strategy_cfg.max_single_weight,
                stock_tiers=tier_by_symbol,
                tactical_allocation_ratio=self.strategy_cfg.tactical_allocation_ratio,
                max_tactical_weight=self.strategy_cfg.max_tactical_weight,
            )
            for symbol, target_weight in targets.items():
                if symbol in portfolio.positions or symbol in sold_today:
                    continue
                signal_date = pd.Timestamp(dates[i + 1])
                exited = last_exit_date.get(symbol)
                severe_exited = severe_event_exit_date.get(symbol)
                if exited is not None and (signal_date - exited).days < self.strategy_cfg.reentry_cooldown_days:
                    continue
                if severe_exited is not None and (signal_date - severe_exited).days < self.strategy_cfg.severe_event_reentry_days:
                    continue
                row_df = today[today["symbol"] == symbol]
                px = next_prices.get(symbol)
                if row_df.empty or px is None or px <= 0:
                    continue

                row = row_df.iloc[0]
                tier = tier_by_symbol.get(symbol, "unknown")
                e = event_scores.get(symbol, {})
                if is_tactical_tier(tier):
                    entry_ok = (
                        stable_regime == "risk_on"
                        and is_tactical_entry_eligible(
                            row,
                            self.strategy_cfg,
                            company_score=float(e.get("company_score", 0.0)),
                            industry_score=float(e.get("industry_score", 0.0)),
                        )
                    )
                else:
                    entry_ok = stable_regime in {"risk_on", "neutral"} and is_core_entry_eligible(
                        row, self.strategy_cfg
                    )
                if not entry_ok:
                    continue

                desired_value = equity_open * target_weight
                buy_qty = int(desired_value / px)
                if buy_qty <= 0:
                    continue
                sim.buy(dates[i + 1], symbol, px, buy_qty, "v3_tactical_entry" if is_tactical_tier(tier) else "v3_core_entry")
                held_mode[symbol] = tier

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


def run_csv(path, initial_cash=200_000, benchmark_path=None, events_path=None, industry_map_path=None, stock_profiles_path=None):
    data = load_csv(path)
    benchmark = load_csv(benchmark_path) if benchmark_path else None
    events = pd.read_csv(events_path) if events_path else None
    industry_map = None
    if industry_map_path:
        mapping = pd.read_csv(industry_map_path)
        industry_map = dict(zip(mapping["symbol"], mapping["industry"]))
    stock_profiles = pd.read_csv(stock_profiles_path) if stock_profiles_path else None
    return Backtester(initial_cash=initial_cash).run(
        data,
        benchmark=benchmark,
        events=events,
        industry_map=industry_map,
        stock_profiles=stock_profiles,
    )
