from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import pandas as pd


@dataclass
class Position:
    quantity: int = 0
    available_quantity: int = 0
    avg_cost: float = 0.0
    entry_date: str | None = None
    highest_close: float = 0.0
    days_held: int = 0
    below_ma60_streak: int = 0


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0

    def market_value(self, prices: dict[str, float]) -> float:
        return sum(p.quantity * prices.get(s, 0.0) for s, p in self.positions.items())

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.market_value(prices)

    def weight(self, symbol: str, prices: dict[str, float]) -> float:
        eq = self.equity(prices)
        return 0.0 if eq <= 0 else self.positions.get(symbol, Position()).quantity * prices.get(symbol, 0) / eq

    def mark_t1(self, date_value: str | pd.Timestamp | None = None, close_prices: dict[str, float] | None = None) -> None:
        close_prices = close_prices or {}
        for symbol, p in self.positions.items():
            p.available_quantity = p.quantity
            p.days_held += 1
            close = close_prices.get(symbol, 0.0)
            if close > 0:
                p.highest_close = max(p.highest_close, close)
