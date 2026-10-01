from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd

@dataclass
class Position:
    quantity: int = 0
    available_quantity: int = 0
    avg_cost: float = 0.0

@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0

    def market_value(self, prices: dict[str,float]) -> float:
        return sum(p.quantity * prices.get(s, 0.0) for s,p in self.positions.items())

    def equity(self, prices: dict[str,float]) -> float:
        return self.cash + self.market_value(prices)

    def weight(self, symbol: str, prices: dict[str,float]) -> float:
        eq = self.equity(prices)
        return 0.0 if eq <= 0 else self.positions.get(symbol, Position()).quantity * prices.get(symbol,0) / eq

    def mark_t1(self) -> None:
        for p in self.positions.values():
            p.available_quantity = p.quantity
