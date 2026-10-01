from __future__ import annotations
from dataclasses import dataclass
from .portfolio import Portfolio, Position

@dataclass(frozen=True)
class ExecutionConfig:
    commission_rate: float = 0.0003
    stamp_duty_rate: float = 0.0005
    slippage_rate: float = 0.0005
    lot_size: int = 100

class Simulator:
    def __init__(self, portfolio: Portfolio, cfg: ExecutionConfig):
        self.portfolio = portfolio
        self.cfg = cfg
        self.trades: list[dict] = []

    def buy(self, date, symbol, price, quantity, reason):
        q = (int(quantity) // self.cfg.lot_size) * self.cfg.lot_size
        if q <= 0: return False
        px = price * (1 + self.cfg.slippage_rate)
        gross = px * q
        fee = gross * self.cfg.commission_rate
        if gross + fee > self.portfolio.cash: return False
        old = self.portfolio.positions.get(symbol, Position())
        new_qty = old.quantity + q
        old.avg_cost = ((old.quantity * old.avg_cost) + gross + fee) / new_qty
        old.quantity = new_qty
        self.portfolio.positions[symbol] = old
        self.portfolio.cash -= gross + fee
        self.trades.append({"date":date,"symbol":symbol,"side":"BUY","quantity":q,"price":px,"fee":fee,"reason":reason})
        return True

    def sell(self, date, symbol, price, quantity, reason):
        p = self.portfolio.positions.get(symbol)
        if not p: return False
        q = min((int(quantity)//self.cfg.lot_size)*self.cfg.lot_size, p.available_quantity)
        if q <= 0: return False
        px = price * (1 - self.cfg.slippage_rate)
        gross = px * q
        fee = gross * (self.cfg.commission_rate + self.cfg.stamp_duty_rate)
        cost = p.avg_cost * q
        self.portfolio.realized_pnl += gross - fee - cost
        p.quantity -= q
        p.available_quantity -= q
        self.portfolio.cash += gross - fee
        self.trades.append({"date":date,"symbol":symbol,"side":"SELL","quantity":q,"price":px,"fee":fee,"reason":reason})
        if p.quantity == 0:
            del self.portfolio.positions[symbol]
        return True
