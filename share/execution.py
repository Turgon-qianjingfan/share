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

    @staticmethod
    def _blocked(price, *, suspended=False, limit_up=None, limit_down=None, side):
        """Conservative limit/suspension check using caller-supplied exchange limits."""
        if suspended:
            return True
        if price is None or price <= 0:
            return True
        eps = max(abs(float(price)) * 1e-8, 1e-8)
        if side == "BUY" and limit_up is not None and price >= float(limit_up) - eps:
            return True
        if side == "SELL" and limit_down is not None and price <= float(limit_down) + eps:
            return True
        return False

    def buy(
        self, date, symbol, price, quantity, reason, *,
        suspended=False, limit_up=None,
    ):
        if self._blocked(price, suspended=suspended, limit_up=limit_up, side="BUY"):
            return False
        q = (int(quantity) // self.cfg.lot_size) * self.cfg.lot_size
        if q <= 0:
            return False
        px = price * (1 + self.cfg.slippage_rate)
        gross = px * q
        fee = gross * self.cfg.commission_rate
        if gross + fee > self.portfolio.cash:
            return False

        position = self.portfolio.positions.get(symbol)
        if position is None:
            position = Position(entry_date=str(date), highest_close=price, days_held=0)

        new_qty = position.quantity + q
        position.avg_cost = (
            (position.quantity * position.avg_cost) + gross + fee
        ) / new_qty
        position.quantity = new_qty
        # New shares bought today are not available to sell until the next session.
        position.available_quantity = min(position.available_quantity, position.quantity - q)
        position.highest_close = max(position.highest_close, price)
        self.portfolio.positions[symbol] = position
        self.portfolio.cash -= gross + fee

        self.trades.append({
            "date": str(date), "symbol": symbol, "side": "BUY",
            "quantity": q, "price": px, "fee": fee, "reason": reason,
        })
        return True

    def sell(
        self, date, symbol, price, quantity, reason, *,
        suspended=False, limit_down=None,
    ):
        if self._blocked(price, suspended=suspended, limit_down=limit_down, side="SELL"):
            return False
        position = self.portfolio.positions.get(symbol)
        if position is None:
            return False

        requested = max(0, int(quantity))
        # Mainland A shares may generally dispose of an odd-lot remainder only
        # when liquidating the entire remaining position, not as a partial sale.
        if requested >= position.quantity and position.available_quantity >= position.quantity:
            q = position.quantity
        else:
            q = min(
                (requested // self.cfg.lot_size) * self.cfg.lot_size,
                position.available_quantity,
            )
        if q <= 0:
            return False

        px = price * (1 - self.cfg.slippage_rate)
        gross = px * q
        fee = gross * (self.cfg.commission_rate + self.cfg.stamp_duty_rate)
        cost = position.avg_cost * q

        self.portfolio.realized_pnl += gross - fee - cost
        position.quantity -= q
        position.available_quantity -= q
        self.portfolio.cash += gross - fee

        self.trades.append({
            "date": str(date), "symbol": symbol, "side": "SELL",
            "quantity": q, "price": px, "fee": fee, "reason": reason,
        })

        if position.quantity == 0:
            del self.portfolio.positions[symbol]
        return True
