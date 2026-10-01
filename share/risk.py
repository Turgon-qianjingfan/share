from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class RiskLimits:
    max_single_weight: float = 0.10
    max_equity_weight: float = 0.60
    min_cash_weight: float = 0.40
    max_risk_per_trade: float = 0.005

    def validate(self) -> None:
        if not 0 < self.max_single_weight <= 1:
            raise ValueError("max_single_weight 必须在 (0,1] 内")
        if not 0 <= self.max_equity_weight <= 1:
            raise ValueError("max_equity_weight 必须在 [0,1] 内")
        if self.max_equity_weight + self.min_cash_weight > 1:
            raise ValueError("股票仓位上限与最低现金比例不能同时超过 100%")
        if not 0 < self.max_risk_per_trade <= 0.05:
            raise ValueError("max_risk_per_trade 应在 (0,5%] 内")
