"""Live-order boundary.

This module intentionally contains no order-submission implementation.
Future Longbridge integration should provide market data only to this project
until paper-trading validation is complete. Real order APIs must never be
called from the backtest engine.
"""

class LiveTradingDisabled(RuntimeError):
    pass

def submit_real_order(*args, **kwargs):
    raise LiveTradingDisabled(
        "实盘下单接口在 v0.1 中被明确禁用；本项目只允许研究、回测和模拟交易。"
    )
