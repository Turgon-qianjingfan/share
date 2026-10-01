# SHARE — Conservative A-Share Paper Trading

一个面向中国 A 股的保守型量化研究与模拟交易项目。

## 当前阶段

> 数据状态（2026-10-01）：Longbridge 当前账户可取得近期日线，但历史 K 线接口返回 history-candlestick quota=0。因此五年数据集尚未生成；项目不会用模拟数据冒充五年历史。

v0.1 是“可审计、低换手、先回测后模拟”的基线版本：

- 日线 OHLCV 驱动
- 20/60 日趋势确认 + RSI + 波动率约束
- 单票仓位上限、总股票仓位上限、最低现金比例
- ATR/固定止损式风险预算
- A 股 100 股手数约束
- T+1 卖出限制
- 可配置滑点、佣金、印花税参数
- 信号在收盘产生，下一交易日开盘成交，避免把未来价格泄漏给当天信号
- CSV 回测数据源与确定性的测试数据
- 预留 Longbridge 数据适配层；目前不会把任何真实下单接口接入模拟器

## 重要原则

这个项目只用于研究、回测和模拟交易，不自动发送真实订单。任何收益率、胜率或参数，都必须通过历史样本和样本外测试验证，不能视为收益承诺。

基本流程：

```
数据 -> 股票池/风险过滤 -> 信号 -> 仓位/风险 -> 模拟成交 -> 组合净值 -> 评估
```

## 快速开始

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -e ".[dev]"

pytest
python run_backtest.py --demo
```

真实历史数据运行：

```bash
python run_backtest.py --csv data/your_daily.csv --initial-cash 1000000
```

CSV 至少需要：

```
date,symbol,open,high,low,close,volume
```

## 后续迭代

下一阶段再加入：基本面因子、市场状态识别、组合级行业约束、滚动 walk-forward、基准指数对照，以及 Longbridge 历史行情的正式适配。


## Strategy V2

V2 addresses two observed failure modes:

1. High RSI no longer forces an existing position to exit. RSI is used mainly for new-entry filtering.
2. Positions are no longer rebalanced every day. A 10-trading-day minimum holding period, 3-ATR trailing stop, 3-day confirmed trend break, and 2.5-percentage-point rebalance threshold reduce churn.
3. When the account falls below the 200,000-yuan principal, normal new equity exposure is capped at 20% until capital recovers.
4. The strategy is still capital-preservation oriented; V2 is not a guarantee that principal cannot decline.

Recent single-stock regression tests on the latest available 1,000 daily bars:
- 603799.SH: final equity about 200,391 yuan; maximum drawdown about -3.88%; about 20 trades.
- 600988.SH: final equity about 205,626 yuan; maximum drawdown about -3.68%; about 22 trades.

These tests are diagnostic only and do not replace the required five-year, full-universe, walk-forward validation.


## V3 research status

V3 is currently a research candidate, not the production baseline.

The 2026-10-01 preliminary test used 10 diversified A-shares, 1,000 daily bars per stock, and 510300.SH as a market-regime proxy. It also included a point-in-time event feed structure for company and industry news.

Preliminary findings:
- V3 increased the average realised equity exposure relative to V2, but exposure remained well below the 40% ceiling.
- The sample still experienced material drawdown, so V3 is not yet accepted as a safer replacement for V2.
- News/events are now first-class inputs rather than narrative annotations. Company events can affect rank and trigger risk exits; industry events affect the industry component of the score.
- The full five-year, point-in-time dataset is still blocked by the current Longbridge historical market-data entitlement.

Promotion rule: V3 should only replace the main strategy after multi-year walk-forward validation, stress testing, and comparison against V2 on the same data.
