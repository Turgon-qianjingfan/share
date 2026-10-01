# Design notes

## Conservative baseline

The baseline is intentionally simple enough to audit:

1. Use only information available by the signal date.
2. Generate signals after the close.
3. Execute at the next trading day's open.
4. Keep at least 40% cash and cap equity exposure at 60%.
5. Cap a single position at 10%.
6. Trade only round lots of 100 shares.
7. Enforce T+1 sell availability.
8. Include commission, stamp duty and slippage.
9. Allow no-trade / all-cash periods.
10. Never submit real orders.

## Data contract

Historical CSV data must contain date, symbol, open, high, low, close and volume. Fundamental data will later be added with publication/announcement dates, not period-end dates, so that the backtest cannot use information that was not public yet.

## Next research gates

Before treating the strategy as useful, run:

- full-history backtest;
- rolling walk-forward tests;
- out-of-sample validation;
- higher-cost and higher-slippage stress tests;
- liquidity/suspension/limit-up/down handling;
- comparison with cash and a broad-market benchmark;
- turnover and drawdown analysis.

A parameter is not accepted merely because it improves one historical backtest.
