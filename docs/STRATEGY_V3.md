# Strategy V3: technical + company events + industry context

V3 adds three decision layers beyond raw price:

- Technical layer: MA20/60, EMA12/26/MACD, ATR and ATR%, RSI, ADX, Bollinger position/width, 5/20/60/120-day returns, 20/60/120-day highs/lows, drawdowns, breakouts, volume ratio, OBV trend, gaps, and relative strength versus a market proxy.
- Company event layer: point-in-time news/facts with direction, severity, source quality, event-type weighting, and exponential time decay. Severe negative company events can force risk exit.
- Industry layer: industry event scores with a slower decay than company-specific news.
- Market regime layer: benchmark trend + breadth determine a dynamic risk budget of 0%, 15%, 30%, or 40%.
- Principal guard: when account equity is below 200,000 yuan, new equity risk is capped at 20% of equity.
- Turnover control: no daily rebalance unless the target/current gap exceeds the configured threshold.
