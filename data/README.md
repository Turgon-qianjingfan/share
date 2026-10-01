# Historical data contract

The final research dataset must cover **2021-10-01 through 2026-10-01** at daily frequency.

Required fields:

```
date,symbol,open,high,low,close,volume
```

Data rules:

- Do not fabricate missing observations.
- Do not forward-fill prices across suspensions.
- Prefer adjusted data only when the adjustment convention is explicitly documented.
- Fundamental features, when added later, must use publication/announcement dates.
- The dataset should contain delisted/suspended observations where available; otherwise survivorship bias must be reported.
- The final five-year run must have a reproducible dataset manifest with source, retrieval date and row counts.

Current status: Longbridge MCP reports history-candlestick quota = 0 for this account. The repository therefore does **not** contain a fabricated five-year dataset. The latest-candlestick endpoint can return recent data, but that is insufficient to claim a five-year training sample.
