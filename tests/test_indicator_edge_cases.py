import numpy as np
import pandas as pd

from share.indicators import add_indicators


def _frame(closes):
    dates = pd.bdate_range("2024-01-02", periods=len(closes))
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "TEST",
            "open": closes,
            "high": np.asarray(closes) + 1,
            "low": np.asarray(closes) - 1,
            "close": closes,
            "volume": 100_000,
        }
    )


def test_rsi_is_100_for_persistent_uptrend_after_warmup():
    data = add_indicators(_frame(np.arange(1.0, 81.0)), rsi_window=14)

    assert data["rsi"].iloc[-1] == 100.0
    assert not pd.isna(data["rsi"].iloc[-1])


def test_rsi_is_neutral_for_flat_prices_after_warmup():
    data = add_indicators(_frame(np.full(40, 10.0)), rsi_window=14)

    assert data["rsi"].iloc[-1] == 50.0
