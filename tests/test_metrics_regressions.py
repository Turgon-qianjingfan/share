import pandas as pd
import pytest

from share.metrics import summarize


def test_max_drawdown_includes_initial_capital_peak():
    equity = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "equity": [190_000.0, 185_000.0, 195_000.0],
        }
    )

    result = summarize(equity, initial_cash=200_000.0)

    assert result["max_drawdown"] == pytest.approx(-0.075)
    assert result["minimum_equity_vs_principal"] == pytest.approx(-0.075)


def test_empty_equity_returns_neutral_metrics():
    result = summarize(pd.DataFrame(columns=["date", "equity"]), initial_cash=200_000)

    assert result["final_equity"] == 200_000
    assert result["total_return"] == 0
    assert result["max_drawdown"] == 0
