"""Regression coverage for next-open accounting and publication-time alignment."""

import numpy as np
import pandas as pd

from ceci2_demo.company_text import align
from ceci2_demo.evaluate import COST, START, replay


def test_stock_text_enters_after_publication_and_expires():
    history = pd.DataFrame(
        index=pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-15"])
    )
    docs = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "kind": "news",
                "published": "2024-01-05T22:00:00Z",
                "tone": 0.6,
            }
        ]
    )
    align(history, "AAPL", docs)
    assert list(history.news_available) == [0, 1, 0]
    assert list(history.news_tone) == [0.0, 0.6, 0.0]
    assert not history.filing_available.any()


def test_spy_replay_reconciles_with_two_fees_and_no_rebalancing():
    dates = pd.bdate_range("2024-01-08", periods=3)
    frame = pd.DataFrame(
        {
            "decision": dates,
            "symbol": "SPY",
            "entry": [10, 11, 12],
            "exit": [11, 12, 14],
            "vol": 0.02,
            "up": 0.7,
            "down": 0.1,
            "mark": dates,
        }
    )
    result = replay(frame, "SPY buy and hold")
    expected = START / (1 + COST) * 1.4 * (1 - COST)
    assert np.isclose(result.equity.iloc[-1], expected)
    assert result.trades.tolist() == [1, 0, 1]
    cash = replay(frame, "Cash")
    assert (cash.equity == START).all()
    assert cash.fees.sum() == 0
