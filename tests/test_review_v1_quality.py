"""Second-source price check: agreement passes, a bad print or a missing day fails, and no network is needed."""

import numpy as np
import pandas as pd
import pytest

from review_v1 import quality
from review_v1.registration import SYMBOLS


def frames(sessions=30, seed=3):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2026-09-15", periods=sessions)
    return {s: pd.DataFrame(dict(Close=100 * np.exp(np.cumsum(rng.normal(0, 0.01, sessions)))), index=index) for s in SYMBOLS}


def second_source(data, first="2026-09-15"):
    rows = []
    for symbol, frame in data.items():
        close = frame.Close.loc[first:]
        rows += [dict(symbol=symbol, session=d.strftime("%Y-%m-%d"), close=float(v)) for d, v in close.items()]
    return pd.DataFrame(rows)


FIRST = "2026-09-29"


def test_returns_table_covers_every_instrument_from_the_first_session():
    data = frames()
    table = quality.returns(data, FIRST)
    assert set(table.symbol) == set(SYMBOLS)
    assert table.session.min() == FIRST
    expected = data["SPY"].Close.pct_change().loc[FIRST:]
    assert table.loc[table.symbol == "SPY", "return"].to_numpy() == pytest.approx(expected.to_numpy())


def test_identical_second_source_passes():
    data = frames()
    table = quality.returns(data, FIRST)
    result = quality.compare(table, second_source(data))
    assert result["passed"] and result["rows_compared"] == result["rows_expected"] == len(table)
    assert result["max_abs_return_difference"] < 1e-12 and result["rows_over_tolerance"] == 0
    assert result["symbols"] == len(SYMBOLS)


def test_a_price_level_difference_does_not_matter_but_a_bad_print_does():
    data = frames()
    table = quality.returns(data, FIRST)
    # A different adjustment base scales every close of one instrument: returns are unchanged.
    scaled = second_source(data)
    scaled.loc[scaled.symbol == "NVDA", "close"] *= 0.25
    assert quality.compare(table, scaled)["passed"]
    # One wrong close moves two returns of that instrument by about 2%.
    bad = second_source(data)
    row = bad.index[(bad.symbol == "NVDA") & (bad.session == "2026-10-06")][0]
    bad.loc[row, "close"] *= 1.02
    result = quality.compare(table, bad)
    assert not result["passed"] and result["rows_over_tolerance"] == 2
    assert result["worst"]["symbol"] == "NVDA" and result["worst"]["session"] in ("2026-10-06", "2026-10-07")
    assert result["max_abs_return_difference"] == pytest.approx(0.02, abs=0.002)


def test_a_day_missing_from_the_second_source_fails_the_check():
    data = frames()
    table = quality.returns(data, FIRST)
    partial = second_source(data)
    partial = partial.loc[~((partial.symbol == "AAPL") & (partial.session == partial.session.max()))]
    result = quality.compare(table, partial)
    assert result["rows_compared"] == result["rows_expected"] - 1
    assert not result["passed"]


def test_largest_moves_are_sorted_by_size():
    data = frames()
    data["TSLA"].loc[pd.Timestamp("2026-10-08"), "Close"] *= 1.12
    moves = quality.largest_moves(quality.returns(data, FIRST), count=5)
    assert len(moves) == 5 and moves[0]["symbol"] == "TSLA"
    sizes = [abs(m["return"]) for m in moves]
    assert sizes == sorted(sizes, reverse=True)


def test_report_states_when_the_second_source_is_unavailable(tmp_path, monkeypatch):
    """No cache and no credentials: the report says so and assumes nothing. No request is made."""
    monkeypatch.setattr(quality, "CACHE", tmp_path)
    monkeypatch.setattr("review_v1.sentiment_link.credentials", lambda: ("", ""))
    report = quality.report(frames(), "unit", FIRST)
    assert report["second_source"]["available"] is False
    assert "no data-API credentials" in report["second_source"]["reason"]
    assert "passed" not in report["second_source"]
    assert report["rows"] == len(SYMBOLS) * len(pd.bdate_range(FIRST, "2026-10-26"))
    assert all("second_source_return" not in move for move in report["largest_moves"])


def test_report_uses_a_cached_second_source_without_credentials(tmp_path, monkeypatch):
    from review_v1.paths import write_json

    data = frames()
    monkeypatch.setattr(quality, "CACHE", tmp_path)
    monkeypatch.setattr("review_v1.sentiment_link.credentials", lambda: pytest.fail("credentials must not be read when the cache exists"))
    write_json(tmp_path / "unit.json", dict(rows=second_source(data).to_dict("records")))
    report = quality.report(data, "unit", FIRST)
    assert report["second_source"]["available"] and report["second_source"]["passed"]
    assert "returns" not in report["second_source"]
    assert all(move["second_source_return"] == pytest.approx(move["return"]) for move in report["largest_moves"])
