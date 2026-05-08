"""
Phase 01 — Data Pipeline Tests
pytest test_data_pipeline.py

Covers:
  1. Equity fetch returns a non-empty, well-formed DataFrame
  2. Crypto fetch returns a non-empty, well-formed DataFrame
  3. Macro fetch returns expected columns
  4. Parquet round-trip preserves data exactly
  5. Gap-fill: no business-day gaps exist after filling
  6. Split-adjust: all OHLC prices are >= adjusted close * 0.99 (monotone scaling)
  7. No-lookahead: gap-fill uses ffill only — confirmed by checking that any
     NaN introduced at the END of the series remains NaN (not filled forward
     from the future)
  8. DataLoader cache: second call reads from disk (file mtime unchanged)
  9. DataLoader force_refresh: re-fetches and updates mtime
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import pytest

# Point the loader at a temp cache dir so tests are isolated
_TEST_CACHE = Path(__file__).parent / ".test_parquet_cache"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loader():
    from src.data.loader import DataLoader
    _TEST_CACHE.mkdir(exist_ok=True)
    return DataLoader(cache_dir=_TEST_CACHE, max_age_hours=24)


@pytest.fixture(scope="module")
def equity_df(loader):
    return loader.load_equity("AAPL", period="1y")


@pytest.fixture(scope="module")
def crypto_df(loader):
    return loader.load_crypto("BTC/USDT", timeframe="1d", limit=365)


@pytest.fixture(scope="module")
def macro_df(loader):
    return loader.load_macro(start="2023-01-01")


# ---------------------------------------------------------------------------
# 1. Equity — basic shape & dtype checks
# ---------------------------------------------------------------------------

def test_equity_not_empty(equity_df):
    assert len(equity_df) > 0, "equity DataFrame is empty"


def test_equity_required_columns(equity_df):
    required = {"Open", "High", "Low", "Close", "Adj_Close", "Volume"}
    missing = required - set(equity_df.columns)
    assert not missing, f"equity DataFrame missing columns: {missing}"


def test_equity_index_is_datetime(equity_df):
    assert pd.api.types.is_datetime64_any_dtype(equity_df.index), \
        "equity index is not datetime"


def test_equity_no_duplicate_index(equity_df):
    dupes = equity_df.index.duplicated().sum()
    assert dupes == 0, f"equity DataFrame has {dupes} duplicate index entries"


def test_equity_prices_positive(equity_df):
    for col in ["Open", "High", "Low", "Close", "Adj_Close"]:
        neg = (equity_df[col] <= 0).sum()
        assert neg == 0, f"column {col} has {neg} non-positive values"


def test_equity_high_gte_low(equity_df):
    bad = (equity_df["High"] < equity_df["Low"]).sum()
    assert bad == 0, f"{bad} rows where High < Low"


# ---------------------------------------------------------------------------
# 2. Crypto — basic shape & dtype checks
# ---------------------------------------------------------------------------

def test_crypto_not_empty(crypto_df):
    assert len(crypto_df) > 0, "crypto DataFrame is empty"


def test_crypto_required_columns(crypto_df):
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(crypto_df.columns)
    assert not missing, f"crypto DataFrame missing columns: {missing}"


def test_crypto_index_is_datetime(crypto_df):
    assert pd.api.types.is_datetime64_any_dtype(crypto_df.index)


def test_crypto_no_duplicate_index(crypto_df):
    dupes = crypto_df.index.duplicated().sum()
    assert dupes == 0, f"crypto DataFrame has {dupes} duplicate index entries"


def test_crypto_prices_positive(crypto_df):
    for col in ["Open", "High", "Low", "Close"]:
        neg = (crypto_df[col] <= 0).sum()
        assert neg == 0, f"crypto column {col} has {neg} non-positive values"


# ---------------------------------------------------------------------------
# 3. Macro — columns and coverage
# ---------------------------------------------------------------------------

def test_macro_not_empty(macro_df):
    assert len(macro_df) > 0, "macro DataFrame is empty"


def test_macro_expected_columns(macro_df):
    # At least VIX and yield_spread must be present; CPI and rates are a bonus
    for col in ["VIX", "yield_spread_10_2"]:
        assert col in macro_df.columns, f"macro DataFrame missing column: {col}"


def test_macro_index_is_datetime(macro_df):
    assert pd.api.types.is_datetime64_any_dtype(macro_df.index)


def test_macro_no_future_dates(macro_df):
    now = pd.Timestamp.utcnow()
    future = (macro_df.index > now).sum()
    assert future == 0, f"{future} macro rows have future dates"


# ---------------------------------------------------------------------------
# 4. Parquet round-trip
# ---------------------------------------------------------------------------

def test_parquet_roundtrip(tmp_path):
    from src.data.loader import _write_parquet, _read_parquet

    df = pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"], utc=True),
    )
    df.index.name = "Date"

    path = tmp_path / "test_roundtrip.parquet"
    _write_parquet(df, path)
    loaded = _read_parquet(path)

    pd.testing.assert_frame_equal(df, loaded, check_names=False)


# ---------------------------------------------------------------------------
# 5. Gap-fill: no missing business-day entries
# ---------------------------------------------------------------------------

def test_gap_fill_no_business_day_gaps():
    from src.data.loader import _gap_fill

    # Create a series with a deliberate weekday gap (skip 2024-01-03)
    idx = pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-05"], utc=True)
    df = pd.DataFrame({"Close": [100.0, 102.0, 103.0], "Volume": [1000, 1200, 1100]}, index=idx)
    df.index.name = "Date"

    filled = _gap_fill(df, freq="B")

    bday_range = pd.bdate_range(start=idx.min(), end=idx.max(), tz="UTC")
    missing = set(bday_range) - set(filled.index)
    assert len(missing) == 0, f"gap_fill left {len(missing)} missing business-day rows"


def test_gap_fill_volume_zero_on_inserted_rows():
    from src.data.loader import _gap_fill

    idx = pd.to_datetime(["2024-01-02", "2024-01-04"], utc=True)
    df = pd.DataFrame({"Close": [100.0, 102.0], "Volume": [500.0, 600.0]}, index=idx)
    df.index.name = "Date"

    filled = _gap_fill(df, freq="B")
    inserted_date = pd.Timestamp("2024-01-03", tz="UTC")
    assert inserted_date in filled.index
    assert filled.loc[inserted_date, "Volume"] == 0.0, \
        "Volume on gap-filled row should be 0"


# ---------------------------------------------------------------------------
# 6. Split-adjust: OHLC values are scaled, not raw
# ---------------------------------------------------------------------------

def test_split_adjust_ohlc_scaled():
    from src.data.loader import _split_adjust

    # Simulate a 2-for-1 split: Adj_Close = Close / 2
    df = pd.DataFrame({
        "Open":      [200.0, 202.0],
        "High":      [205.0, 207.0],
        "Low":       [198.0, 200.0],
        "Close":     [204.0, 206.0],
        "Adj_Close": [102.0, 103.0],  # half of Close (post-split adjusted)
    }, index=pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True))
    df.index.name = "Date"

    adjusted = _split_adjust(df)

    # After adjustment, Close should equal Adj_Close
    pd.testing.assert_series_equal(
        adjusted["Close"].round(6),
        adjusted["Adj_Close"].round(6),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# 7. No-lookahead: ffill does NOT propagate from the future
# ---------------------------------------------------------------------------

def test_no_lookahead_ffill_direction():
    """
    Place a NaN at the START of the series (position 0).
    After ffill the first row must remain NaN — confirming we only fill
    forward (past → future) and never backward.
    """
    from src.data.loader import _gap_fill
    import numpy as np

    # Build a clean business-day range, then manually introduce a NaN at row 0
    idx = pd.bdate_range("2024-01-02", periods=5, tz="UTC")
    df = pd.DataFrame(
        {"Close": [np.nan, 101.0, 102.0, 103.0, 104.0], "Volume": [0.0] * 5},
        index=idx,
    )
    df.index.name = "Date"

    filled = _gap_fill(df, freq="B")

    # The first row NaN should remain NaN (ffill cannot look backwards)
    assert pd.isna(filled["Close"].iloc[0]), \
        "Lookahead detected: NaN at row 0 was filled from a future value"


def test_no_lookahead_inserted_rows_use_prior_value():
    """
    A gap-filled row must carry the value of the PRIOR row, not a future row.
    """
    from src.data.loader import _gap_fill

    # 2024-01-02 → Close=100, gap on 2024-01-03 (Wednesday), 2024-01-04 → Close=200
    idx = pd.to_datetime(["2024-01-02", "2024-01-04"], utc=True)
    df = pd.DataFrame({"Close": [100.0, 200.0], "Volume": [1.0, 1.0]}, index=idx)
    df.index.name = "Date"

    filled = _gap_fill(df, freq="B")
    gap_date = pd.Timestamp("2024-01-03", tz="UTC")

    # Must be 100 (prior), NOT 200 (future)
    assert filled.loc[gap_date, "Close"] == 100.0, \
        f"Lookahead: gap row filled with future value {filled.loc[gap_date, 'Close']}"


# ---------------------------------------------------------------------------
# 8. DataLoader cache: second call reads from disk (mtime unchanged)
# ---------------------------------------------------------------------------

def test_loader_cache_hit(loader):
    path = _TEST_CACHE / "AAPL_daily.parquet"
    assert path.exists(), "Cache file was not created on first load"

    mtime_before = path.stat().st_mtime
    time.sleep(0.05)
    loader.load_equity("AAPL")  # should hit cache
    mtime_after = path.stat().st_mtime

    assert mtime_before == mtime_after, \
        "Cache file was re-written on second call (expected cache hit)"


# ---------------------------------------------------------------------------
# 9. DataLoader force_refresh: re-fetches and updates mtime
# ---------------------------------------------------------------------------

def test_loader_force_refresh(loader):
    path = _TEST_CACHE / "AAPL_daily.parquet"
    mtime_before = path.stat().st_mtime
    time.sleep(0.1)
    loader.load_equity("AAPL", force_refresh=True)
    mtime_after = path.stat().st_mtime

    assert mtime_after > mtime_before, \
        "force_refresh=True did not update the cache file"
