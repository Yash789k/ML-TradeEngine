"""
Phase 01 — DataLoader
Central class for fetching, caching, and serving clean OHLCV + macro data.

Design decisions:
  - Parquet store partitioned by asset slug and timeframe under data/parquet/
  - Cache is considered stale if the file is older than `max_age_hours`
  - Gap-fill: reindex to business-day (or hourly) range then forward-fill price
    columns; volume gaps are filled with 0
  - Split-adjust: Yahoo supplies Adj_Close directly; we rescale O/H/L/Close by
    the ratio Adj_Close / Close so the entire OHLCV history is split-adjusted
  - No lookahead guarantee: all fills use ffill (past → future only)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.sources import (
    fetch_crypto_ohlcv,
    fetch_equity_daily,
    fetch_equity_hourly,
    fetch_macro_fred,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PARQUET_ROOT = _PROJECT_ROOT / "data" / "parquet"
_PARQUET_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parquet_path(asset_slug: str, timeframe: str) -> Path:
    return _PARQUET_ROOT / f"{asset_slug}_{timeframe}.parquet"


def _is_stale(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return True
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours > max_age_hours


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(table, path, compression="snappy")


def _read_parquet(path: Path) -> pd.DataFrame:
    df = pq.read_table(path).to_pandas()
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


# ---------------------------------------------------------------------------
# Gap-fill & split-adjust
# ---------------------------------------------------------------------------

_PRICE_COLS = ["Open", "High", "Low", "Close", "Adj_Close"]
_VOL_COL = "Volume"


def _gap_fill(df: pd.DataFrame, freq: str = "B") -> pd.DataFrame:
    """
    Reindex to `freq` and forward-fill price columns; zero-fill volume.
    `freq='B'` = business day (equities/macro), `freq='h'` = hourly (crypto intraday).
    Crypto daily data uses 'D' because exchanges trade 24/7.
    """
    start, end = df.index.min(), df.index.max()
    if freq == "D":
        full_idx = pd.date_range(start=start, end=end, freq="D", tz="UTC")
    elif freq == "h":
        full_idx = pd.date_range(start=start, end=end, freq="h", tz="UTC")
    else:
        full_idx = pd.bdate_range(start=start, end=end, tz="UTC")

    df = df.reindex(full_idx)
    price_present = [c for c in _PRICE_COLS if c in df.columns]
    df[price_present] = df[price_present].ffill()
    if _VOL_COL in df.columns:
        df[_VOL_COL] = df[_VOL_COL].fillna(0)
    df.index.name = "Date"
    return df


def _split_adjust(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rescale O/H/L/Close so they match Adj_Close magnitude.
    Ratio = Adj_Close / Close computed row-wise; applied to all price cols.
    When Close == 0 the ratio is left as 1 to avoid division-by-zero.
    """
    if "Adj_Close" not in df.columns or "Close" not in df.columns:
        return df
    ratio = df["Adj_Close"] / df["Close"].replace(0, float("nan"))
    ratio = ratio.fillna(1.0)
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = df[col] * ratio
    return df


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

AssetType = Literal["equity", "crypto", "macro"]


class DataLoader:
    """
    Unified data loader for the ML Trade Engine.

    Usage
    -----
    loader = DataLoader()

    # Equity daily (5 years, split-adjusted, gap-filled)
    aapl = loader.load_equity("AAPL")

    # Crypto daily
    btc = loader.load_crypto("BTC/USDT")

    # Macro (VIX, yield curve, CPI …)
    macro = loader.load_macro()

    Parameters
    ----------
    cache_dir      : override default parquet store location
    max_age_hours  : how old a cached file can be before re-fetching
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_age_hours: float = 12.0,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else _PARQUET_ROOT
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_hours = max_age_hours

    # ------------------------------------------------------------------
    # Equity
    # ------------------------------------------------------------------

    def load_equity(
        self,
        ticker: str,
        period: str = "5y",
        timeframe: str = "daily",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Load equity OHLCV. Checks parquet cache first.
        timeframe : 'daily' | 'hourly'
        """
        slug = ticker.upper().replace(".", "-")
        path = self.cache_dir / f"{slug}_{timeframe}.parquet"

        if not force_refresh and not _is_stale(path, self.max_age_hours):
            return _read_parquet(path)

        if timeframe == "hourly":
            df = fetch_equity_hourly(ticker, period=period)
            df = _gap_fill(df, freq="h")
        else:
            df = fetch_equity_daily(ticker, period=period)
            df = _gap_fill(df, freq="B")
            df = _split_adjust(df)

        _write_parquet(df, path)
        return df

    # ------------------------------------------------------------------
    # Crypto
    # ------------------------------------------------------------------

    def load_crypto(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1d",
        limit: int = 1825,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Load crypto OHLCV from Binance via ccxt.
        timeframe : ccxt timeframe string, e.g. '1d', '1h', '4h'
        """
        slug = symbol.replace("/", "-")
        path = self.cache_dir / f"{slug}_{timeframe}.parquet"

        if not force_refresh and not _is_stale(path, self.max_age_hours):
            return _read_parquet(path)

        df = fetch_crypto_ohlcv(symbol, timeframe=timeframe, limit=limit)
        freq = "D" if timeframe == "1d" else "h"
        df = _gap_fill(df, freq=freq)

        _write_parquet(df, path)
        return df

    # ------------------------------------------------------------------
    # Macro
    # ------------------------------------------------------------------

    def load_macro(
        self,
        start: str = "2019-01-01",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Load FRED macro data: VIX, yield spread (10Y-2Y), CPI, rates.
        """
        path = self.cache_dir / "macro_daily.parquet"

        if not force_refresh and not _is_stale(path, self.max_age_hours):
            return _read_parquet(path)

        df = fetch_macro_fred(start=start)
        _write_parquet(df, path)
        return df

    # ------------------------------------------------------------------
    # Convenience: load all default assets at once
    # ------------------------------------------------------------------

    def load_all(
        self,
        equity_tickers: list[str] | None = None,
        crypto_symbols: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch and cache all default assets.
        Returns a dict keyed by asset slug.
        """
        if equity_tickers is None:
            equity_tickers = ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]
        if crypto_symbols is None:
            crypto_symbols = ["BTC/USDT", "ETH/USDT"]

        result: dict[str, pd.DataFrame] = {}

        for ticker in equity_tickers:
            print(f"  [equity] fetching {ticker} …")
            result[ticker] = self.load_equity(ticker)

        for sym in crypto_symbols:
            slug = sym.replace("/", "-")
            print(f"  [crypto] fetching {sym} …")
            result[slug] = self.load_crypto(sym)

        print("  [macro] fetching FRED macro data …")
        result["macro"] = self.load_macro()

        return result
