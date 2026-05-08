"""
Phase 01 — Data Sources
Equity  : Yahoo Finance v8 chart API (no yfinance dependency)
Crypto  : ccxt / Binance public REST (no API key required)
Macro   : FRED via pandas-datareader (VIX proxy, yield spread, CPI)
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Optional

import ccxt
import pandas as pd
import pandas_datareader.data as web
import requests

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_PERIOD_DAYS: dict[str, int] = {
    "1d": 1, "5d": 5, "1mo": 30, "3mo": 90,
    "6mo": 180, "1y": 365, "2y": 730, "5y": 1825, "10y": 3650,
}

_YF_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _to_timestamps(period: str) -> tuple[int, int]:
    days = _PERIOD_DAYS.get(period, 365)
    end = int(datetime.utcnow().timestamp())
    start = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    return start, end


# ---------------------------------------------------------------------------
# Equity — Yahoo Finance v8 chart API
# ---------------------------------------------------------------------------

def fetch_equity_daily(
    ticker: str,
    period: str = "5y",
    retries: int = 3,
) -> pd.DataFrame:
    """
    Pull daily OHLCV for an equity ticker from Yahoo Finance.
    Returns a DataFrame indexed by UTC date with columns:
        Open  High  Low  Close  Adj_Close  Volume
    Adj_Close is the split/dividend-adjusted close supplied directly by Yahoo.
    """
    start, end = _to_timestamps(period)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&period1={start}&period2={end}&events=splits,dividends"
    )
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=_YF_HEADERS, timeout=15)
            r.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

    chart = r.json()["chart"]["result"][0]
    timestamps = chart["timestamp"]
    q = chart["indicators"]["quote"][0]
    adj = chart["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])

    df = pd.DataFrame(
        {
            "Open": q["open"],
            "High": q["high"],
            "Low": q["low"],
            "Close": q["close"],
            "Adj_Close": adj,
            "Volume": q["volume"],
        },
        index=pd.to_datetime(timestamps, unit="s", utc=True).normalize(),
    )
    df.index.name = "Date"
    df["ticker"] = ticker
    return df.dropna(subset=["Open", "Close"])


def fetch_equity_hourly(
    ticker: str,
    period: str = "60d",
    retries: int = 3,
) -> pd.DataFrame:
    """
    Pull hourly OHLCV (Yahoo limits hourly history to ~60 days).
    """
    start, end = _to_timestamps(period)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1h&period1={start}&period2={end}"
    )
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=_YF_HEADERS, timeout=15)
            r.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

    chart = r.json()["chart"]["result"][0]
    timestamps = chart["timestamp"]
    q = chart["indicators"]["quote"][0]

    df = pd.DataFrame(
        {
            "Open": q["open"],
            "High": q["high"],
            "Low": q["low"],
            "Close": q["close"],
            "Volume": q["volume"],
        },
        index=pd.to_datetime(timestamps, unit="s", utc=True),
    )
    df.index.name = "Date"
    df["ticker"] = ticker
    return df.dropna(subset=["Open", "Close"])


# ---------------------------------------------------------------------------
# Crypto — ccxt / Binance public API (no key required)
# ---------------------------------------------------------------------------

# Kraken is used instead of Binance — available without geo-restriction
_KRAKEN = ccxt.kraken({"enableRateLimit": True})

_CCXT_TIMEFRAME: dict[str, str] = {
    "1d": "1d",
    "1h": "1h",
    "4h": "4h",
}

# Kraken returns max 720 bars per call
_KRAKEN_PAGE = 720


def fetch_crypto_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    limit: int = 1825,  # ~5 years of daily bars
) -> pd.DataFrame:
    """
    Pull OHLCV for a crypto pair via ccxt/Kraken public endpoint.
    symbol  : ccxt-style pair, e.g. 'BTC/USDT', 'ETH/USDT'
    timeframe: '1d', '1h', '4h'
    limit   : total number of bars to return
    """
    tf = _CCXT_TIMEFRAME.get(timeframe, timeframe)
    bars: list = []
    since = None
    while len(bars) < limit:
        batch = _KRAKEN.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=_KRAKEN_PAGE)
        if not batch:
            break
        bars.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < _KRAKEN_PAGE:
            break

    bars = bars[-limit:]
    df = pd.DataFrame(bars, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    df.index = (
        pd.to_datetime(df["ts"], unit="ms", utc=True).dt.normalize()
        if tf == "1d"
        else pd.to_datetime(df["ts"], unit="ms", utc=True)
    )
    df.index.name = "Date"
    df = df.drop(columns=["ts"])
    df["Adj_Close"] = df["Close"]  # no split adjustments for crypto
    df["ticker"] = symbol.replace("/", "-")
    return df


# ---------------------------------------------------------------------------
# Macro — FRED via pandas-datareader
# ---------------------------------------------------------------------------

# FRED series used:
#   VIXCLS  : CBOE Volatility Index (daily)
#   T10Y2Y  : 10-Year minus 2-Year Treasury yield spread (daily) — yield curve
#   CPIAUCSL: Consumer Price Index, all urban consumers (monthly)
#   DGS10   : 10-Year Treasury Constant Maturity Rate (daily)
#   DGS2    : 2-Year Treasury Constant Maturity Rate (daily)

_FRED_SERIES = {
    "VIX": "VIXCLS",
    "yield_spread_10_2": "T10Y2Y",
    "CPI": "CPIAUCSL",
    "rate_10y": "DGS10",
    "rate_2y": "DGS2",
}


def fetch_macro_fred(
    start: str = "2019-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch macro indicators from FRED.
    Returns a daily-indexed DataFrame (monthly series like CPI are
    forward-filled to daily frequency).
    Columns: VIX  yield_spread_10_2  CPI  rate_10y  rate_2y
    """
    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%d")

    frames = {}
    for col, series_id in _FRED_SERIES.items():
        try:
            s = web.DataReader(series_id, "fred", start, end)[series_id]
            s.index = pd.to_datetime(s.index, utc=True)
            frames[col] = s
        except Exception as exc:
            print(f"[WARN] FRED series {series_id} failed: {exc}")

    if not frames:
        raise RuntimeError("All FRED series failed — check internet connection.")

    df = pd.concat(frames, axis=1)
    df.index.name = "Date"

    # Reindex to daily business-day frequency and forward-fill gaps
    bday_idx = pd.bdate_range(start=start, end=end, tz="UTC")
    df = df.reindex(bday_idx).ffill()
    return df
