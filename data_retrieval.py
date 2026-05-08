"""
Phase 01 — Data Retrieval CLI

Fetches OHLCV and macro data into the Parquet cache via DataLoader.
All data is split-adjusted, gap-filled, and stored in data/parquet/.

Usage
-----
# Default 5 tickers, 5 years
python3 data_retrieval.py

# Expanded universe for Phase 06 research
python3 data_retrieval.py --ticker AAPL MSFT GOOGL AMZN NVDA META TSLA \
  JPM BAC GS XLK XLF XLE XLV XLI XLP GLD TLT IWM DIA BTC-USD ETH-USD \
  --period 10y

# Force refresh (ignore cache)
python3 data_retrieval.py --force-refresh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_TICKERS: list[str] = ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]

_RESEARCH_TICKERS: list[str] = [
    # Tech mega-caps
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # Financials
    "JPM", "BAC", "GS",
    # Sector ETFs (regime diversity)
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP",
    # Rates & safe havens
    "GLD", "TLT",
    # Broad market
    "IWM", "DIA", "SPY", "QQQ",
]

_DEFAULT_CRYPTO: list[str] = ["BTC/USDT", "ETH/USDT"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch and cache OHLCV + macro data for the ML Trade Engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--ticker", nargs="+", default=None,
        metavar="TICK",
        help="Equity tickers to fetch (default: AAPL MSFT GOOGL SPY QQQ). "
             "Use --research for the full 20-ticker universe.",
    )
    p.add_argument(
        "--research", action="store_true",
        help="Fetch the full Phase 06 research universe (~22 tickers).",
    )
    p.add_argument(
        "--period", default="5y",
        choices=["1y", "2y", "5y", "10y"],
        help="Look-back period (default: 5y). Use 10y for research universe.",
    )
    p.add_argument(
        "--no-crypto", action="store_true",
        help="Skip crypto (BTC/ETH).",
    )
    p.add_argument(
        "--no-macro", action="store_true",
        help="Skip FRED macro data (VIX, yield spread, CPI).",
    )
    p.add_argument(
        "--force-refresh", action="store_true",
        help="Re-fetch even if the cached file is fresh.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve ticker list
    if args.research:
        tickers = _RESEARCH_TICKERS
        if args.period == "5y":
            args.period = "10y"   # research needs longer history
        print(f"[data_retrieval] Research universe: {len(tickers)} equity tickers, period={args.period}")
    elif args.ticker:
        tickers = [t.upper() for t in args.ticker]
        print(f"[data_retrieval] Custom tickers: {tickers}, period={args.period}")
    else:
        tickers = _DEFAULT_TICKERS
        print(f"[data_retrieval] Default tickers: {tickers}, period={args.period}")

    crypto_symbols = [] if args.no_crypto else _DEFAULT_CRYPTO

    # Late import to keep startup fast
    from src.data.loader import DataLoader

    loader = DataLoader()

    errors: list[str] = []

    # ── Equity ────────────────────────────────────────────────────────────────
    print("\n── Equity ───────────────────────────────────────────────────────────")
    for ticker in tickers:
        try:
            df = loader.load_equity(ticker, period=args.period,
                                    force_refresh=args.force_refresh)
            date_range = f"{df.index.min().date()} → {df.index.max().date()}"
            print(f"  ✓ {ticker:8s}  {len(df):5d} rows  {date_range}")
        except Exception as exc:
            print(f"  ✗ {ticker:8s}  ERROR: {exc}")
            errors.append(ticker)

    # ── Crypto ────────────────────────────────────────────────────────────────
    if crypto_symbols:
        print("\n── Crypto ───────────────────────────────────────────────────────────")
        for sym in crypto_symbols:
            try:
                df = loader.load_crypto(sym, force_refresh=args.force_refresh)
                date_range = f"{df.index.min().date()} → {df.index.max().date()}"
                print(f"  ✓ {sym:12s}  {len(df):5d} rows  {date_range}")
            except Exception as exc:
                print(f"  ✗ {sym:12s}  ERROR: {exc}")
                errors.append(sym)

    # ── Macro ─────────────────────────────────────────────────────────────────
    if not args.no_macro:
        print("\n── Macro (FRED) ─────────────────────────────────────────────────────")
        try:
            start_macro = "2018-01-01" if args.period in ("5y", "10y") else "2020-01-01"
            df = loader.load_macro(start=start_macro,
                                   force_refresh=args.force_refresh)
            date_range = f"{df.index.min().date()} → {df.index.max().date()}"
            print(f"  ✓ macro      {len(df):5d} rows  {date_range}")
            print(f"    columns: {list(df.columns)}")
        except Exception as exc:
            print(f"  ✗ macro  ERROR: {exc}")
            errors.append("macro")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(tickers) + len(crypto_symbols) + (0 if args.no_macro else 1)
    print(f"\n── Done: {total - len(errors)}/{total} assets cached successfully.")
    if errors:
        print(f"   Failed: {errors}")
        sys.exit(1)


if __name__ == "__main__":
    main()
