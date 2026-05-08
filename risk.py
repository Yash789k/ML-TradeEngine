"""
Phase 05 — Risk Management CLI

Runs the risk-managed portfolio simulation on top of Phase 04 signals,
applying Kelly sizing, ATR trailing stops, portfolio heat limits, and
the max-drawdown circuit breaker.

Usage
-----
  # Single ticker with defaults
  python3 risk.py --ticker AAPL

  # All tickers
  python3 risk.py --all-tickers

  # Custom parameters
  python3 risk.py --ticker AAPL MSFT --kelly 0.25 --max-position 0.15

  # Aggressive: full-Kelly, no commission reduction
  python3 risk.py --all-tickers --kelly 0.5 --atr-mult 1.5

Outputs (per ticker) written to data/risk/{ticker}/
  equity_curve.parquet     — daily NAV
  var_cvar.parquet         — rolling VaR (95%) and CVaR (99%)
  trade_log.parquet        — entry / exit events with stop-hit flags
  risk_summary.json        — metrics, VaR report, before/after comparison

Requirements
------------
  Phase 04 must have been run:
    - data/backtest/{ticker}/backtest_summary.json (win/loss ratio)
    - data/backtest/{ticker}/signals.parquet       (signals + probabilities)
  Phase 02 feature caches must exist:
    - data/features/{ticker}_features.parquet      (ATR values)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("risk")

_PROJECT_ROOT  = Path(__file__).resolve().parent
_FEATURES_DIR  = _PROJECT_ROOT / "data" / "features"
_BACKTEST_DIR  = _PROJECT_ROOT / "data" / "backtest"

_DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_features(ticker: str) -> "pd.DataFrame":
    import pandas as pd
    import pyarrow.parquet as pq

    path = _FEATURES_DIR / f"{ticker}_features.parquet"
    if not path.exists():
        log.error("Feature cache missing: %s — run train.py first.", path)
        sys.exit(1)

    df = pq.read_table(str(path)).to_pandas()
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def _load_signals(ticker: str) -> "pd.DataFrame":
    import pandas as pd
    import pyarrow.parquet as pq

    path = _BACKTEST_DIR / ticker / "signals.parquet"
    if not path.exists():
        log.error("Signal cache missing: %s — run backtest.py first.", path)
        sys.exit(1)

    df = pq.read_table(str(path)).to_pandas()
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def _load_phase04_metrics(ticker: str) -> dict:
    import json

    path = _BACKTEST_DIR / ticker / "backtest_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("strategy_metrics", {})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 05 — ML Trade Engine Risk Manager",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ticker",       nargs="+", default=["AAPL"])
    p.add_argument("--all-tickers",  action="store_true",
                   help=f"Run all default tickers: {_DEFAULT_TICKERS}")
    p.add_argument("--capital",      type=float, default=10_000.0)
    p.add_argument("--commission",   type=float, default=0.001)
    p.add_argument("--slippage",     type=float, default=0.0005)
    p.add_argument("--kelly",        type=float, default=0.5,
                   help="Kelly multiplier (0.5 = half-Kelly, 0.25 = quarter-Kelly)")
    p.add_argument("--max-position", type=float, default=0.95,
                   help="Fraction of cash to deploy per trade (single-ticker default 0.95)")
    p.add_argument("--atr-mult",     type=float, default=2.0,
                   help="ATR multiplier for trailing stop")
    p.add_argument("--max-heat",     type=float, default=0.20,
                   help="Max portfolio heat fraction")
    p.add_argument("--cb-threshold", type=float, default=0.85,
                   help="Circuit breaker threshold (equity / peak)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args    = parse_args()
    tickers = _DEFAULT_TICKERS if args.all_tickers else args.ticker

    from src.backtest.metrics import compute_metrics
    from src.risk.engine      import RiskEngine

    engine = RiskEngine(
        initial_capital  = args.capital,
        commission       = args.commission,
        slippage         = args.slippage,
        kelly_multiplier = args.kelly,
        position_frac    = args.max_position,
        atr_multiplier   = args.atr_mult,
        cb_threshold     = args.cb_threshold,
    )

    all_results = {}

    for ticker in tickers:
        try:
            log.info("══════════════════════════════════════════")
            log.info("  Risk analysis: %s", ticker)
            log.info("══════════════════════════════════════════")

            signal_df = _load_signals(ticker)
            feat_df   = _load_features(ticker)
            p04_m     = _load_phase04_metrics(ticker)

            result = engine.run(ticker, signal_df, feat_df)

            risk_m = compute_metrics(
                result.daily_returns,
                result.trade_log if len(result.trade_log) > 0 else None,
            )

            log.info("  [Phase 04 raw]  Sharpe=%+.3f  CAGR=%+.1f%%  MaxDD=%.1f%%",
                     p04_m.get("sharpe_ratio", 0),
                     p04_m.get("cagr", 0) * 100,
                     p04_m.get("max_drawdown", 0) * 100)
            log.info("  [Phase 05 risk] Sharpe=%+.3f  CAGR=%+.1f%%  MaxDD=%.1f%%  "
                     "VaR95=%.1f%%  CVaR99=%.1f%%",
                     risk_m["sharpe_ratio"],
                     risk_m["cagr"] * 100,
                     risk_m["max_drawdown"] * 100,
                     result.risk_report.get("var_95_pct", 0) * 100,
                     result.risk_report.get("cvar_99_pct", 0) * 100)

            engine.save_results(ticker, result, p04_m)
            all_results[ticker] = (p04_m, risk_m, result)

        except Exception as exc:
            log.error("Risk analysis failed for %s: %s", ticker, exc, exc_info=True)
            continue

    # ── Summary comparison table ───────────────────────────────────────
    if not all_results:
        return

    log.info("")
    log.info("══════════════════════════════════════════════════════════════════════")
    log.info("  PHASE 05 RISK SUMMARY — Before vs After Risk Controls")
    log.info("══════════════════════════════════════════════════════════════════════")
    log.info("  %-6s  %-23s  %-23s  %s",
             "Ticker", "── Phase 04 raw ──", "── Phase 05 risk ──", "Gate")
    log.info("  %-6s  %-7s %-7s %-7s  %-7s %-7s %-7s  %s",
             "", "Sharpe", "CAGR%", "MaxDD%",
             "Sharpe", "CAGR%", "MaxDD%", "Sharpe≥1.0")
    log.info("  " + "-" * 68)

    for ticker, (p04, r05, _) in all_results.items():
        gate = "✓ PASS" if r05["sharpe_ratio"] >= 1.0 else "✗ fail"
        log.info(
            "  %-6s  %+7.3f %+7.1f %+7.1f  %+7.3f %+7.1f %+7.1f  %s",
            ticker,
            p04.get("sharpe_ratio", 0),   p04.get("cagr", 0) * 100,
            p04.get("max_drawdown", 0) * 100,
            r05["sharpe_ratio"],           r05["cagr"] * 100,
            r05["max_drawdown"] * 100,
            gate,
        )

    log.info("══════════════════════════════════════════════════════════════════════")
    log.info("Results saved to: data/risk/")
    log.info("Next step → Phase 06: Live Signal Engine")


if __name__ == "__main__":
    main()
