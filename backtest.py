"""
Phase 04 — Backtesting CLI

Usage
-----
  # Backtest a single ticker with defaults
  python3 backtest.py --ticker AAPL

  # Backtest multiple tickers, long/short mode, higher confidence gate
  python3 backtest.py --ticker AAPL MSFT QQQ --mode long_short --confidence 0.45

  # Full run with custom capital and commission
  python3 backtest.py --ticker AAPL --capital 50000 --commission 0.001 --mc-paths 2000

  # Use all tickers with previously cached features (no network call needed)
  python3 backtest.py --all-tickers

Outputs (per ticker) written to data/backtest/{ticker}/
  equity_curves.parquet    — daily NAV for strategy, buy-and-hold, MC percentiles
  signals.parquet          — daily probability + signal values
  trade_log.parquet        — individual entry/exit events
  metrics_comparison.parquet — strategy vs benchmark table
  parameter_sweep.parquet  — threshold × commission Sharpe matrix
  backtest_summary.json    — metrics + MC CI + best sweep parameters

Requirements
------------
  Trained models must exist under data/models/{ticker}/ from Phase 03.
  Feature caches must exist under data/features/{ticker}_features.parquet from Phase 02.
  If OOS predictions (oos_predictions.parquet) don't exist, the final model
  is used as a fallback (mild look-ahead bias — re-run train.py to fix).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("backtest")

_PROJECT_ROOT = Path(__file__).resolve().parent
_FEATURES_DIR = _PROJECT_ROOT / "data" / "features"
_MODELS_ROOT  = _PROJECT_ROOT / "data" / "models"

_DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]


# ---------------------------------------------------------------------------
# Feature loader
# ---------------------------------------------------------------------------

def _load_features(ticker: str) -> "pd.DataFrame":
    """
    Load cached Phase 02 feature Parquet.  Raises if cache is missing.
    """
    import pyarrow.parquet as pq

    path = _FEATURES_DIR / f"{ticker}_features.parquet"
    if not path.exists():
        log.error("Feature cache not found: %s  →  Run train.py first.", path)
        sys.exit(1)

    df = pq.read_table(str(path)).to_pandas()
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = _to_utc(df.index)
    log.info("Loaded features for %s: %d rows × %d cols", ticker, *df.shape)
    return df


def _to_utc(idx):
    import pandas as pd
    idx = pd.to_datetime(idx)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 04 — ML Trade Engine Backtester",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ticker",       nargs="+", default=["AAPL"],
                   help="Ticker(s) to backtest.")
    p.add_argument("--all-tickers",  action="store_true",
                   help=f"Run all default tickers: {_DEFAULT_TICKERS}")
    p.add_argument("--capital",      type=float, default=10_000.0,
                   help="Starting capital in USD.")
    p.add_argument("--commission",   type=float, default=0.001,
                   help="Per-side commission fraction (10 bps default).")
    p.add_argument("--slippage",     type=float, default=0.0005,
                   help="Per-side slippage fraction (5 bps default).")
    p.add_argument("--confidence",   type=float, default=0.38,
                   help="Min ensemble confidence to emit a non-FLAT signal.")
    p.add_argument("--mode",         choices=["long_only", "long_short"],
                   default="long_only",
                   help="Portfolio mode.")
    p.add_argument("--mc-paths",     type=int, default=1000,
                   help="Monte Carlo bootstrap paths.")
    p.add_argument("--mc-years",     type=float, default=3.0,
                   help="Monte Carlo forward-projection horizon (years).")
    p.add_argument("--risk-free",    type=float, default=0.05,
                   help="Annual risk-free rate for Sharpe/Sortino.")
    p.add_argument("--no-mc",        action="store_true",
                   help="Skip Monte Carlo (faster run).")
    return p.parse_args()


def main():
    args    = parse_args()
    tickers = _DEFAULT_TICKERS if args.all_tickers else args.ticker

    from src.backtest.runner import BacktestRunner

    runner = BacktestRunner(
        initial_capital      = args.capital,
        commission           = args.commission,
        slippage             = args.slippage,
        confidence_threshold = args.confidence,
        mode                 = args.mode,
        mc_paths             = 0 if args.no_mc else args.mc_paths,
        mc_years             = args.mc_years,
        risk_free_rate       = args.risk_free,
    )

    all_results = {}
    for ticker in tickers:
        try:
            feat_df = _load_features(ticker)
            result  = runner.run(ticker, feat_df)
            all_results[ticker] = result
        except Exception as exc:
            log.error("Backtest failed for %s: %s", ticker, exc, exc_info=True)
            continue

    # ── Print summary table ────────────────────────────────────────────
    if all_results:
        log.info("")
        log.info("══════════════════════════════════════════════════════════")
        log.info("  BACKTEST SUMMARY")
        log.info("══════════════════════════════════════════════════════════")
        header = f"{'Ticker':>8}  {'Sharpe':>7}  {'CAGR%':>7}  {'MaxDD%':>7}  "
        header += f"{'WinRate%':>9}  {'Trades':>7}  {'vs B&H':>8}"
        log.info(header)
        log.info("-" * 68)

        import pandas as pd
        for ticker, res in all_results.items():
            sm  = res["strategy_metrics"]
            bh  = res["benchmark_equity"]
            bhr = bh.pct_change().fillna(0.0)
            from src.backtest.metrics import compute_metrics
            bh_m = compute_metrics(bhr)
            edge = sm["sharpe_ratio"] - bh_m["sharpe_ratio"]
            log.info(
                "%8s  %7.3f  %7.1f  %7.1f  %9.1f  %7d  %+8.3f",
                ticker,
                sm["sharpe_ratio"],
                sm["cagr"] * 100,
                sm["max_drawdown"] * 100,
                sm.get("win_rate", 0) * 100,
                sm.get("n_trades", 0),
                edge,
            )

        log.info("══════════════════════════════════════════════════════════")
        log.info("Results saved to: data/backtest/")
        log.info("Launch dashboard: python3 dashboard.py  (Phase 07)")


if __name__ == "__main__":
    main()
