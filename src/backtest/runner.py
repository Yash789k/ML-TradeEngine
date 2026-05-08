"""
Phase 04 — BacktestRunner

Orchestrates the full Phase 04 pipeline for one or more tickers:

  1. Load feature matrix from Phase 02 cache
  2. Build ensemble signals from Phase 03 models (OOS or final-model fallback)
  3. Simulate portfolio P&L (long-only and long/short)
  4. Compare against buy-and-hold benchmark
  5. Run 1000-path Monte Carlo bootstrap
  6. Sweep confidence threshold × commission parameters
  7. Persist results to data/backtest/{ticker}/

All results are serialised to Parquet (equity curves, trade logs) and JSON
(metrics, MC summary, sweep table) so Phase 07 (Dashboard) can load them
without re-running the simulation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKTEST_DIR = _PROJECT_ROOT / "data" / "backtest"
_MODELS_ROOT  = _PROJECT_ROOT / "data" / "models"


class BacktestRunner:
    """
    Runs the full Phase 04 backtest for a single ticker.

    Parameters
    ----------
    initial_capital     : starting portfolio value in USD
    commission          : per-side commission fraction (default 10 bps)
    slippage            : per-side slippage fraction (default 5 bps)
    confidence_threshold: min ensemble confidence to emit non-FLAT signal
    mode                : 'long_only' or 'long_short'
    mc_paths            : number of Monte Carlo bootstrap paths
    mc_years            : forward-projection horizon for Monte Carlo
    risk_free_rate      : annual risk-free rate (default 5 %)
    """

    def __init__(
        self,
        initial_capital: float      = 10_000.0,
        commission: float           = 0.001,
        slippage: float             = 0.0005,
        confidence_threshold: float = 0.38,
        mode: str                   = "long_only",
        mc_paths: int               = 1000,
        mc_years: float             = 3.0,
        risk_free_rate: float       = 0.05,
    ):
        self.initial_capital      = initial_capital
        self.commission           = commission
        self.slippage             = slippage
        self.confidence_threshold = confidence_threshold
        self.mode                 = mode
        self.mc_paths             = mc_paths
        self.mc_years             = mc_years
        self.risk_free_rate       = risk_free_rate

    # ------------------------------------------------------------------

    def run(self, ticker: str, feat_df: pd.DataFrame) -> dict:
        """
        Execute the full Phase 04 pipeline for `ticker`.

        Parameters
        ----------
        ticker  : asset symbol (must match a trained model directory)
        feat_df : Phase 02 feature matrix (must include 'Close' column)

        Returns
        -------
        results dict with sub-keys:
          signals, sim_result, benchmark_equity, metrics_comparison,
          monte_carlo, sweep_table, output_dir
        """
        from src.backtest.metrics    import compare_to_benchmark, compute_metrics
        from src.backtest.montecarlo import run_monte_carlo
        from src.backtest.signals    import build_signal_df
        from src.backtest.simulator  import (
            buy_and_hold,
            run_simulation,
            sweep_parameters,
        )

        log.info("══════════════════════════════════════════")
        log.info("  Backtesting: %s", ticker)
        log.info("══════════════════════════════════════════")

        # ── 1. Build signal DataFrame ──────────────────────────────────
        signal_df = build_signal_df(
            ticker,
            feat_df,
            confidence_threshold=self.confidence_threshold,
        )

        if len(signal_df) < 20:
            raise ValueError(
                f"{ticker}: insufficient signal rows ({len(signal_df)}). "
                "Ensure OOS predictions or final models exist."
            )

        # ── 2. Simulate strategy ───────────────────────────────────────
        sim = run_simulation(
            signal_df,
            initial_capital=self.initial_capital,
            commission     =self.commission,
            slippage       =self.slippage,
            mode           =self.mode,
        )

        # ── 3. Benchmark: buy-and-hold ─────────────────────────────────
        bh_equity   = buy_and_hold(signal_df, self.initial_capital)
        bh_returns  = bh_equity.pct_change().fillna(0.0)

        # ── 4. Metrics comparison ──────────────────────────────────────
        metrics_cmp = compare_to_benchmark(
            strategy_returns  = sim.daily_returns,
            benchmark_returns = bh_returns,
            strategy_name     = f"ML ({self.mode})",
            benchmark_name    = "Buy & Hold",
            trade_log         = sim.trade_log,
            risk_free         = self.risk_free_rate,
        )
        strat_metrics = compute_metrics(
            sim.daily_returns, sim.trade_log, self.risk_free_rate
        )

        log.info("  [Strategy]    Sharpe=%.3f  CAGR=%.1f%%  MaxDD=%.1f%%  WinRate=%.1f%%",
                 strat_metrics["sharpe_ratio"],
                 strat_metrics["cagr"] * 100,
                 strat_metrics["max_drawdown"] * 100,
                 strat_metrics.get("win_rate", 0.0) * 100)
        log.info("  [Buy & Hold]  Sharpe=%.3f  CAGR=%.1f%%  MaxDD=%.1f%%",
                 compute_metrics(bh_returns)["sharpe_ratio"],
                 compute_metrics(bh_returns)["cagr"] * 100,
                 compute_metrics(bh_returns)["max_drawdown"] * 100)

        # ── 5. Monte Carlo bootstrap ───────────────────────────────────
        if self.mc_paths > 0:
            log.info("  Running %d-path Monte Carlo (%.0f-year horizon) …",
                     self.mc_paths, self.mc_years)
            mc = run_monte_carlo(
                sim.daily_returns,
                n_paths         = self.mc_paths,
                n_years         = self.mc_years,
                initial_capital = self.initial_capital,
                risk_free       = self.risk_free_rate,
            )
            s = mc["summary"]
            log.info(
                "  MC Sharpe  p5=%.2f  median=%.2f  p95=%.2f",
                s["sharpe_p5"], s["sharpe_median"], s["sharpe_p95"],
            )
            log.info(
                "  MC CAGR    p5=%.1f%%  median=%.1f%%  p95=%.1f%%",
                s["cagr_p5"] * 100, s["cagr_median"] * 100, s["cagr_p95"] * 100,
            )
            log.info("  Prob(ruin) = %.1f%%", s["prob_ruin"] * 100)
        else:
            log.info("  Monte Carlo skipped (--no-mc).")
            mc = {"summary": {}, "percentiles_df": None, "equity_paths": None}

        # ── 6. Parameter sweep ─────────────────────────────────────────
        log.info("  Running threshold × commission parameter sweep …")
        sweep_df = sweep_parameters(
            signal_df,
            thresholds  = [0.35, 0.40, 0.45, 0.50],
            commissions = [0.0005, 0.001, 0.002],
            initial_capital = self.initial_capital,
            mode        = self.mode,
        )
        best_row = sweep_df.loc[sweep_df["sharpe"].idxmax()]
        log.info(
            "  Best sweep combo: threshold=%.2f  commission=%.4f  "
            "Sharpe=%.3f  CAGR=%.1f%%",
            best_row["threshold"], best_row["commission"],
            best_row["sharpe"],    best_row["cagr"] * 100,
        )

        # ── 7. Persist results ─────────────────────────────────────────
        out_dir = _BACKTEST_DIR / ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        self._save_results(
            ticker, out_dir, signal_df, sim, bh_equity,
            metrics_cmp, strat_metrics, mc, sweep_df,
        )

        return {
            "ticker":             ticker,
            "signal_df":          signal_df,
            "sim_result":         sim,
            "benchmark_equity":   bh_equity,
            "metrics_comparison": metrics_cmp,
            "strategy_metrics":   strat_metrics,
            "monte_carlo":        mc,
            "sweep_table":        sweep_df,
            "output_dir":         str(out_dir),
        }

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_results(
        self,
        ticker: str,
        out_dir: Path,
        signal_df,
        sim,
        bh_equity,
        metrics_cmp,
        strat_metrics,
        mc,
        sweep_df,
    ):
        import pyarrow as pa
        import pyarrow.parquet as pq

        def _write_parquet(df: pd.DataFrame, name: str):
            tbl = pa.Table.from_pandas(df, preserve_index=True)
            pq.write_table(tbl, str(out_dir / f"{name}.parquet"), compression="snappy")

        # Equity curves (strategy + benchmark, indexed by date)
        curves = pd.DataFrame({
            "strategy": sim.equity_curve,
            "buy_hold": bh_equity,
        })
        _write_parquet(curves, "equity_curves")

        # MC percentile curves saved separately (forward-projection, integer index)
        pct_df = mc.get("percentiles_df")
        if pct_df is not None:
            _write_parquet(pct_df.reset_index(drop=True), "mc_percentiles")

        # Signals
        _write_parquet(signal_df[["p_down", "p_flat", "p_up",
                                  "signal", "confidence", "filtered_signal",
                                  "close"]], "signals")

        # Trade log
        if len(sim.trade_log) > 0:
            _write_parquet(sim.trade_log, "trade_log")

        # Metrics comparison
        _write_parquet(metrics_cmp.reset_index(), "metrics_comparison")

        # Sweep table
        _write_parquet(sweep_df, "parameter_sweep")

        # MC summary + strategy metrics → JSON
        summary = {
            "ticker":           ticker,
            "strategy_metrics": strat_metrics,
            "monte_carlo":      mc["summary"],
            "best_sweep": {
                "threshold":  float(sweep_df.loc[sweep_df["sharpe"].idxmax(), "threshold"]),
                "commission": float(sweep_df.loc[sweep_df["sharpe"].idxmax(), "commission"]),
                "sharpe":     float(sweep_df["sharpe"].max()),
            },
            "sim_params": sim.params,
        }
        (out_dir / "backtest_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )

        log.info("  Results saved to %s", out_dir)
