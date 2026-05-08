"""
Phase 06A — Strategy Zoo Runner

Orchestrates running all strategies across all tickers and persisting results.

Design
------
  ZooRunner.run() iterates over every (ticker, strategy) pair and:
    1. Calls strategy.run(ohlcv, macro, pair_df) to generate signals + metrics
    2. Saves the equity curve and trade log as Parquet
    3. Accumulates a summary scorecard DataFrame

  PairsArbStrategy receives `pair_df=spy_df` automatically (SPY as default pair).
  CarryProxyStrategy receives `macro` automatically.
  All other strategies receive OHLCV only.

Output Layout
-------------
  data/research/{ticker}/{strategy_name}/
    equity_curve.parquet
    trade_log.parquet
    metrics.json

  data/research/scorecard.parquet   ← consolidated across all runs
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from src.backtest.simulator import buy_and_hold, SimResult
from src.research.strategies.base import BaseStrategy
from src.research.strategies.pairs_arb import PairsArbStrategy
from src.research.strategies.carry_proxy import CarryProxyStrategy

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RESEARCH_ROOT = _PROJECT_ROOT / "data" / "research"


# ---------------------------------------------------------------------------
# Zoo Runner
# ---------------------------------------------------------------------------

class ZooRunner:
    """
    Run all strategies on all tickers and persist results.

    Parameters
    ----------
    strategies       : list of BaseStrategy instances
    initial_capital  : starting equity per run (default $10,000)
    commission       : commission fraction per side (default 0.1%)
    slippage         : slippage fraction per side  (default 0.05%)
    output_dir       : override default output path
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        initial_capital: float = 10_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.strategies      = strategies
        self.initial_capital = initial_capital
        self.commission      = commission
        self.slippage        = slippage
        self.output_dir      = Path(output_dir) if output_dir else _RESEARCH_ROOT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        ohlcv_dict: dict[str, pd.DataFrame],
        spy_df: Optional[pd.DataFrame] = None,
        macro: Optional[pd.DataFrame] = None,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Run all strategies on all tickers.

        Parameters
        ----------
        ohlcv_dict : mapping ticker → OHLCV DataFrame (DataLoader output)
        spy_df     : SPY OHLCV for pairs strategies (optional; None = skip pairs)
        macro      : FRED macro DataFrame for carry strategy (optional)
        save       : persist equity curves and metrics to disk

        Returns
        -------
        scorecard_df : DataFrame indexed by (ticker, strategy) with all metrics
        """
        rows: list[dict] = []
        total = len(ohlcv_dict) * len(self.strategies)
        done  = 0

        for ticker, ohlcv in ohlcv_dict.items():
            for strategy in self.strategies:
                t0 = time.time()
                log.info("  [ZooRunner] %s × %s …", ticker, strategy.name)

                # Determine extra kwargs for specialised strategies
                pair_df = spy_df if isinstance(strategy, PairsArbStrategy) else None
                mac     = macro  if isinstance(strategy, CarryProxyStrategy) else None

                try:
                    out = strategy.run(
                        ohlcv,
                        macro            = mac,
                        pair_df          = pair_df,
                        initial_capital  = self.initial_capital,
                        commission       = self.commission,
                        slippage         = self.slippage,
                    )
                    metrics: dict = out["metrics"]
                    sim_result: SimResult = out["result"]

                except Exception as exc:
                    log.warning("    FAILED %s × %s: %s", ticker, strategy.name, exc)
                    metrics     = {"error": str(exc)}
                    sim_result  = None  # type: ignore[assignment]

                elapsed = round(time.time() - t0, 2)

                # Save to disk
                if save and sim_result is not None:
                    self._persist(ticker, strategy.name, sim_result, metrics)

                # Accumulate scorecard row
                row = {"ticker": ticker, "strategy": strategy.name, **metrics}
                rows.append(row)
                done += 1
                pct   = 100 * done / total
                log.info(
                    "    done (%ds)  Sharpe=%.3f  CAGR=%.1f%%  [%d/%d %.0f%%]",
                    elapsed,
                    metrics.get("sharpe_ratio", float("nan")),
                    metrics.get("cagr", float("nan")) * 100,
                    done, total, pct,
                )

        scorecard_df = pd.DataFrame(rows).set_index(["ticker", "strategy"])

        if save:
            sc_path = self.output_dir / "scorecard.parquet"
            sc_path.parent.mkdir(parents=True, exist_ok=True)
            scorecard_df.to_parquet(sc_path)
            log.info("[ZooRunner] Scorecard saved → %s", sc_path)

        return scorecard_df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(
        self,
        ticker: str,
        strategy_name: str,
        result: SimResult,
        metrics: dict,
    ) -> None:
        out_dir = self.output_dir / ticker / strategy_name
        out_dir.mkdir(parents=True, exist_ok=True)

        result.equity_curve.to_frame("equity").to_parquet(
            out_dir / "equity_curve.parquet"
        )

        if result.trade_log is not None and len(result.trade_log) > 0:
            result.trade_log.to_parquet(out_dir / "trade_log.parquet")

        with open(out_dir / "metrics.json", "w") as f:
            json.dump(
                {k: (float(v) if isinstance(v, float) else v)
                 for k, v in metrics.items()},
                f, indent=2,
            )
