"""
Phase 06A — BaseStrategy

Abstract base class for all quant strategies in the Strategy Zoo.

Each concrete strategy only needs to implement `generate_signals()`.
The base class handles:
  - Conversion to the simulator's signal encoding  ({2=long, 1=flat, 0=short})
  - Running the Phase 04 simulator and computing metrics
  - Building a standardised result dict

Signal convention (internal, subclass-facing):
    +1  → go long
     0  → stay flat / cash
    -1  → go short  (only used when long_only=False)

Simulator encoding (external):
     2  → long
     1  → flat
     0  → short
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.metrics import compute_metrics
from src.backtest.simulator import SimResult, buy_and_hold, run_simulation


class BaseStrategy(ABC):
    """
    Abstract base for all Phase 06A strategies.

    Attributes
    ----------
    name       : human-readable identifier used in output tables
    long_only  : if True, short signals (-1) are treated as flat (0)
    """

    name: str = "base"
    long_only: bool = True

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_signals(
        self,
        ohlcv: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None,
        pair_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Compute a daily signal Series from OHLCV (and optionally macro / pair data).

        Parameters
        ----------
        ohlcv   : DataFrame with columns Open, High, Low, Close, Volume
                  indexed by UTC DatetimeIndex (output of DataLoader.load_equity)
        macro   : optional FRED macro DataFrame (VIX, yield_spread_10_2, CPI, ...)
        pair_df : optional second-asset OHLCV for pairs strategies

        Returns
        -------
        pd.Series of int  — values in {+1, 0, -1}, same index as ohlcv.
        No NaN values; pre-signal warm-up rows should be 0.
        """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _close(ohlcv: pd.DataFrame) -> pd.Series:
        """Return split-adjusted close (prefers Adj_Close if present)."""
        if "Adj_Close" in ohlcv.columns:
            return ohlcv["Adj_Close"].ffill()
        return ohlcv["Close"].ffill()

    @staticmethod
    def _encode(signals: pd.Series, long_only: bool) -> pd.Series:
        """Map internal {+1, 0, -1} → simulator {2, 1, 0}."""
        if long_only:
            encoded = np.where(signals == 1, 2, 1)           # -1 treated as flat
        else:
            encoded = np.where(signals == 1, 2,
                     np.where(signals == -1, 0, 1))
        return pd.Series(encoded, index=signals.index, name="filtered_signal")

    def build_signal_df(
        self,
        ohlcv: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None,
        pair_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Build the DataFrame expected by `run_simulation`."""
        sigs = self.generate_signals(ohlcv, macro=macro, pair_df=pair_df)
        sigs = sigs.fillna(0).astype(int)

        close = self._close(ohlcv).reindex(sigs.index).ffill()

        return pd.DataFrame({
            "close":           close,
            "filtered_signal": self._encode(sigs, self.long_only),
        })

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        ohlcv: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None,
        pair_df: Optional[pd.DataFrame] = None,
        initial_capital: float = 10_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ) -> dict:
        """
        Generate signals, simulate, and return metrics + raw SimResult.

        Returns
        -------
        dict with keys:
          "metrics"   : compute_metrics() output dict
          "bh_metrics": buy-and-hold metrics for comparison
          "result"    : SimResult (equity_curve, daily_returns, trade_log)
          "strategy"  : self.name
        """
        signal_df = self.build_signal_df(ohlcv, macro=macro, pair_df=pair_df)
        mode      = "long_only" if self.long_only else "long_short"

        sim_result: SimResult = run_simulation(
            signal_df,
            initial_capital = initial_capital,
            commission       = commission,
            slippage         = slippage,
            mode             = mode,
        )
        bh_equity  = buy_and_hold(signal_df, initial_capital)
        bh_returns = bh_equity.pct_change().fillna(0)

        metrics    = compute_metrics(sim_result.daily_returns, sim_result.trade_log)
        bh_metrics = compute_metrics(bh_returns)

        return {
            "strategy":   self.name,
            "metrics":    metrics,
            "bh_metrics": bh_metrics,
            "result":     sim_result,
        }

    def describe(self) -> str:
        return f"<Strategy name='{self.name}' long_only={self.long_only}>"

    def __repr__(self) -> str:
        return self.describe()
