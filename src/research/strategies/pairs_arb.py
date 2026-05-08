"""
Phase 06A — Pairs / Statistical Arbitrage

Single-asset implementation: long the target ticker when it is
statistically cheap relative to a benchmark pair (default: SPY).

Signal Logic
------------
  1. Compute rolling OLS hedge ratio β over `ols_window` bars:
       log(ticker) = α + β·log(spy) + ε
  2. Spread  = log(ticker) − β·log(spy)
  3. Z-score = (spread − μ_spread) / σ_spread   [rolling `z_window` bars]
  4. Entry  : z-score < −`z_entry`   (ticker cheap vs pair)
  5. Exit   : z-score > −`z_exit`    (spread mean-reverts)

Long-only: we only capture the long leg of the pair.
With long_short=True the strategy also shorts when z > +z_entry.

Regime Hypothesis
-----------------
  Works best in low-correlation ranging regimes; breaks down when
  the cointegrating relationship changes (structural breaks).

Academic Reference
------------------
  Gatev, Goetzmann & Rouwenhorst (2006); Vidyamurthy (2004).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.research.strategies.base import BaseStrategy


def _rolling_beta(log_y: pd.Series, log_x: pd.Series, window: int) -> pd.Series:
    """Rolling OLS slope (hedge ratio) via expanding/rolling covariance."""
    cov  = log_y.rolling(window).cov(log_x)
    var  = log_x.rolling(window).var()
    beta = (cov / var.replace(0, np.nan)).ffill()
    return beta


class PairsArbStrategy(BaseStrategy):
    """
    Statistical arbitrage vs a benchmark pair (long-only or long/short).

    Parameters
    ----------
    ols_window  : rolling window for hedge ratio estimation (default 60)
    z_window    : rolling window for spread z-score (default 60)
    z_entry     : |z-score| threshold to enter (default 2.0)
    z_exit      : |z-score| threshold to exit  (default 0.5)
    long_short  : if True, also take short positions when z > +z_entry
    """

    name = "Pairs_StatArb"

    def __init__(
        self,
        ols_window: int = 60,
        z_window: int = 60,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        long_short: bool = False,
    ) -> None:
        self.ols_window  = ols_window
        self.z_window    = z_window
        self.z_entry     = z_entry
        self.z_exit      = z_exit
        self.long_only   = not long_short

    def generate_signals(
        self,
        ohlcv: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None,
        pair_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        close = self._close(ohlcv)

        if pair_df is None:
            # No pair provided — strategy produces all-flat signal
            return pd.Series(0, index=close.index, name="signal")

        pair_close = self._close(pair_df).reindex(close.index).ffill()

        # Log prices
        log_y = np.log(close.replace(0, np.nan)).ffill()
        log_x = np.log(pair_close.replace(0, np.nan)).ffill()

        # Rolling hedge ratio and spread
        beta   = _rolling_beta(log_y, log_x, self.ols_window)
        spread = log_y - beta * log_x

        # Z-score of spread
        s_mean = spread.rolling(self.z_window).mean()
        s_std  = spread.rolling(self.z_window).std()
        z      = (spread - s_mean) / s_std.replace(0, np.nan)

        # Stateful signals
        signals  = np.zeros(len(close), dtype=int)
        in_long  = False
        in_short = False

        for i in range(len(close)):
            zi = z.iloc[i]
            if np.isnan(zi):
                signals[i] = 0
                continue

            if not in_long and not in_short:
                if zi < -self.z_entry:
                    in_long  = True
                elif (not self.long_only) and zi > self.z_entry:
                    in_short = True
            elif in_long and zi >= -self.z_exit:
                in_long = False
            elif in_short and zi <= self.z_exit:
                in_short = False

            signals[i] = 1 if in_long else (-1 if in_short else 0)

        return pd.Series(signals, index=close.index, name="signal")
