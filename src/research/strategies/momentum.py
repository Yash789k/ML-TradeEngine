"""
Phase 06A — Momentum Strategy (12-1)

Classic cross-sectional momentum adapted for single-asset time-series momentum.

Signal Logic
------------
  12-month total return minus most-recent 1-month return (skip-month momentum).
  Long when the lagged momentum score is positive; cash otherwise.

  momentum(t) = Close(t-21) / Close(t-252) − 1  (12 months ago → 1 month ago)

  Entry  : momentum(t) > threshold  (default 0)
  Exit   : momentum(t) <= threshold

Regime Hypothesis
-----------------
  Works best in trending / bull regimes; fails in sharp reversals.

Academic Reference
------------------
  Jegadeesh & Titman (1993), Asness, Moskowitz & Pedersen (2013).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.research.strategies.base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    Time-series momentum (12-1).

    Parameters
    ----------
    lookback_long  : long lookback in trading days (default 252 ≈ 12 months)
    lookback_skip  : skip period in trading days  (default 21  ≈ 1 month)
    threshold      : minimum momentum score to go long (default 0)
    """

    name = "Momentum_12_1"

    def __init__(
        self,
        lookback_long: int = 252,
        lookback_skip: int = 21,
        threshold: float = 0.0,
    ) -> None:
        self.lookback_long = lookback_long
        self.lookback_skip = lookback_skip
        self.threshold = threshold

    def generate_signals(
        self,
        ohlcv: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None,
        pair_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        close = self._close(ohlcv)

        # Skip-month momentum: return from 12m ago to 1m ago
        long_ago  = close.shift(self.lookback_long)
        short_ago = close.shift(self.lookback_skip)
        momentum  = short_ago / long_ago - 1.0

        signal = (momentum > self.threshold).astype(int)
        signal = signal.fillna(0)
        return signal
