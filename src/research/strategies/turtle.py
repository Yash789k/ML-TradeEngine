"""
Phase 06A — Turtle Trading Strategy (Donchian Channel Breakout)

Signal Logic
------------
  Entry : Close > N-day Donchian high  (new N-day high breakout)
          Uses the prior bar's high to avoid lookahead.
  Exit  : Close < M-day Donchian low   (price breaks the shorter exit channel)
          Uses the prior bar's low.

  Default: 20-day entry channel, 10-day exit channel (original Turtle rules).

Position is held long until the exit channel is breached.
ATR-based units are applied externally by the Risk Engine; here we size 100%.

Regime Hypothesis
-----------------
  Works best in strongly trending regimes (bull or volatility-breakout states).
  Suffers in ranging markets due to frequent false breakouts.

Academic Reference
------------------
  Dennis & Eckhardt (1983); Faith (2003) "Way of the Turtle".
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.research.strategies.base import BaseStrategy


class TurtleStrategy(BaseStrategy):
    """
    Donchian channel breakout (Turtle Trading Rules System 1).

    Parameters
    ----------
    entry_bars  : look-back for entry channel (default 20)
    exit_bars   : look-back for exit  channel (default 10)
    """

    name = "Turtle_Breakout"

    def __init__(
        self,
        entry_bars: int = 20,
        exit_bars: int = 10,
    ) -> None:
        self.entry_bars = entry_bars
        self.exit_bars  = exit_bars

    def generate_signals(
        self,
        ohlcv: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None,
        pair_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        close = self._close(ohlcv)

        # Shift by 1 to use prior bars → no lookahead
        entry_high = close.rolling(self.entry_bars).max().shift(1)
        exit_low   = close.rolling(self.exit_bars).min().shift(1)

        entry_cond = close > entry_high
        exit_cond  = close < exit_low

        # Stateful: hold from breakout until channel exit
        signals  = np.zeros(len(close), dtype=int)
        in_trade = False
        for i in range(len(close)):
            if not in_trade and bool(entry_cond.iloc[i]):
                in_trade = True
            elif in_trade and bool(exit_cond.iloc[i]):
                in_trade = False
            signals[i] = 1 if in_trade else 0

        return pd.Series(signals, index=close.index, name="signal").fillna(0)
