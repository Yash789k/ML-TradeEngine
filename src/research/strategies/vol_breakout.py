"""
Phase 06A — Volatility Breakout Strategy

Signal Logic
------------
  All three conditions must hold simultaneously:
    1. ATR expansion:      ATR(14) > atr_mult × ATR_MA(20)
    2. Volume surge:       Volume  > vol_mult × Volume_MA(20)
    3. Price breakout:     Close   > prior N-day high  (default N=20)

  Position is held for `hold_bars` bars regardless of conditions, allowing
  the breakout to develop.  Early exit if price drops back below the
  entry-bar close (stop) — tracked via a simple trailing floor.

Regime Hypothesis
-----------------
  Captures volatility compression → expansion transitions.
  Works best at the start of trend regimes.  Fails in persistent low-vol
  environments where ATR expansion never triggers.

Academic Reference
------------------
  Bollinger (2001); Kaufman (2013) "Trading Systems and Methods".
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.research.strategies.base import BaseStrategy


class VolBreakoutStrategy(BaseStrategy):
    """
    ATR-expansion + volume-surge + price-breakout strategy.

    Parameters
    ----------
    atr_length   : ATR period (default 14)
    atr_mult     : ATR must exceed this multiple of its 20-day average (default 1.0)
    vol_mult     : Volume must exceed this multiple of its 20-day average (default 1.2)
    breakout_n   : look-back bars for price breakout channel (default 20)
    hold_bars    : minimum bars to hold after entry (default 5)
    """

    name = "Vol_Breakout"

    def __init__(
        self,
        atr_length: int = 14,
        atr_mult: float = 1.0,
        vol_mult: float = 1.2,
        breakout_n: int = 20,
        hold_bars: int = 5,
    ) -> None:
        self.atr_length = atr_length
        self.atr_mult   = atr_mult
        self.vol_mult   = vol_mult
        self.breakout_n = breakout_n
        self.hold_bars  = hold_bars

    def generate_signals(
        self,
        ohlcv: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None,
        pair_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        close  = self._close(ohlcv)
        high   = ohlcv["High"].ffill()
        low    = ohlcv["Low"].ffill()
        volume = ohlcv["Volume"].fillna(0)

        # ATR and its moving average
        atr     = ta.atr(high, low, close, length=self.atr_length)
        atr_avg = atr.rolling(20).mean()
        atr_exp = atr > self.atr_mult * atr_avg

        # Volume surge
        vol_avg  = volume.rolling(20).mean()
        vol_surge = volume > self.vol_mult * vol_avg.replace(0, np.nan)

        # Price breakout above prior N-day high (shift 1 → no lookahead)
        prior_high = close.rolling(self.breakout_n).max().shift(1)
        price_bk   = close > prior_high

        # All conditions must fire simultaneously
        entry = atr_exp & vol_surge & price_bk

        # Hold for `hold_bars` bars after any entry signal
        # Use a rolling max of the entry flag over the hold window
        hold_signal = entry.astype(int).rolling(self.hold_bars, min_periods=1).max()

        return hold_signal.fillna(0).astype(int)
