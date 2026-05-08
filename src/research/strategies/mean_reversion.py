"""
Phase 06A — Mean Reversion Strategy (Bollinger Bands + RSI)

Signal Logic
------------
  Entry : Close < lower Bollinger Band  AND  RSI(14) < 30
          (price is 2σ below 20-day mean and oversold)
  Hold  : stay long until price recovers to middle band OR RSI > 70
  Exit  : Close ≥ middle Bollinger Band  OR  RSI(14) > 70

Bollinger Bands use a 20-day SMA ± 2σ.
RSI uses a 14-day Wilder smoothing (pandas_ta default).

Regime Hypothesis
-----------------
  Works best in high-volatility ranging regimes; fails in strong trends.

Academic Reference
------------------
  Bollinger (2001); Connors & Alvarez (2009) mean-reversion frameworks.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.research.strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    Bollinger Band + RSI mean-reversion (long-only).

    Parameters
    ----------
    bb_length   : Bollinger Band period (default 20)
    bb_std      : number of standard deviations (default 2)
    rsi_length  : RSI period (default 14)
    rsi_entry   : RSI threshold for entry — enter when RSI < rsi_entry (default 30)
    rsi_exit    : RSI threshold for exit  — exit  when RSI > rsi_exit  (default 70)
    """

    name = "Mean_Reversion"

    def __init__(
        self,
        bb_length: int = 20,
        bb_std: float = 2.0,
        rsi_length: int = 14,
        rsi_entry: float = 30.0,
        rsi_exit: float = 70.0,
    ) -> None:
        self.bb_length  = bb_length
        self.bb_std     = bb_std
        self.rsi_length = rsi_length
        self.rsi_entry  = rsi_entry
        self.rsi_exit   = rsi_exit

    def generate_signals(
        self,
        ohlcv: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None,
        pair_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        close = self._close(ohlcv)

        # Bollinger Bands
        bb = ta.bbands(close, length=self.bb_length, std=self.bb_std)
        lower_col  = [c for c in bb.columns if c.startswith("BBL")][0]
        middle_col = [c for c in bb.columns if c.startswith("BBM")][0]
        lower_band  = bb[lower_col]
        middle_band = bb[middle_col]

        # RSI
        rsi = ta.rsi(close, length=self.rsi_length)

        # Entry / exit conditions (vectorised masks)
        entry_cond = (close < lower_band) & (rsi < self.rsi_entry)
        exit_cond  = (close >= middle_band) | (rsi > self.rsi_exit)

        # Stateful forward-fill: once in trade, hold until exit
        signals = np.zeros(len(close), dtype=int)
        in_trade = False
        for i in range(len(close)):
            if not in_trade and bool(entry_cond.iloc[i]):
                in_trade = True
            elif in_trade and bool(exit_cond.iloc[i]):
                in_trade = False
            signals[i] = 1 if in_trade else 0

        return pd.Series(signals, index=close.index, name="signal").fillna(0)
