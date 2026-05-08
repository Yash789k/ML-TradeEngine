"""
Phase 06A — EMA Crossover Strategy

Signal Logic
------------
  Entry : Fast EMA crosses above Slow EMA  AND  ADX > adx_threshold
          (confirmed uptrend with sufficient directional strength)
  Exit  : Fast EMA crosses below Slow EMA  OR   ADX drops below adx_min

  Default: EMA(20) / EMA(50) with ADX(14) > 20 filter.

The ADX filter prevents false entries in low-volatility, choppy markets.
Position is long-only; cross below immediately exits.

Regime Hypothesis
-----------------
  Works best in low-noise trending regimes; ADX filter cuts ranging-market whipsaws.

Academic Reference
------------------
  Murphy (1999) "Technical Analysis of the Financial Markets".
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import pandas_ta as ta

from src.research.strategies.base import BaseStrategy


class EMACrossoverStrategy(BaseStrategy):
    """
    Dual-EMA crossover with ADX trend-strength filter.

    Parameters
    ----------
    fast        : fast EMA span in days (default 20)
    slow        : slow EMA span in days (default 50)
    adx_length  : ADX period (default 14)
    adx_min     : minimum ADX for a valid signal (default 20)
    """

    name = "EMA_Crossover"

    def __init__(
        self,
        fast: int = 20,
        slow: int = 50,
        adx_length: int = 14,
        adx_min: float = 20.0,
    ) -> None:
        self.fast       = fast
        self.slow       = slow
        self.adx_length = adx_length
        self.adx_min    = adx_min

    def generate_signals(
        self,
        ohlcv: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None,
        pair_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        close = self._close(ohlcv)
        high  = ohlcv["High"].ffill()
        low   = ohlcv["Low"].ffill()

        fast_ema = close.ewm(span=self.fast, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow, adjust=False).mean()

        adx_df  = ta.adx(high, low, close, length=self.adx_length)
        adx_col = [c for c in adx_df.columns if c.startswith("ADX")][0]
        adx     = adx_df[adx_col].fillna(0)

        # Long when fast > slow AND ADX confirms trend
        raw_signal = ((fast_ema > slow_ema) & (adx >= self.adx_min)).astype(int)
        return raw_signal.fillna(0)
