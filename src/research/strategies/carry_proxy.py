"""
Phase 06A — Carry Proxy Strategy (Yield Spread Signal)

Signal Logic
------------
  Use the 10Y-2Y yield spread as a proxy for the equity carry environment:
    - Positive spread (normal curve)  → favourable carry → long equities
    - Inverted spread (inverted curve) → negative carry   → cash

  Additional trend filter: price must be above its 200-day moving average
  to avoid catching falling knives in a deteriorating macro environment.

  Entry : yield_spread > 0  AND  Close > MA(200)
  Exit  : yield_spread <= 0 OR   Close < MA(200) * (1 - ma_buffer)

  ma_buffer gives a small hysteresis band to reduce whipsaw around the MA.

Regime Hypothesis
-----------------
  Positive yield spread historically correlates with economic expansion and
  risk-on regimes.  Strategy lags macro turning points but avoids deep
  recession drawdowns.

Academic Reference
------------------
  Fama & French (1989); Ilmanen (2011) "Expected Returns".
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.research.strategies.base import BaseStrategy


class CarryProxyStrategy(BaseStrategy):
    """
    Macro carry proxy using 10Y-2Y yield spread.

    Requires a `macro` DataFrame with column `yield_spread_10_2`.

    Parameters
    ----------
    ma_period   : long-term trend MA period (default 200)
    ma_buffer   : fractional buffer below MA for exit hysteresis (default 0.02)
    spread_col  : column name for yield spread in macro DataFrame
    """

    name = "Carry_Proxy"

    def __init__(
        self,
        ma_period: int = 200,
        ma_buffer: float = 0.02,
        spread_col: str = "yield_spread_10_2",
    ) -> None:
        self.ma_period  = ma_period
        self.ma_buffer  = ma_buffer
        self.spread_col = spread_col

    def generate_signals(
        self,
        ohlcv: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None,
        pair_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        close = self._close(ohlcv)
        ma200 = close.rolling(self.ma_period).mean()

        if macro is None or self.spread_col not in macro.columns:
            # Fall back to pure trend filter when macro is unavailable
            signal = (close > ma200).astype(int).fillna(0)
            return signal

        # Align macro to equity index
        spread = (
            macro[self.spread_col]
            .reindex(close.index)
            .ffill()
            .fillna(0)
        )

        entry_cond = (spread > 0) & (close > ma200)
        exit_cond  = (spread <= 0) | (close < ma200 * (1 - self.ma_buffer))

        # Stateful hold
        signals  = np.zeros(len(close), dtype=int)
        in_trade = False
        for i in range(len(close)):
            if not in_trade and bool(entry_cond.iloc[i]):
                in_trade = True
            elif in_trade and bool(exit_cond.iloc[i]):
                in_trade = False
            signals[i] = 1 if in_trade else 0

        return pd.Series(signals, index=close.index, name="signal").fillna(0)
