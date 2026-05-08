"""
Phase 06A — Alpha Trends Strategy (Novel — Paper Thesis Strategy)

This is the original strategy at the core of the research paper:

  "Regime-Gated Alpha Trends: A Unified Framework for Strategy
   Selection Under Non-Stationary Market States"

Signal Logic
------------
  Three independent filters must ALL be active simultaneously:

  1. HMM Regime Gate : HMM-detected market state == bull (state 2)
                       Prevents entering during bear or ranging regimes.

  2. Trend Filter    : Close > EMA(200)
                       Ensures we are trading in the direction of the
                       long-term structural trend.

  3. Momentum Gate   : 3-month (63-bar) return > momentum_threshold
                       Confirms near-term momentum is positive before entry.

  Entry : all three conditions active
  Exit  : any one condition becomes false

Why This Is Novel
-----------------
  Most trend-following strategies use only price-based filters (1+2).
  The HMM regime gate (1) is explicitly non-stationary — it captures the
  latent market state from return/volatility dynamics — and dynamically
  suppresses false entries in ranging or bear states.

  Combining regime awareness with a classical price trend + momentum filter
  reduces drawdown (bear state suppression) while preserving upside
  (bull state capture), producing a higher Calmar ratio than any individual
  filter in isolation.

  This combination is the paper's primary contribution.

Regime Hypothesis
-----------------
  Best performance during sustained bull regimes with positive momentum.
  Deliberately quiet during bear and ranging regimes — this is intentional
  and is the mechanism that reduces drawdown versus pure momentum.

Parameters
----------
  ema_period        : long-term trend EMA span (default 200)
  momentum_period   : momentum lookback in bars (default 63 ≈ 3 months)
  momentum_threshold: minimum 3-month return to qualify (default 0.0)
  n_states          : number of HMM states (default 3: bear/ranging/bull)
  random_state      : HMM random seed for reproducibility (default 42)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.features.regime import add_hmm_regime
from src.research.strategies.base import BaseStrategy


class AlphaTrendsStrategy(BaseStrategy):
    """
    Regime-gated trend + momentum strategy — the paper's core contribution.

    Parameters
    ----------
    ema_period          : EMA span for long-term trend filter (default 200)
    momentum_period     : bars for momentum lookback (default 63)
    momentum_threshold  : minimum return for momentum gate (default 0.0)
    n_states            : HMM state count (default 3)
    random_state        : HMM seed (default 42)
    min_hold            : minimum bars to hold once entered (default 10)
                          prevents HMM-churn whipsawing on high-trend tickers
                          where the regime gate stays open and EMA/momentum
                          flip rapidly on short-term noise
    """

    name = "Alpha_Trends"

    def __init__(
        self,
        ema_period: int = 200,
        momentum_period: int = 63,
        momentum_threshold: float = 0.0,
        n_states: int = 3,
        random_state: int = 42,
        min_hold: int = 10,
    ) -> None:
        self.ema_period          = ema_period
        self.momentum_period     = momentum_period
        self.momentum_threshold  = momentum_threshold
        self.n_states            = n_states
        self.random_state        = random_state
        self.min_hold            = min_hold

    def generate_signals(
        self,
        ohlcv: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None,
        pair_df: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        close = self._close(ohlcv)

        # ── Filter 1: HMM Regime Gate ─────────────────────────────────────
        log_return    = np.log(close / close.shift(1))
        realized_vol  = log_return.rolling(21).std()

        hmm_input = pd.DataFrame({
            "log_return":      log_return,
            "realized_vol_21": realized_vol,
        }).dropna()

        if len(hmm_input) < max(50, self.n_states * 10):
            # Not enough data to fit HMM — return all-flat
            return pd.Series(0, index=close.index, name="signal")

        try:
            hmm_input_with_regime, _ = add_hmm_regime(
                hmm_input.copy(),
                n_states=self.n_states,
                random_state=self.random_state,
            )
            regime = hmm_input_with_regime["hmm_regime"].reindex(close.index).ffill()
        except Exception:
            regime = pd.Series(-1, index=close.index)

        bull_regime = (regime == self.n_states - 1)  # highest-mean state = bull

        # ── Filter 2: Long-term Trend ─────────────────────────────────────
        ema_long   = close.ewm(span=self.ema_period, adjust=False).mean()
        above_ema  = close > ema_long

        # ── Filter 3: Near-term Momentum Gate ────────────────────────────
        momentum   = close.pct_change(self.momentum_period)
        strong_mom = momentum > self.momentum_threshold

        # ── Composite signal ─────────────────────────────────────────────
        raw = (bull_regime & above_ema & strong_mom).astype(int).fillna(0).values

        # ── Minimum hold (anti-whipsaw) ───────────────────────────────────
        # Once entered, hold for at least `min_hold` bars regardless of whether
        # the composite condition temporarily drops.  Prevents HMM-churn on
        # persistently bull-trending tickers (NVDA, GS, BAC) where the regime
        # gate is almost always open and EMA/momentum flip on short-term noise.
        smoothed     = np.zeros(len(raw), dtype=int)
        hold_counter = 0
        in_trade     = False

        for i in range(len(raw)):
            if not in_trade:
                if raw[i] == 1:
                    in_trade     = True
                    hold_counter = 1
                    smoothed[i]  = 1
            else:
                smoothed[i]  = 1
                hold_counter += 1
                if hold_counter >= self.min_hold and raw[i] == 0:
                    in_trade     = False
                    hold_counter = 0

        return pd.Series(smoothed, index=close.index, name="signal")
