"""
Phase 05 — ATR Trailing Stop

Implements a 2× ATR trailing stop on long positions.

Mechanics
---------
  On entry at price P with ATR = A:
      initial_stop = P - 2 * A

  Each subsequent day:
      trailing_stop = max(trailing_stop, close - 2 * atr)

  Exit rule: exit the position if close < trailing_stop.

This is a trailing (ratcheting) stop — it only moves up, never down.
This locks in profits on winning trades while cutting losses on losers.

For short positions (long/short mode):
      initial_stop = P + 2 * A
      trailing_stop = min(trailing_stop, close + 2 * atr)
      Exit if close > trailing_stop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StopState:
    """Mutable stop state for one open position."""
    active:       bool  = False
    stop_level:   float = 0.0
    direction:    int   = 1      # 1 = long, -1 = short
    entry_price:  float = 0.0
    entry_atr:    float = 0.0


def initial_stop(entry_price: float, atr: float, direction: int = 1) -> float:
    """Compute the initial stop price on entry."""
    return entry_price - direction * 2.0 * atr


def update_trailing_stop(
    state: StopState,
    current_price: float,
    current_atr: float,
) -> StopState:
    """
    Ratchet the trailing stop up (for longs) or down (for shorts).
    Returns the updated StopState (mutated in place for efficiency).
    """
    if not state.active:
        return state

    new_candidate = current_price - state.direction * 2.0 * current_atr
    if state.direction == 1:   # long: stop can only rise
        state.stop_level = max(state.stop_level, new_candidate)
    else:                       # short: stop can only fall
        state.stop_level = min(state.stop_level, new_candidate)

    return state


def is_stop_hit(state: StopState, current_price: float) -> bool:
    """Returns True if the current price has violated the stop level."""
    if not state.active:
        return False
    if state.direction == 1:
        return current_price < state.stop_level
    else:
        return current_price > state.stop_level


# ---------------------------------------------------------------------------
# Vectorised stop-hit detection (used in testing / analytics)
# ---------------------------------------------------------------------------

def compute_trailing_stops(
    closes: np.ndarray,
    atrs: np.ndarray,
    entry_idx: int,
    atr_multiplier: float = 2.0,
    direction: int = 1,
) -> np.ndarray:
    """
    Compute the full trailing-stop series starting from entry_idx.

    Returns an array of stop levels (same length as closes), with
    NaN before entry_idx.
    """
    n      = len(closes)
    stops  = np.full(n, np.nan)

    if entry_idx >= n:
        return stops

    stops[entry_idx] = closes[entry_idx] - direction * atr_multiplier * atrs[entry_idx]

    for i in range(entry_idx + 1, n):
        candidate = closes[i] - direction * atr_multiplier * atrs[i]
        if direction == 1:
            stops[i] = max(stops[i - 1], candidate)
        else:
            stops[i] = min(stops[i - 1], candidate)

    return stops
