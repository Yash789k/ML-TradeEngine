"""
Phase 05 — Position Sizing

Implements two complementary sizing methods that are combined to produce
a final position-fraction recommendation for each signal bar.

Kelly Criterion (fractional Kelly)
-----------------------------------
The Kelly formula for a binary outcome bet:

    f* = (p × b - q) / b

where:
    p  = probability of winning  (ensemble p_up when signal is UP)
    q  = 1 - p                   (probability of losing)
    b  = win/loss ratio           (avg_win_$ / avg_loss_$ from Phase 04 stats)

Half-Kelly (f = f*/2) is used throughout — a well-known risk-reduction
convention that halves variance at the cost of ~25% lower long-run growth.

Negative Kelly values (edge < 0) → position_fraction = 0 (don't trade).

ATR Volatility Scaling
-----------------------
Position size is also scaled inversely by current ATR relative to its
rolling median.  A bar with ATR = 2× median is twice as volatile, so the
position is halved.  This prevents over-sizing into volatile regimes.

    atr_scale = atr_median / max(atr_current, 1e-9)
    atr_scale = clip(atr_scale, 0.25, 2.0)  # prevent extreme scaling

Combined fraction
-----------------
    raw_frac = kelly_frac × atr_scale
    position_frac = clip(raw_frac × 0.5, 0, max_position)

where max_position = 0.20 for single-ticker (or 0.05 for multi-ticker portfolio).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Kelly fraction computation
# ---------------------------------------------------------------------------

def kelly_fraction(
    p_win: float,
    win_loss_ratio: float,
    kelly_multiplier: float = 0.5,
) -> float:
    """
    Compute the (fractional) Kelly position size.

    Parameters
    ----------
    p_win           : probability of a winning trade (e.g. p_up from ensemble)
    win_loss_ratio  : avg_win / avg_loss (must be > 0)
    kelly_multiplier: fraction of full Kelly to use (default 0.5 = half-Kelly)

    Returns
    -------
    float in [0, 1] — fraction of bankroll to bet.
    Returns 0.0 if the edge is negative (Kelly formula < 0).
    """
    if win_loss_ratio <= 0 or p_win <= 0 or p_win >= 1:
        return 0.0

    p_loss  = 1.0 - p_win
    f_full  = (p_win * win_loss_ratio - p_loss) / win_loss_ratio
    f_frac  = f_full * kelly_multiplier

    return float(max(f_frac, 0.0))


def kelly_series(
    p_up: np.ndarray,
    win_loss_ratio: float,
    kelly_multiplier: float = 0.5,
) -> np.ndarray:
    """Vectorised Kelly fraction for an array of p_up values."""
    p_loss = 1.0 - p_up
    f_full = (p_up * win_loss_ratio - p_loss) / max(win_loss_ratio, 1e-9)
    return np.clip(f_full * kelly_multiplier, 0.0, None)


# ---------------------------------------------------------------------------
# ATR volatility scaling
# ---------------------------------------------------------------------------

def atr_scale_series(
    atr: np.ndarray,
    window: int = 63,   # ~3-month rolling median
    min_scale: float = 0.25,
    max_scale: float = 2.0,
) -> np.ndarray:
    """
    Return a per-bar scaling factor: high ATR → smaller position.

    scale[i] = rolling_median(ATR, window) / ATR[i]
    clipped to [min_scale, max_scale]
    """
    import pandas as pd
    atr_s   = pd.Series(atr, dtype=float)
    median  = atr_s.rolling(window, min_periods=5).median()
    scale   = (median / atr_s.replace(0, np.nan)).fillna(1.0)
    return np.clip(scale.values, min_scale, max_scale)


# ---------------------------------------------------------------------------
# Combined position fraction
# ---------------------------------------------------------------------------

def position_fractions(
    p_up: np.ndarray,
    atr: np.ndarray,
    win_loss_ratio: float,
    kelly_multiplier: float = 0.5,
    max_position: float = 0.20,
    atr_window: int = 63,
) -> np.ndarray:
    """
    Compute per-bar position fractions combining Kelly and ATR volatility scaling.

    Parameters
    ----------
    p_up            : ensemble probability of UP outcome (n,)
    atr             : daily ATR values aligned with p_up (n,)
    win_loss_ratio  : avg_win / avg_loss from Phase 04 trade log stats
    kelly_multiplier: 0.5 = half-Kelly
    max_position    : hard cap on fraction of capital per position
    atr_window      : rolling window for ATR median baseline

    Returns
    -------
    fractions : np.ndarray (n,) in [0, max_position]
    """
    k   = kelly_series(p_up, win_loss_ratio, kelly_multiplier)
    atr_s = atr_scale_series(atr, atr_window)
    raw = k * atr_s
    return np.clip(raw, 0.0, max_position)
