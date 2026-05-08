"""
Phase 05 — Portfolio-Level Risk Controls

Two portfolio-wide constraints are enforced before any new position is opened:

1. Portfolio Heat Limit
   --------------------
   "Heat" = total fraction of capital currently at risk across all open positions.
   No new position is opened if heat would exceed `max_heat` (default 20%).

   For a single-ticker strategy this means the active position fraction is
   always ≤ max_heat, regardless of the Kelly recommendation.

   For a multi-ticker portfolio, heat is the sum of individual position
   fractions.  A new signal is skipped if the portfolio is already at max heat.

2. Correlation Filter
   -------------------
   Adding a position in asset X is blocked if the Pearson correlation between
   X's returns and the existing portfolio's returns exceeds `max_corr` (default 0.7)
   over the trailing `corr_window` days.

   Rationale: correlated positions provide less diversification benefit and
   amplify drawdowns when a common factor reverses.

   For single-ticker strategies this check is a no-op (correlation with self
   is always 1.0, but the single-ticker engine bypasses the filter).

Circuit Breaker
---------------
The circuit breaker is tracked directly in RiskEngine.run() to keep it close
to the equity-curve state.  It is exposed here as a helper function.

    is_circuit_broken(equity, peak_equity, threshold=0.85) → bool

Once triggered, trading resumes only after equity recovers above
`resume_threshold` (default 90% of peak, i.e. half-recovery of the 15% loss).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Portfolio heat
# ---------------------------------------------------------------------------

class HeatTracker:
    """
    Tracks total portfolio heat (sum of active position fractions).

    Parameters
    ----------
    max_heat : float
        Maximum allowed total heat (default 0.20 = 20%).
    """

    def __init__(self, max_heat: float = 0.20):
        self.max_heat = max_heat
        self._positions: dict[str, float] = {}   # ticker → fraction

    @property
    def current_heat(self) -> float:
        return sum(self._positions.values())

    def can_open(self, ticker: str, fraction: float) -> bool:
        """Return True if opening `fraction` for `ticker` stays within heat limit."""
        existing = self._positions.get(ticker, 0.0)
        projected = self.current_heat - existing + fraction
        return projected <= self.max_heat + 1e-9

    def open(self, ticker: str, fraction: float) -> None:
        self._positions[ticker] = fraction

    def close(self, ticker: str) -> None:
        self._positions.pop(ticker, None)

    def reset(self) -> None:
        self._positions.clear()


# ---------------------------------------------------------------------------
# Correlation filter
# ---------------------------------------------------------------------------

def portfolio_correlation(
    new_returns: pd.Series,
    portfolio_returns: pd.Series,
    window: int = 63,
) -> float:
    """
    Compute the rolling Pearson correlation between a candidate asset's
    returns and the current portfolio returns over the trailing `window` bars.

    Returns float in [-1, 1].  Returns 0.0 if insufficient data.
    """
    aligned = pd.concat([new_returns, portfolio_returns], axis=1).dropna()
    if len(aligned) < 10:
        return 0.0
    tail = aligned.tail(window)
    corr = tail.iloc[:, 0].corr(tail.iloc[:, 1])
    return float(corr) if np.isfinite(corr) else 0.0


def correlation_allows_entry(
    new_returns: pd.Series,
    portfolio_returns: pd.Series,
    max_corr: float = 0.70,
    window: int = 63,
) -> bool:
    """
    Return True if the candidate position is sufficiently uncorrelated with
    the existing portfolio (i.e. |corr| < max_corr).

    For a single-ticker strategy this always returns True (called with
    portfolio_returns = None).
    """
    if portfolio_returns is None or len(portfolio_returns.dropna()) == 0:
        return True
    corr = portfolio_correlation(new_returns, portfolio_returns, window)
    return abs(corr) < max_corr


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

def is_circuit_broken(
    equity: float,
    peak_equity: float,
    threshold: float = 0.85,
) -> bool:
    """
    Return True if equity has dropped more than (1 - threshold) from peak.
    Default threshold=0.85 → triggers on 15% drawdown from peak.
    """
    if peak_equity <= 0:
        return False
    return equity / peak_equity < threshold


def circuit_breaker_reset(
    equity: float,
    peak_equity: float,
    resume_threshold: float = 0.90,
) -> bool:
    """
    Return True if a triggered circuit breaker should be released.
    Default resume_threshold=0.90 → resumes after recovering 50% of the drawdown.
    """
    if peak_equity <= 0:
        return True
    return equity / peak_equity >= resume_threshold
