"""
Phase 05 — Value at Risk (VaR) and Conditional Value at Risk (CVaR)

Historical Simulation Method (non-parametric)
----------------------------------------------
No distributional assumptions are made.  The empirical return distribution
from the trailing `window` days is used directly.

VaR(α) at confidence level α:
    The loss not exceeded with probability α.
    VaR_95 = -percentile(returns, 5)        (positive number = potential loss)

CVaR(α) / Expected Shortfall (ES):
    The expected loss *given* that VaR is breached.
    CVaR_99 = -mean(returns[returns < -VaR_99])

Both are reported as positive numbers (loss magnitudes).

Rolling daily computation
--------------------------
For a backtest context, VaR and CVaR are re-estimated each day using the
preceding `window` returns so they adapt to changing volatility regimes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Point-in-time estimates
# ---------------------------------------------------------------------------

def var_historical(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Historical VaR at `confidence` level.
    Returns a positive number (the potential loss magnitude).
    """
    if len(returns) == 0:
        return 0.0
    return float(-np.percentile(returns, (1 - confidence) * 100))


def cvar_historical(returns: np.ndarray, confidence: float = 0.99) -> float:
    """
    Historical CVaR (Expected Shortfall) at `confidence` level.
    Returns a positive number.
    """
    if len(returns) == 0:
        return 0.0
    var = var_historical(returns, confidence)
    tail = returns[returns < -var]
    if len(tail) == 0:
        return var
    return float(-tail.mean())


# ---------------------------------------------------------------------------
# Rolling series
# ---------------------------------------------------------------------------

def rolling_var_cvar(
    returns: pd.Series,
    window: int = 252,
    var_confidence: float  = 0.95,
    cvar_confidence: float = 0.99,
) -> pd.DataFrame:
    """
    Compute rolling daily VaR and CVaR over a trailing `window` of returns.

    Parameters
    ----------
    returns         : daily return Series
    window          : lookback window in trading days (default 252 = 1 year)
    var_confidence  : confidence level for VaR  (default 95%)
    cvar_confidence : confidence level for CVaR (default 99%)

    Returns
    -------
    DataFrame with columns: var_95, cvar_99
    Rows before window are filled with the first valid estimate.
    """
    clean = returns.replace([np.inf, -np.inf], 0).fillna(0)
    n = len(clean)

    var_arr  = np.empty(n)
    cvar_arr = np.empty(n)

    for i in range(n):
        start = max(0, i - window + 1)
        window_ret = clean.iloc[start : i + 1].values
        var_arr[i]  = max(var_historical(window_ret, var_confidence),  0.0)
        cvar_arr[i] = max(cvar_historical(window_ret, cvar_confidence), 0.0)

    return pd.DataFrame(
        {"var_95": var_arr, "cvar_99": cvar_arr},
        index=returns.index,
    )


# ---------------------------------------------------------------------------
# Portfolio-level risk summary
# ---------------------------------------------------------------------------

def risk_summary(
    returns: pd.Series,
    window: int = 252,
    initial_capital: float = 10_000.0,
) -> dict[str, float]:
    """
    Compute a concise risk summary for a return series.

    Returns a dict with:
      var_95_pct       : point-in-time VaR at 95% (using full history)
      cvar_99_pct      : point-in-time CVaR at 99%
      var_95_dollar    : VaR in dollar terms (vs initial_capital)
      cvar_99_dollar   : CVaR in dollar terms
      worst_day_pct    : single worst daily return
      best_day_pct     : single best daily return
      negative_days_pct: fraction of days with negative returns
    """
    clean = returns.dropna().replace([np.inf, -np.inf], 0)

    var_pct  = var_historical(clean.values,  0.95)
    cvar_pct = cvar_historical(clean.values, 0.99)

    return {
        "var_95_pct":        round(var_pct,  4),
        "cvar_99_pct":       round(cvar_pct, 4),
        "var_95_dollar":     round(var_pct  * initial_capital, 2),
        "cvar_99_dollar":    round(cvar_pct * initial_capital, 2),
        "worst_day_pct":     round(float(clean.min()), 4),
        "best_day_pct":      round(float(clean.max()), 4),
        "negative_days_pct": round(float((clean < 0).mean()), 4),
    }
