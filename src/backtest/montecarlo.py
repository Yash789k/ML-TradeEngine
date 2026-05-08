"""
Phase 04 — Monte Carlo Bootstrap Simulation

Generates 1000 bootstrapped equity paths by resampling the strategy's daily
returns with replacement.  This answers: "given the observed return distribution,
what range of outcomes could we expect over N years?"

Key outputs
-----------
  - 5th / 50th / 95th percentile equity curves
  - Confidence intervals for Sharpe, max drawdown, CAGR
  - Probability of ruin (equity < 50 % of initial capital)

Bootstrap design
----------------
  Block bootstrap (block_size=5 days) is used instead of i.i.d. sampling to
  preserve short-term autocorrelation in daily returns.  This produces more
  realistic confidence intervals than naive i.i.d. resampling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.metrics import cagr, max_drawdown, sharpe_ratio


# ---------------------------------------------------------------------------
# Core bootstrap engine
# ---------------------------------------------------------------------------

def _block_bootstrap(
    returns: np.ndarray,
    n_paths: int,
    n_days: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate `n_paths` bootstrapped daily-return sequences of length `n_days`
    by sampling non-overlapping blocks of size `block_size`.

    Returns an array of shape (n_paths, n_days).
    """
    n_blocks = int(np.ceil(n_days / block_size))
    n_orig   = len(returns)
    paths    = np.empty((n_paths, n_blocks * block_size), dtype=np.float64)

    for p in range(n_paths):
        starts = rng.integers(0, n_orig - block_size + 1, size=n_blocks)
        blocks = [returns[s : s + block_size] for s in starts]
        paths[p] = np.concatenate(blocks)

    return paths[:, :n_days]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_monte_carlo(
    returns: pd.Series,
    n_paths: int   = 1000,
    n_years: float = 3.0,
    block_size: int = 5,
    initial_capital: float = 10_000.0,
    risk_free: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Run Monte Carlo bootstrap simulation on daily returns.

    Parameters
    ----------
    returns         : daily return Series from the strategy
    n_paths         : number of simulated equity paths (default 1000)
    n_years         : forward simulation horizon in years
    block_size      : block size for block bootstrap (default 5 = 1 trading week)
    initial_capital : starting equity for each simulated path
    risk_free       : annual risk-free rate used for Sharpe calculation
    seed            : RNG seed for reproducibility

    Returns
    -------
    dict with keys:
      paths_df         : DataFrame (n_days × n_paths) of equity curves
      percentiles_df   : 5 / 25 / 50 / 75 / 95 percentile equity curves
      sharpe_ci        : (p5, median, p95) for Sharpe ratio
      cagr_ci          : (p5, median, p95) for CAGR
      max_dd_ci        : (p5, median, p95) for max drawdown
      prob_ruin        : fraction of paths ending below 50% of initial capital
      summary          : dict of all CI values for logging
    """
    rng     = np.random.default_rng(seed)
    clean   = returns.dropna().replace([np.inf, -np.inf], 0).values
    n_days  = int(n_years * 252)

    raw_paths  = _block_bootstrap(clean, n_paths, n_days, block_size, rng)
    # Cumulative equity
    equity_paths = initial_capital * np.cumprod(1 + raw_paths, axis=1)

    # ── Per-path metrics ──────────────────────────────────────────────
    sharpes  = np.array([sharpe_ratio(pd.Series(raw_paths[i]), risk_free)
                         for i in range(n_paths)])
    cagrs    = np.array([cagr(pd.Series(raw_paths[i]))           for i in range(n_paths)])
    max_dds  = np.array([max_drawdown(pd.Series(raw_paths[i]))   for i in range(n_paths)])

    def _ci(arr: np.ndarray) -> tuple[float, float, float]:
        return (float(np.percentile(arr, 5)),
                float(np.percentile(arr, 50)),
                float(np.percentile(arr, 95)))

    sharpe_ci = _ci(sharpes)
    cagr_ci   = _ci(cagrs)
    max_dd_ci = _ci(max_dds)

    # ── Equity-curve percentiles ───────────────────────────────────────
    pct_df = pd.DataFrame(
        {
            "p5":    np.percentile(equity_paths, 5,  axis=0),
            "p25":   np.percentile(equity_paths, 25, axis=0),
            "p50":   np.percentile(equity_paths, 50, axis=0),
            "p75":   np.percentile(equity_paths, 75, axis=0),
            "p95":   np.percentile(equity_paths, 95, axis=0),
        }
    )

    prob_ruin = float((equity_paths[:, -1] < initial_capital * 0.5).mean())

    summary = {
        "n_paths":        n_paths,
        "n_years":        n_years,
        "sharpe_p5":      round(sharpe_ci[0], 3),
        "sharpe_median":  round(sharpe_ci[1], 3),
        "sharpe_p95":     round(sharpe_ci[2], 3),
        "cagr_p5":        round(cagr_ci[0], 4),
        "cagr_median":    round(cagr_ci[1], 4),
        "cagr_p95":       round(cagr_ci[2], 4),
        "max_dd_p5":      round(max_dd_ci[0], 4),
        "max_dd_median":  round(max_dd_ci[1], 4),
        "max_dd_p95":     round(max_dd_ci[2], 4),
        "prob_ruin":      round(prob_ruin, 4),
    }

    return {
        "equity_paths":  equity_paths,
        "percentiles_df": pct_df,
        "sharpe_ci":     sharpe_ci,
        "cagr_ci":       cagr_ci,
        "max_dd_ci":     max_dd_ci,
        "prob_ruin":     prob_ruin,
        "summary":       summary,
    }
