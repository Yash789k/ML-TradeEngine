"""
Phase 04 — Performance Metrics

Pure-numpy / pandas implementations of standard quantitative finance metrics.
All metrics are annualised assuming 252 trading days per year.

Functions are intentionally stateless (no class state) so they can be
called from any context: backtesting, Monte Carlo, live monitoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_TRADING_DAYS = 252
_SQRT_252     = np.sqrt(_TRADING_DAYS)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series, risk_free: float = 0.05) -> float:
    """Annualised Sharpe ratio.  risk_free is annual; converted to daily."""
    daily_rf = (1 + risk_free) ** (1 / _TRADING_DAYS) - 1
    excess   = returns - daily_rf
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return 0.0
    return float(excess.mean() / returns.std(ddof=1) * _SQRT_252)


def sortino_ratio(returns: pd.Series, risk_free: float = 0.05) -> float:
    """Annualised Sortino ratio (downside deviation denominator)."""
    daily_rf  = (1 + risk_free) ** (1 / _TRADING_DAYS) - 1
    excess    = returns - daily_rf
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    if not np.isfinite(downside) or downside == 0:
        return 0.0
    return float(excess.mean() / downside * _SQRT_252)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a non-positive fraction, including initial NAV."""
    equity = (1 + returns).cumprod()
    peak   = equity.cummax().clip(lower=1.0)
    dd     = (equity - peak) / peak
    return float(dd.min()) if len(dd) else 0.0   # negative value — the caller negates if needed


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate."""
    n_years = len(returns) / _TRADING_DAYS
    if n_years == 0:
        return 0.0
    total = (1 + returns).prod()
    return float(total ** (1 / n_years) - 1)


def calmar_ratio(returns: pd.Series) -> float:
    """CAGR / |max_drawdown|."""
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return float(cagr(returns) / mdd)


def win_rate(trade_log: pd.DataFrame) -> float:
    """Fraction of CLOSED trades with positive P&L."""
    exits = trade_log[trade_log.get("type", pd.Series(index=trade_log.index, dtype=str)) == "exit"]
    if len(exits) == 0:
        return 0.0
    return float((exits["pnl"] > 0).mean())


def avg_trade_pnl(trade_log: pd.DataFrame) -> dict[str, float]:
    """Average winning and losing P&L per trade."""
    exits = trade_log[trade_log.get("type", pd.Series(index=trade_log.index, dtype=str)) == "exit"]
    if len(exits) == 0:
        return {"avg_win": 0.0, "avg_loss": 0.0, "expectancy": 0.0}
    wins  = exits[exits["pnl"] > 0]["pnl"]
    losses= exits[exits["pnl"] <= 0]["pnl"]
    avg_w = float(wins.mean()) if len(wins) else 0.0
    avg_l = float(losses.mean()) if len(losses) else 0.0
    wr    = len(wins) / len(exits)
    exp   = wr * avg_w + (1 - wr) * avg_l
    return {"avg_win": avg_w, "avg_loss": avg_l, "expectancy": exp}


# ---------------------------------------------------------------------------
# Composite metrics dict
# ---------------------------------------------------------------------------

def compute_metrics(
    returns: pd.Series,
    trade_log: pd.DataFrame | None = None,
    risk_free: float = 0.05,
) -> dict[str, float]:
    """
    Compute the full suite of performance metrics in one call.

    Parameters
    ----------
    returns   : daily return Series
    trade_log : output of simulator.run_simulation().trade_log (optional)
    risk_free : annual risk-free rate (default 5 %)

    Returns
    -------
    dict with keys:
      total_return, cagr, sharpe_ratio, sortino_ratio, max_drawdown,
      calmar_ratio, volatility, win_rate (if trade_log supplied)
    """
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("Returns contain missing or non-finite observations")
    clean = returns.astype(float)
    total_ret = float((1 + clean).prod() - 1)
    vol       = float(clean.std(ddof=1) * _SQRT_252) if len(clean) > 1 else 0.0

    result = {
        "total_return":  round(total_ret, 4),
        "cagr":          round(cagr(clean), 4),
        "sharpe_ratio":  round(sharpe_ratio(clean, risk_free), 3),
        "sortino_ratio": round(sortino_ratio(clean, risk_free), 3),
        "max_drawdown":  round(max_drawdown(clean), 4),
        "calmar_ratio":  round(calmar_ratio(clean), 3),
        "volatility":    round(vol, 4),
        "n_days":        len(clean),
    }

    if trade_log is not None:
        result["win_rate"] = round(win_rate(trade_log), 4)
        result["n_closed_trades"] = int((trade_log.get("type", pd.Series(dtype=str)) == "exit").sum())
        result["n_trades"] = int((trade_log.get("type", pd.Series(dtype=str)) == "entry").sum())
        pnl_stats = avg_trade_pnl(trade_log)
        result.update({k: round(v, 4) for k, v in pnl_stats.items()})

    return result


# ---------------------------------------------------------------------------
# Strategy vs benchmark comparison table
# ---------------------------------------------------------------------------

def compare_to_benchmark(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    strategy_name: str = "Strategy",
    benchmark_name: str = "Buy & Hold",
    trade_log: pd.DataFrame | None = None,
    risk_free: float = 0.05,
) -> pd.DataFrame:
    """
    Return a formatted comparison DataFrame of key metrics.
    """
    if not strategy_returns.index.equals(benchmark_returns.index) or benchmark_returns.isna().any():
        raise ValueError("Benchmark must cover exactly the strategy observation period")
    s_m = compute_metrics(strategy_returns,  trade_log, risk_free)
    b_m = compute_metrics(benchmark_returns, None,      risk_free)

    keys = ["total_return", "cagr", "sharpe_ratio", "sortino_ratio",
            "max_drawdown", "calmar_ratio", "volatility"]

    rows = {
        k: {strategy_name: s_m.get(k, np.nan), benchmark_name: b_m.get(k, np.nan)}
        for k in keys
    }
    df = pd.DataFrame(rows).T
    df.index.name = "metric"
    return df
