"""
Phase 04 — Vectorized Portfolio Simulator

A clean numpy/pandas daily-bar portfolio engine.  Replaces VectorBT, which is
incompatible with pandas ≥ 3.0 and numpy ≥ 2.x.

Design
------
  Signals are close-of-day decisions.  Execution is assumed at next bar's
  close (T+1 fill), consistent with overnight positioning.

  Entry/exit commission is charged as a fraction of trade notional.
  Slippage is modelled as an additional fixed fraction of notional.

  Long-only mode  : UP=long, FLAT/DOWN=cash
  Long/short mode : UP=long, DOWN=short, FLAT=cash

Position sizing
---------------
  Full-notional (100 % of equity) by default.
  Kelly / ATR sizing is added by the Risk Engine in Phase 05.

Output
------
  Returns a SimResult dataclass with:
    equity_curve   : pd.Series  daily portfolio NAV
    daily_returns  : pd.Series  daily percentage return
    trade_log      : pd.DataFrame  entry/exit events
    summary        : dict  high-level stats (computed by metrics module)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class SimResult:
    equity_curve:  pd.Series
    daily_returns: pd.Series
    trade_log:     pd.DataFrame
    params:        dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core simulation engine
# ---------------------------------------------------------------------------

def run_simulation(
    signal_df: pd.DataFrame,
    initial_capital: float = 10_000.0,
    commission: float = 0.001,      # 10 bps per side (entry + exit = 20 bps round-trip)
    slippage: float   = 0.0005,     # 5 bps additional fill cost
    mode: Literal["long_only", "long_short"] = "long_only",
) -> SimResult:
    """
    Simulate daily portfolio value given a signal DataFrame.

    Parameters
    ----------
    signal_df        : output of signals.build_signal_df() — must have columns
                       `close` and `filtered_signal`
    initial_capital  : starting equity in dollars
    commission       : fraction of notional charged per side (entry OR exit)
    slippage         : additional fill cost fraction per side
    mode             : 'long_only' (UP=long, rest=cash) or
                       'long_short' (UP=long, DOWN=short, FLAT=cash)

    Returns
    -------
    SimResult with equity_curve, daily_returns, trade_log
    """
    cost_per_side = commission + slippage

    prices   = signal_df["close"].values
    signals  = signal_df["filtered_signal"].values
    n        = len(prices)

    equity       = np.empty(n)
    equity[0]    = initial_capital
    positions    = np.empty(n, dtype=int)   # -1=short, 0=cash, 1=long
    positions[0] = 0

    trades: list[dict] = []
    current_pos  = 0
    cash         = initial_capital
    shares       = 0.0
    entry_price  = 0.0
    entry_idx    = 0

    for i in range(1, n):
        price = prices[i]

        # Map signal to target position
        sig = int(signals[i - 1])       # signal is set at close of i-1, filled at i
        if mode == "long_only":
            target = 1 if sig == 2 else 0
        else:
            target = 1 if sig == 2 else (-1 if sig == 0 else 0)

        # ── Execute position change ────────────────────────────────────
        if target != current_pos:
            # Close existing position
            if current_pos != 0:
                proceeds = abs(shares) * price * (1 - cost_per_side)
                pnl      = (proceeds - abs(shares) * entry_price) * np.sign(current_pos)
                trades.append({
                    "type":        "exit",
                    "date":        signal_df.index[i],
                    "price":       price,
                    "shares":      shares,
                    "direction":   current_pos,
                    "pnl":         pnl,
                    "entry_price": entry_price,
                    "hold_bars":   i - entry_idx,
                })
                cash = equity[i - 1] * (1 - cost_per_side)

            # Open new position
            if target != 0:
                cost_fill = cash * cost_per_side
                cash     -= cost_fill
                shares    = cash / price * target    # signed shares
                entry_price = price
                entry_idx   = i
                trades.append({
                    "type":        "entry",
                    "date":        signal_df.index[i],
                    "price":       price,
                    "shares":      shares,
                    "direction":   target,
                    "pnl":         np.nan,
                    "entry_price": entry_price,
                    "hold_bars":   0,
                })
            else:
                shares = 0.0

            current_pos = target

        # ── Mark-to-market ─────────────────────────────────────────────
        if current_pos == 0:
            equity[i] = cash
        elif current_pos == 1:
            equity[i] = abs(shares) * price
        else:  # short
            equity[i] = 2 * equity[i - 1] - abs(shares) * price

        positions[i] = current_pos

    equity_s  = pd.Series(equity,  index=signal_df.index, name="equity")
    returns_s = equity_s.pct_change().fillna(0.0).rename("returns")

    trade_df = pd.DataFrame(trades)

    return SimResult(
        equity_curve  = equity_s,
        daily_returns = returns_s,
        trade_log     = trade_df,
        params        = {
            "initial_capital": initial_capital,
            "commission":      commission,
            "slippage":        slippage,
            "mode":            mode,
        },
    )


# ---------------------------------------------------------------------------
# Buy-and-hold benchmark helper
# ---------------------------------------------------------------------------

def buy_and_hold(
    signal_df: pd.DataFrame,
    initial_capital: float = 10_000.0,
) -> pd.Series:
    """
    Simulate a simple buy-and-hold of the asset over the same period.
    Returns a daily equity Series for comparison.
    """
    prices = signal_df["close"].values
    ratio  = prices / prices[0]
    return pd.Series(
        initial_capital * ratio,
        index=signal_df.index,
        name="buy_hold",
    )


# ---------------------------------------------------------------------------
# Threshold / commission parameter sweep
# ---------------------------------------------------------------------------

def sweep_parameters(
    signal_df: pd.DataFrame,
    thresholds: list[float] = (0.35, 0.40, 0.45, 0.50),
    commissions: list[float] = (0.0005, 0.001, 0.002),
    initial_capital: float = 10_000.0,
    mode: str = "long_only",
) -> pd.DataFrame:
    """
    Sweep over confidence thresholds × commission rates and compute Sharpe ratio.
    Returns a DataFrame with columns: threshold, commission, sharpe, cagr, max_dd, n_trades.
    """
    from src.backtest.metrics import compute_metrics

    rows = []
    for thr in thresholds:
        # Re-apply confidence filter with this threshold
        swept_df = signal_df.copy()
        swept_df["filtered_signal"] = np.where(
            swept_df["confidence"] >= thr,
            swept_df["signal"],
            1,
        )
        for comm in commissions:
            result = run_simulation(swept_df, initial_capital, commission=comm, mode=mode)
            m      = compute_metrics(result.daily_returns)
            n_trd  = int(len(result.trade_log[result.trade_log["type"] == "entry"]))
            rows.append({
                "threshold":  thr,
                "commission": comm,
                "sharpe":     round(m["sharpe_ratio"], 3),
                "cagr":       round(m["cagr"], 3),
                "max_dd":     round(m["max_drawdown"], 3),
                "n_trades":   n_trd,
            })

    return pd.DataFrame(rows)
