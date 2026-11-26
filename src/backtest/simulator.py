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
    ledger:        pd.DataFrame = field(default_factory=pd.DataFrame)


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
    if mode not in ("long_only", "long_short"):
        raise ValueError("Unsupported position mode")
    if not np.isfinite([initial_capital, commission, slippage]).all() or initial_capital <= 0 or min(commission, slippage) < 0:
        raise ValueError("Capital must be positive and costs nonnegative")
    if signal_df.empty or not signal_df.index.is_monotonic_increasing or not signal_df.index.is_unique:
        raise ValueError("Signals require nonempty, unique, ordered observations")
    prices = signal_df["close"].to_numpy(dtype=float)
    signals = signal_df["filtered_signal"].to_numpy()
    if not np.isfinite(prices).all() or (prices <= 0).any() or not np.isin(signals, [0, 1, 2]).all():
        raise ValueError("Invalid prices or signals")
    cost_per_side = commission + slippage
    cash, shares, entry_price, entry_fee = float(initial_capital), 0.0, 0.0, 0.0
    entry_idx, current_pos, realized = 0, 0, 0.0
    trades, history = [], []
    for i, price in enumerate(prices):
        sig = int(signals[i - 1]) if i else 1
        target = 1 if sig == 2 else (-1 if mode == "long_short" and sig == 0 else 0)
        if target != current_pos:
            if current_pos:
                exit_fee = abs(shares) * price * cost_per_side
                pnl = shares * (price - entry_price) - entry_fee - exit_fee
                cash += shares * price - exit_fee
                realized += pnl
                trades.append(dict(type="exit", date=signal_df.index[i], price=price,
                    shares=shares, direction=current_pos, pnl=pnl, entry_price=entry_price,
                    hold_bars=i-entry_idx, fee=exit_fee, entry_fee=entry_fee))
                shares = 0.0
            if target:
                if cash <= 0:
                    raise ValueError("Account insolvent; margin liquidation required")
                # Signed cash/share ledger. Short proceeds remain in cash, with a
                # matching marked liability. Budget includes entry friction.
                shares = target * cash / (price * (1 + cost_per_side))
                entry_fee = abs(shares) * price * cost_per_side
                cash -= shares * price + entry_fee
                entry_price, entry_idx = price, i
                trades.append(dict(type="entry", date=signal_df.index[i], price=price,
                    shares=shares, direction=target, pnl=np.nan, entry_price=price,
                    hold_bars=0, fee=entry_fee, entry_fee=entry_fee))
            current_pos = target
        nav = cash + shares * price
        if nav < 0:
            raise ValueError("Account insolvent; leveraged short loss exceeds capital")
        unrealized = shares * (price - entry_price) - entry_fee if shares else 0.0
        history.append(dict(cash=cash, shares=shares, price=price, equity=nav,
                            realized_pnl=realized, unrealized_pnl=unrealized))
    ledger = pd.DataFrame(history, index=signal_df.index)
    equity = ledger["equity"]
    return SimResult(equity, equity.pct_change(fill_method=None).fillna(0).rename("returns"),
        pd.DataFrame(trades, columns=["type", "date", "price", "shares", "direction", "pnl",
                                     "entry_price", "hold_bars", "fee", "entry_fee"]),
        dict(initial_capital=initial_capital, commission=commission, slippage=slippage,
             mode=mode, accounting_version=2, execution="next_close",
             terminal_policy="mark_open_positions", cost_model="notional_cash_charge"), ledger)


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
