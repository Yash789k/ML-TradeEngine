"""Exploratory single-instrument risk simulation.

Uses a fixed ex-ante payoff assumption, causal policy inputs, close-observed
stops, and marked cash/holdings. It is a separate risk ablation, not the shared
next-close portfolio execution contract. Classifier UP probabilities have not
been calibrated to stop-managed trade outcomes; no Kelly optimality is claimed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.risk.portfolio import (
    circuit_breaker_reset,
    is_circuit_broken,
)
from src.risk.sizing import kelly_series
from src.risk.stops import StopState, initial_stop, is_stop_hit, update_trailing_stop
from src.risk.var import risk_summary, rolling_var_cvar

log = logging.getLogger(__name__)

_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
_BACKTEST_DIR  = _PROJECT_ROOT / "data" / "backtest"
_RISK_DIR      = _PROJECT_ROOT / "data" / "risk"


@dataclass
class RiskResult:
    equity_curve:  pd.Series
    daily_returns: pd.Series
    trade_log:     pd.DataFrame
    var_series:    pd.DataFrame
    risk_report:   dict = field(default_factory=dict)
    params:        dict = field(default_factory=dict)
    ledger:        pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Win/loss ratio estimation
# ---------------------------------------------------------------------------

def _estimate_win_loss_ratio(ticker: str, fallback: float = 1.5) -> float:
    """
    Load avg_win / avg_loss from the Phase 04 backtest summary JSON.
    Falls back to `fallback` if the file is missing or values are invalid.
    """
    path = _BACKTEST_DIR / ticker / "backtest_summary.json"
    if not path.exists():
        log.warning("No Phase 04 summary for %s — using win/loss ratio %.2f", ticker, fallback)
        return fallback

    with open(path) as f:
        data = json.load(f)

    m        = data.get("strategy_metrics", {})
    avg_win  = m.get("avg_win",  0.0)
    avg_loss = abs(m.get("avg_loss", 0.0))

    if avg_win <= 0 or avg_loss <= 0:
        return fallback

    ratio = avg_win / avg_loss
    log.info("  Win/loss ratio for %s: avg_win=$%.0f  avg_loss=$%.0f  ratio=%.2f",
             ticker, avg_win, avg_loss, ratio)
    return ratio


# ---------------------------------------------------------------------------
# RiskEngine
# ---------------------------------------------------------------------------

class RiskEngine:
    """
    Risk-managed single-ticker portfolio simulator.

    Parameters
    ----------
    initial_capital   : starting equity in USD
    commission        : per-side commission fraction (default 10 bps)
    slippage          : per-side slippage fraction   (default  5 bps)
    position_frac     : fraction of cash deployed per trade (default 0.95)
    kelly_multiplier  : fraction of full Kelly (default 0.5 = half-Kelly)
    min_kelly         : minimum half-Kelly threshold to enter a trade
                        (filter: skip if half-Kelly ≤ min_kelly)
    atr_multiplier    : ATR multiple for trailing stop-loss (default 2.0)
    cb_threshold      : circuit-breaker drawdown level (default 0.85 → 15% DD)
    cb_resume         : drawdown level at which circuit breaker resets (default 0.90)
    var_window        : rolling lookback for VaR computation in days (default 252)
    """

    def __init__(
        self,
        initial_capital:  float = 10_000.0,
        commission:       float = 0.001,
        slippage:         float = 0.0005,
        position_frac:    float = 0.95,
        kelly_multiplier: float = 0.5,
        min_kelly:        float = 0.0,
        atr_multiplier:   float = 2.0,
        trailing_stop:    bool  = False,   # True = ratchet up; False = fixed hard stop
        cb_threshold:     float = 0.85,
        cb_resume:        float = 0.90,
        cb_cooldown_bars: int   = 63,    # also reset after N inactive bars (~3 months)
        var_window:       int   = 252,
        win_loss_ratio:   float = 1.5,
    ):
        self.initial_capital  = initial_capital
        self.commission       = commission
        self.slippage         = slippage
        self.position_frac    = position_frac
        self.kelly_multiplier = kelly_multiplier
        self.min_kelly        = min_kelly
        self.atr_multiplier   = atr_multiplier
        self.trailing_stop    = trailing_stop
        self.cb_threshold     = cb_threshold
        self.cb_resume        = cb_resume
        self.cb_cooldown_bars = cb_cooldown_bars
        self.var_window       = var_window
        self.win_loss_ratio = win_loss_ratio

    # ------------------------------------------------------------------

    def run(
        self,
        ticker:    str,
        signal_df: pd.DataFrame,
        feat_df:   pd.DataFrame,
        policy_engine=None,
    ) -> RiskResult:
        """
        Execute the risk-managed simulation for a single ticker.

        Parameters
        ----------
        ticker        : asset symbol (used for Phase 04 win/loss calibration)
        signal_df     : from signals.build_signal_df() — columns: p_up,
                        filtered_signal, close
        feat_df       : Phase 02 feature matrix — must include 'atr_14'
        policy_engine : optional src.risk.policy_engine.PolicyEngine — when
                        given, the ATR stop multiplier widens with volatility,
                        position size is capped in stress, and new entries are
                        blocked on ineligible bars (dynamic policy layer).
        """
        if "atr_14" not in feat_df.columns:
            raise KeyError("feat_df missing 'atr_14'. Re-run Phase 02 pipeline.")

        atr_aligned = feat_df["atr_14"].reindex(signal_df.index).ffill()
        atr_arr     = atr_aligned.values.astype(np.float64)
        prices      = signal_df["close"].values.astype(np.float64)
        p_up_arr    = signal_df["p_up"].values.astype(np.float64)
        sig_arr     = signal_df["filtered_signal"].values.astype(int)
        n           = len(prices)

        # ── Dynamic policy (per-bar) or static constants ────────────────
        if policy_engine is not None:
            pol = policy_engine.compute_policy(feat_df).reindex(signal_df.index).ffill()
            atr_mult_arr = pol["atr_mult"].fillna(self.atr_multiplier).values
            size_cap_arr = pol["size_cap"].fillna(1.0).values
            eligible_arr = pol["eligible"].fillna(True).astype(bool).values
        else:
            atr_mult_arr = np.full(n, self.atr_multiplier)
            size_cap_arr = np.ones(n)
            eligible_arr = np.ones(n, dtype=bool)

        # ── Kelly filter calibration ───────────────────────────────────
        win_loss_ratio = self.win_loss_ratio  # fixed ex-ante assumption, not estimated from future trades
        half_kellys    = kelly_series(p_up_arr, win_loss_ratio, self.kelly_multiplier)

        # ── Simulation state ───────────────────────────────────────────
        cost_per_side  = self.commission + self.slippage
        equity         = np.empty(n)
        equity[0]      = self.initial_capital
        peak_equity    = self.initial_capital

        cash           = float(self.initial_capital)
        shares         = 0.0
        current_pos    = 0            # 0 = cash, 1 = long
        entry_price    = 0.0
        entry_cost     = 0.0          # gross dollars paid on entry
        stop_state        = StopState()
        circuit_broken    = False
        cb_bars_inactive  = 0          # bars since circuit breaker tripped
        trades: list[dict] = []
        history = [dict(cash=cash, shares=0., price=prices[0], equity=cash)]

        for i in range(1, n):
            price   = prices[i]
            atr_val = float(atr_arr[i - 1])

            # ── Update peak and circuit breaker ────────────────────────
            peak_equity = max(peak_equity, cash + shares * price)

            if circuit_broken:
                cb_bars_inactive += 1
                equity_recovered = circuit_breaker_reset(
                    cash + shares * price, peak_equity, self.cb_resume
                )
                time_elapsed = cb_bars_inactive >= self.cb_cooldown_bars
                if equity_recovered or time_elapsed:
                    circuit_broken   = False
                    cb_bars_inactive = 0
                    log.debug(
                        "  [%s] Circuit breaker RESET at bar %d (equity_ok=%s, time=%s)",
                        ticker, i, equity_recovered, time_elapsed,
                    )
            else:
                if is_circuit_broken(cash + shares * price, peak_equity, self.cb_threshold):
                    circuit_broken   = True
                    cb_bars_inactive = 0
                    log.debug("  [%s] Circuit breaker TRIPPED at bar %d", ticker, i)

            # ── ATR stop check (trailing or hard fixed) ───────────────
            # Trailing stop: ratchets up daily — good for trend following but
            # exits medium-term signals too early (clips winning trades).
            # Hard stop (default): fixed at entry_price - 2*ATR, never moves —
            # caps catastrophic losses while letting winners reach signal exit.
            stop_triggered = False
            if current_pos == 1:
                if self.trailing_stop:
                    update_trailing_stop(
                        stop_state, price, atr_val,
                        atr_multiplier=float(atr_mult_arr[i - 1]),
                    )
                if is_stop_hit(stop_state, price):
                    gross  = shares * price
                    net    = gross * (1.0 - cost_per_side)
                    pnl    = net - entry_cost
                    cash  += net
                    shares = 0.0
                    current_pos   = 0
                    stopped_at = stop_state.stop_level
                    stop_state    = StopState()
                    stop_triggered = True
                    trades.append({
                        "type": "exit", "reason": "stop",
                        "date": signal_df.index[i], "price": price,
                        "pnl": pnl, "entry_price": entry_price,
                        "stop_level": stopped_at,
                    })

            # ── Signal processing ──────────────────────────────────────
            if not stop_triggered:
                sig    = int(sig_arr[i - 1])
                target = 1 if sig == 2 else 0     # long-only

                # Kelly filter: skip trade if expected edge is too small
                hk = float(half_kellys[i - 1])
                if target == 1 and hk <= self.min_kelly:
                    target = 0

                # Block new entries during circuit breaker
                if circuit_broken and current_pos == 0:
                    target = 0

                # Dynamic policy: block new entries on ineligible bars
                # (extreme volatility / unknown regime); existing positions
                # are managed by stops, not force-closed.
                if not eligible_arr[i - 1] and current_pos == 0:
                    target = 0

                # ── Execute position change ────────────────────────────
                if target != current_pos:

                    # Close existing position
                    if current_pos == 1:
                        gross  = shares * price
                        net    = gross * (1.0 - cost_per_side)
                        pnl    = net - entry_cost
                        cash  += net
                        shares = 0.0
                        current_pos = 0
                        stop_state  = StopState()
                        trades.append({
                            "type": "exit", "reason": "signal",
                            "date": signal_df.index[i], "price": price,
                            "pnl": pnl, "entry_price": entry_price,
                            "stop_level": np.nan,
                        })

                    # Open new long position
                    if target == 1 and cash > 0 and np.isfinite(atr_val) and atr_val > 0:
                        deploy_gross = cash * self.position_frac * float(size_cap_arr[i - 1])
                        deploy_net   = deploy_gross / (1.0 + cost_per_side)
                        shares       = deploy_net / price
                        entry_cost   = deploy_gross
                        entry_price  = price
                        cash        -= deploy_gross
                        current_pos  = 1
                        stop_state   = StopState(
                            active      = True,
                            stop_level  = initial_stop(
                                price, atr_val, direction=1,
                                atr_multiplier=float(atr_mult_arr[i - 1]),
                            ),
                            direction   = 1,
                            entry_price = price,
                            entry_atr   = atr_val,
                        )
                        trades.append({
                            "type": "entry", "reason": "signal",
                            "date": signal_df.index[i], "price": price,
                            "pnl": np.nan, "entry_price": entry_price,
                            "stop_level": stop_state.stop_level,
                        })

            # ── Mark-to-market  ────────────────────────────────────────
            equity[i] = cash + shares * price
            history.append(dict(cash=cash, shares=shares, price=price, equity=equity[i]))

        # ── Post-simulation analytics ──────────────────────────────────
        equity_s  = pd.Series(equity, index=signal_df.index, name="equity")
        returns_s = equity_s.pct_change().fillna(0.0).rename("returns")
        trade_df  = pd.DataFrame(trades, columns=["type", "reason", "date", "price", "pnl", "entry_price", "stop_level"])

        var_df   = rolling_var_cvar(returns_s, window=self.var_window)
        risk_rep = risk_summary(returns_s, initial_capital=self.initial_capital)

        n_kelly_filtered = int((half_kellys <= self.min_kelly).sum())
        log.info(
            "  [%s] Kelly filter removed %d/%d signal bars (%.0f%%)",
            ticker, n_kelly_filtered, n,
            n_kelly_filtered / n * 100,
        )

        params = {
            "initial_capital":  self.initial_capital,
            "commission":       self.commission,
            "slippage":         self.slippage,
            "position_frac":    self.position_frac,
            "kelly_multiplier": self.kelly_multiplier,
            "min_kelly":        self.min_kelly,
            "atr_multiplier":   self.atr_multiplier,
            "trailing_stop":    self.trailing_stop,
            "cb_threshold":     self.cb_threshold,
            "cb_cooldown_bars": self.cb_cooldown_bars,
            "win_loss_ratio":   win_loss_ratio,
            "calibration": "fixed_ex_ante_assumption_not_trade_probability_calibration",
            "status": "exploratory_close_stop_simulation",
            "n_kelly_filtered": n_kelly_filtered,
            "dynamic_policy":   policy_engine is not None,
        }

        return RiskResult(
            equity_curve  = equity_s,
            daily_returns = returns_s,
            trade_log     = trade_df,
            var_series    = var_df,
            risk_report   = risk_rep,
            params        = params,
            ledger        = pd.DataFrame(history, index=signal_df.index),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(
        self,
        ticker:          str,
        result:          RiskResult,
        phase04_metrics: dict,
    ) -> Path:
        """Persist risk results to data/risk/{ticker}/."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from src.backtest.metrics import compute_metrics

        out_dir = _RISK_DIR / ticker
        out_dir.mkdir(parents=True, exist_ok=True)

        def _write(df: pd.DataFrame, name: str):
            pq.write_table(
                pa.Table.from_pandas(df, preserve_index=True),
                str(out_dir / f"{name}.parquet"),
                compression="snappy",
            )

        _write(result.equity_curve.to_frame(), "equity_curve")
        _write(result.var_series, "var_cvar")
        _write(result.trade_log, "trade_log")

        risk_metrics = compute_metrics(
            result.daily_returns,
            result.trade_log if len(result.trade_log) > 0 else None,
        )

        summary = {
            "ticker":          ticker,
            "risk_metrics":    risk_metrics,
            "var_report":      result.risk_report,
            "phase04_metrics": phase04_metrics,
            "improvement": {
                k: round(risk_metrics.get(k, 0) - phase04_metrics.get(k, 0), 4)
                for k in ["sharpe_ratio", "cagr", "max_drawdown", "calmar_ratio"]
            },
            "params": result.params,
        }

        (out_dir / "risk_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        log.info("  Risk results saved → %s", out_dir)
        return out_dir
