"""
Phase 05+ — Dynamic Policy Layer

Replaces the hard-coded operating constants (confidence threshold 0.38,
ATR stop multiplier 2.0, full position cap) with context-dependent values
computed at decision time from HMM regime state and realised volatility
(deck slide 39: "The policy layer changes thresholds, size caps, stop
distances, and eligibility rules while keeping the underlying strategy
intact").

Rules
-----
  Confidence threshold  TIGHTENS when the regime is unstable:
      base 0.38  →  +0.10 while the HMM state has changed within the last
      `stability_bars` bars (transitioning market = lower-quality signals).

  ATR stop multiplier   WIDENS when volatility is elevated:
      base 2.0   →  linear ramp to 3.0 as the causal volatility percentile
      moves from 0.80 to 1.00 (stops must breathe in stress, or every
      position is knocked out by noise).

  Position size cap     SHRINKS in stress:
      1.0 normally; 0.5 when vol percentile > 0.90 or the asset trades
      more than 10 % below its trailing 63-day peak.

  Trade eligibility     BLOCKS new entries in extremes:
      vol percentile > 0.95, or unknown regime (-1).

Causality
---------
Every input is computable at the close of bar t: the volatility percentile
is an expanding rank against history strictly before t, and regime
stability looks backward only.  The policy frame can therefore be used
bar-by-bar in backtests without look-ahead.

Usage
-----
    from src.risk.policy_engine import PolicyEngine

    pe     = PolicyEngine()
    frame  = pe.compute_policy(feat_df)      # per-bar DataFrame (backtest)
    latest = pe.decide_latest(feat_df)       # PolicyDecision (live engine)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Static defaults — what the system used before the policy layer existed.
BASE_CONFIDENCE  = 0.38
BASE_ATR_MULT    = 2.0
MAX_ATR_MULT     = 3.0
BASE_SIZE_CAP    = 1.0
STRESS_SIZE_CAP  = 0.5

_VOL_RAMP_START  = 0.80   # vol percentile where ATR widening begins
_VOL_STRESS      = 0.90   # size cap halves above this
_VOL_HALT        = 0.95   # no new entries above this
_UNSTABLE_BUMP   = 0.10   # confidence add-on while regime transitions
_DD_STRESS       = -0.10  # trailing-peak drawdown that halves size
_DD_WINDOW       = 63


def _midrank_pctile(w: np.ndarray) -> float:
    """
    Causal midrank percentile of the last value against strictly-prior
    history.  Ties count half so a constant series ranks at 0.5 instead
    of 1.0 (a plain <= rank would flag flat volatility as 'extreme').
    """
    hist, x = w[:-1], w[-1]
    if len(hist) == 0:
        return 0.5
    return float(((hist < x).sum() + 0.5 * (hist == x).sum()) / len(hist))


@dataclass
class PolicyDecision:
    """Operating parameters for one decision bar."""

    confidence_threshold: float
    atr_multiplier:       float
    size_cap:             float
    eligible:             bool
    regime_stable:        bool
    vol_pctile:           float

    def as_dict(self) -> dict:
        return {
            "confidence_threshold": round(self.confidence_threshold, 4),
            "atr_multiplier":       round(self.atr_multiplier, 4),
            "size_cap":             round(self.size_cap, 4),
            "eligible":             self.eligible,
            "regime_stable":        self.regime_stable,
            "vol_pctile":           round(self.vol_pctile, 4),
        }


class PolicyEngine:
    """
    Context-dependent operating policy.

    Parameters
    ----------
    base_confidence : confidence threshold in a stable regime (default 0.38)
    base_atr_mult   : ATR stop multiplier at normal volatility (default 2.0)
    max_atr_mult    : ATR stop multiplier at extreme volatility (default 3.0)
    stability_bars  : bars the HMM state must be unchanged to count as
                      stable (default 5)
    unstable_bump   : confidence add-on in unstable regimes (default 0.10;
                      set 0.0 to disable threshold tightening — the 22-ticker
                      ablation showed the bump fragments signals into extra
                      round trips, see docs/POLICY_RESULTS.md)
    """

    def __init__(
        self,
        base_confidence: float = BASE_CONFIDENCE,
        base_atr_mult:   float = BASE_ATR_MULT,
        max_atr_mult:    float = MAX_ATR_MULT,
        stability_bars:  int   = 5,
        unstable_bump:   float = _UNSTABLE_BUMP,
    ) -> None:
        self.base_confidence = base_confidence
        self.base_atr_mult   = base_atr_mult
        self.max_atr_mult    = max_atr_mult
        self.stability_bars  = stability_bars
        self.unstable_bump   = unstable_bump

    # ------------------------------------------------------------------
    # Vectorised per-bar policy (backtesting)
    # ------------------------------------------------------------------

    def compute_policy(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the full per-bar policy frame for a feature matrix.

        Returns a DataFrame indexed like feat_df with columns:
          conf_threshold, atr_mult, size_cap, eligible,
          regime_stable, vol_pctile
        """
        idx = feat_df.index
        n   = len(feat_df)

        # ── causal volatility percentile (midrank: ties count half, so a
        #    constant series sits at 0.5, not 1.0) ─────────────────────────
        vol = feat_df.get("realized_vol_21", pd.Series(np.nan, index=idx))
        vol = vol.ffill().fillna(0.0)
        vol_pct = (
            vol.expanding(min_periods=20)
            .apply(_midrank_pctile, raw=True)
            .fillna(0.5)
        )

        # ── regime stability: unchanged over the last `stability_bars` ──
        regime = feat_df.get("hmm_regime", pd.Series(-1, index=idx)).astype(int)
        stable = pd.Series(True, index=idx)
        for lag in range(1, self.stability_bars):
            stable &= regime.eq(regime.shift(lag))
        stable &= regime >= 0
        stable = stable.fillna(False)

        # ── drawdown state vs trailing peak ─────────────────────────────
        close = feat_df["Close"]
        peak  = close.rolling(_DD_WINDOW, min_periods=1).max()
        dd    = (close / peak - 1.0).fillna(0.0)

        # ── assemble rules ──────────────────────────────────────────────
        conf = np.full(n, self.base_confidence)
        conf[~stable.values] += self.unstable_bump

        ramp = np.clip((vol_pct.values - _VOL_RAMP_START) / (1.0 - _VOL_RAMP_START), 0, 1)
        atr_mult = self.base_atr_mult + ramp * (self.max_atr_mult - self.base_atr_mult)

        size_cap = np.full(n, BASE_SIZE_CAP)
        stress   = (vol_pct.values > _VOL_STRESS) | (dd.values < _DD_STRESS)
        size_cap[stress] = STRESS_SIZE_CAP

        eligible = (vol_pct.values <= _VOL_HALT) & (regime.values >= 0)

        return pd.DataFrame(
            {
                "conf_threshold": conf,
                "atr_mult":       atr_mult,
                "size_cap":       size_cap,
                "eligible":       eligible,
                "regime_stable":  stable.values,
                "vol_pctile":     vol_pct.values,
            },
            index=idx,
        )

    # ------------------------------------------------------------------
    # Latest-bar decision (live engine)
    # ------------------------------------------------------------------

    def decide_latest(self, feat_df: pd.DataFrame) -> PolicyDecision:
        """
        Policy for the most recent bar of feat_df — used by the live
        SignalGenerator before acting on today's signal.
        """
        row = self.compute_policy(feat_df).iloc[-1]
        decision = PolicyDecision(
            confidence_threshold = float(row["conf_threshold"]),
            atr_multiplier       = float(row["atr_mult"]),
            size_cap             = float(row["size_cap"]),
            eligible             = bool(row["eligible"]),
            regime_stable        = bool(row["regime_stable"]),
            vol_pctile           = float(row["vol_pctile"]),
        )
        log.info("[Policy] %s", decision.as_dict())
        return decision
