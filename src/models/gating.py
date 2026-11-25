"""
Phase 03+ — Mixture-of-Experts Gating Model

Replaces the static soft-vote weights of the EnsembleClassifier with
context-dependent weights learned from market state.

  static  :  P_ens = Σ_m  w_m        · P_m      (w fixed for all bars)
  moe     :  P_ens = Σ_m  w_m(x_t)   · P_m      (w a function of context)

Gate design
-----------
One binary logistic regression per expert, each predicting
P(expert m's argmax equals the true label | context).  The per-expert
correctness probabilities are floored and L1-normalised into a weight
vector.  This is deliberately simple: with ~1000 training rows per
ticker a softmax network overfits, while per-expert logistic curves are
stable and interpretable (deck slide 38: "the gate can overfit recent
noise, so it needs walk-forward testing, regularization, and minimum
evidence rules").

Context features (deck slide 38 "Gating input")
-----------------------------------------------
  regime_bear / regime_ranging / regime_bull : one-hot HMM state
  vol_21           : realised 21-day volatility
  vol_pctile       : causal expanding percentile of vol_21
  drawdown_state   : close / trailing 63-day peak − 1
  acc_<expert>     : rolling 63-bar accuracy per expert (shifted 1 bar)

All context columns must be computable at decision time — the builder
functions below only use information up to and including bar t−1 where
the value describes model performance, and bar t for market state.

No-leakage training protocol
----------------------------
The gate is trained strictly walk-forward on out-of-sample expert
predictions: for evaluation block k, fit on all OOS rows before the
block start (see scripts/compare_moe.py).  Inside ModelTrainer the same
rule applies per CV fold — the gate for fold k sees only folds < k.

Usage
-----
    from src.models.gating import GatingModel, build_gate_context

    ctx  = build_gate_context(feat_df, model_probs, y_true)
    gate = GatingModel(min_train=100)
    gate.fit(ctx.iloc[:500], correct[:500])
    W    = gate.predict_weights(ctx.iloc[500:])     # (n, n_experts)
"""

from __future__ import annotations

import logging
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

_EXPERT_NAMES = ["xgb", "lgbm", "lstm"]

_MIN_WEIGHT   = 0.05     # floor: no expert is ever fully silenced
_ACC_WINDOW   = 63       # rolling accuracy window (≈ one quarter)
_DD_WINDOW    = 63       # trailing-peak window for drawdown state


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------

def build_market_context(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the market-state half of the gate context from a Phase 02
    feature matrix.  All columns are causal.

    Returns a DataFrame indexed like feat_df with columns:
      regime_bear, regime_ranging, regime_bull, vol_21, vol_pctile,
      drawdown_state
    """
    ctx = pd.DataFrame(index=feat_df.index)

    regime = feat_df.get("hmm_regime", pd.Series(-1, index=feat_df.index))
    ctx["regime_bear"]    = (regime == 0).astype(float)
    ctx["regime_ranging"] = (regime == 1).astype(float)
    ctx["regime_bull"]    = (regime == 2).astype(float)

    vol = feat_df.get("realized_vol_21", pd.Series(np.nan, index=feat_df.index))
    ctx["vol_21"] = vol.ffill().fillna(0.0)

    # Causal expanding midrank percentile of volatility (ties count half,
    # so flat series rank 0.5 rather than 1.0)
    from src.risk.policy_engine import _midrank_pctile
    ctx["vol_pctile"] = (
        ctx["vol_21"].expanding(min_periods=20)
        .apply(_midrank_pctile, raw=True)
        .fillna(0.5)
    )

    close = feat_df["Close"]
    peak  = close.rolling(_DD_WINDOW, min_periods=1).max()
    ctx["drawdown_state"] = (close / peak - 1.0).fillna(0.0)

    return ctx


def rolling_expert_accuracy(
    model_probs: list[np.ndarray],
    y_true: np.ndarray,
    window: int = _ACC_WINDOW,
    horizon: int = 5,
) -> np.ndarray:
    """
    Rolling per-expert accuracy, shifted past the full label horizon so the value at row t
    only reflects outcomes known strictly before t.

    Parameters
    ----------
    model_probs : list of (n, 3) probability matrices, one per expert
    y_true      : (n,) integer labels

    Returns
    -------
    (n, n_experts) rolling accuracy matrix (default 1/3 during warm-up)
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    n   = len(y_true)
    acc = np.full((n, len(model_probs)), 1.0 / 3)
    for m, probs in enumerate(model_probs):
        correct = (np.argmax(probs, axis=1) == y_true).astype(float)
        s = (
            pd.Series(correct)
            .rolling(window, min_periods=10).mean()
            .shift(horizon + 1)                         # full label maturity plus one bar
            .fillna(1.0 / 3)
        )
        acc[:, m] = s.values
    return acc


def build_gate_context(
    feat_df: pd.DataFrame,
    model_probs: list[np.ndarray],
    y_true: np.ndarray,
    index: Optional[pd.Index] = None,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Full gate context = market state (from features) + rolling expert
    accuracy (from past OOS outcomes).

    Parameters
    ----------
    feat_df     : Phase 02 feature matrix (superset of the target rows)
    model_probs : list of (n, 3) per-expert probability matrices
    y_true      : (n,) true labels aligned with model_probs rows
    index       : the DatetimeIndex of the prediction rows; defaults to
                  the last n rows of feat_df
    """
    n = len(y_true)
    if index is None:
        index = feat_df.index[-n:]

    market = build_market_context(feat_df).reindex(index).ffill().fillna(0.0)
    acc    = rolling_expert_accuracy(model_probs, y_true, horizon=horizon)

    ctx = market.copy()
    for m, name in enumerate(_EXPERT_NAMES[: len(model_probs)]):
        ctx[f"acc_{name}"] = acc[:, m]
    return ctx


# ---------------------------------------------------------------------------
# GatingModel
# ---------------------------------------------------------------------------

class GatingModel:
    """
    Per-expert logistic-regression gate.

    Parameters
    ----------
    n_experts  : number of experts to weight (default 3: XGB, LGBM, LSTM)
    l2         : inverse regularisation strength passed as C=1/l2
    min_train  : minimum training rows; below this fit() is a no-op and
                 predict_weights() returns uniform weights (minimum
                 evidence rule)
    min_weight : per-expert weight floor applied before normalisation
    """

    def __init__(
        self,
        n_experts:  int   = 3,
        l2:         float = 1.0,
        min_train:  int   = 100,
        min_weight: float = _MIN_WEIGHT,
    ) -> None:
        self.n_experts  = n_experts
        self.l2         = l2
        self.min_train  = min_train
        self.min_weight = min_weight

        self._scaler: Optional[StandardScaler]      = None
        self._models: list[Optional[LogisticRegression]] = [None] * n_experts
        self._base_rates = np.full(n_experts, 1.0 / 3)
        self.fitted = False

    # ------------------------------------------------------------------

    def fit(self, context: pd.DataFrame, correct: np.ndarray) -> "GatingModel":
        """
        Fit one logistic regression per expert.

        Parameters
        ----------
        context : (n, k) gate context DataFrame (build_gate_context output)
        correct : (n, n_experts) boolean/0-1 matrix — expert argmax == truth

        Returns self.  If n < min_train the gate stays unfitted and will
        emit uniform weights.
        """
        X = np.asarray(context, dtype=float)
        C = np.asarray(correct, dtype=float)
        if C.ndim != 2 or C.shape[1] != self.n_experts:
            raise ValueError(f"correct must have shape (n, {self.n_experts})")
        if len(X) != len(C):
            raise ValueError("context and correct must have equal length")

        if len(X) < self.min_train:
            log.info("[Gate] %d rows < min_train=%d — staying unfitted (uniform).",
                     len(X), self.min_train)
            self.fitted = False
            return self

        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)

        for m in range(self.n_experts):
            y = C[:, m]
            self._base_rates[m] = float(y.mean()) if len(y) else 1.0 / 3
            if len(np.unique(y)) < 2:
                # Degenerate target (expert always/never right) — keep base rate
                self._models[m] = None
                continue
            clf = LogisticRegression(C=1.0 / self.l2, max_iter=1000)
            clf.fit(Xs, y)
            self._models[m] = clf

        self.fitted = True
        return self

    # ------------------------------------------------------------------

    def predict_weights(self, context: pd.DataFrame) -> np.ndarray:
        """
        Context-dependent expert weights.

        Returns (n, n_experts) matrix; each row is non-negative, floored
        at min_weight, and sums to 1.  Unfitted gate → uniform weights.
        """
        n = len(context)
        if not self.fitted or self._scaler is None:
            return np.full((n, self.n_experts), 1.0 / self.n_experts)

        Xs = self._scaler.transform(np.asarray(context, dtype=float))
        raw = np.empty((n, self.n_experts))
        for m in range(self.n_experts):
            clf = self._models[m]
            if clf is None:
                raw[:, m] = self._base_rates[m]
            else:
                raw[:, m] = clf.predict_proba(Xs)[:, 1]

        raw = np.maximum(raw, self.min_weight)
        return raw / raw.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "n_experts":  self.n_experts,
                "l2":         self.l2,
                "min_train":  self.min_train,
                "min_weight": self.min_weight,
                "scaler":     self._scaler,
                "models":     self._models,
                "base_rates": self._base_rates,
                "fitted":     self.fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "GatingModel":
        blob = joblib.load(path)
        gate = cls(
            n_experts  = blob["n_experts"],
            l2         = blob["l2"],
            min_train  = blob["min_train"],
            min_weight = blob["min_weight"],
        )
        gate._scaler     = blob["scaler"]
        gate._models     = blob["models"]
        gate._base_rates = blob["base_rates"]
        gate.fitted      = blob["fitted"]
        return gate
