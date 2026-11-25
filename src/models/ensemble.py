"""
Phase 03 — Ensemble Classifier (Soft-Vote / Mixture-of-Experts)

Combines probability outputs from XGBoost, LightGBM, and LSTM.

Two modes (select via ensemble_mode):

  static : P_ensemble[t, c] = Σ_m  w_m        * P_m[t, c]
           Fixed weight vector for every bar (original behaviour).

  moe    : P_ensemble[t, c] = Σ_m  w_m(x_t)   * P_m[t, c]
           Per-bar weights from a fitted GatingModel (src/models/gating.py)
           conditioned on HMM regime, volatility, drawdown state, and
           rolling per-expert accuracy.

Default weights are equal (1/3 each).  Weights can be overridden to reflect
per-model out-of-sample performance (e.g., weight by validation F1 score).

All input arrays must have shape (n_rows, 3) and probabilities in [0, 1].
The ensemble also exposes a signal column — the argmax class — and an
optional confidence score (max probability).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from src.models.gating import GatingModel


_CLASS_NAMES = {0: "DOWN", 1: "FLAT", 2: "UP"}


class EnsembleClassifier:
    """
    Ensemble over three model outputs.

    Parameters
    ----------
    weights       : optional (3,) weight vector [w_xgb, w_lgbm, w_lstm].
                    Defaults to equal weights.  L1-normalized internally.
                    Used in 'static' mode and as the 'moe' fallback when no
                    gate/context is supplied.
    ensemble_mode : 'static' (fixed soft-vote, default) or 'moe'
                    (context-dependent weights from a GatingModel).
    gate          : fitted GatingModel; required for 'moe' mode.
    """

    def __init__(
        self,
        weights: Optional[list[float]] = None,
        ensemble_mode: Literal["static", "moe"] = "static",
        gate: Optional["GatingModel"] = None,
    ) -> None:
        if ensemble_mode not in ("static", "moe"):
            raise ValueError(f"ensemble_mode must be 'static' or 'moe', got {ensemble_mode!r}")
        if ensemble_mode == "moe" and gate is None:
            raise ValueError("ensemble_mode='moe' requires a fitted GatingModel via gate=")

        self.ensemble_mode = ensemble_mode
        self.gate          = gate

        if weights is not None:
            w = np.array(weights, dtype=float)
            self.weights = w / w.sum()
        else:
            self.weights = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

    # ------------------------------------------------------------------
    # Core aggregation
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        xgb_proba:  np.ndarray,
        lgbm_proba: np.ndarray,
        lstm_proba: np.ndarray,
        context: Optional["pd.DataFrame"] = None,
    ) -> np.ndarray:
        """
        Weighted average of the three probability matrices.

        Parameters
        ----------
        xgb_proba, lgbm_proba, lstm_proba : ndarray of shape (n, 3)
        context : gate context DataFrame (build_gate_context output),
                  aligned row-for-row with the probability matrices.
                  Only used in 'moe' mode; when omitted the ensemble
                  falls back to its static weights.

        Returns
        -------
        ensemble_proba : ndarray of shape (n, 3), rows sum to 1.0
        """
        self._validate(xgb_proba,  "xgb_proba")
        self._validate(lgbm_proba, "lgbm_proba")
        self._validate(lstm_proba, "lstm_proba")

        if self.ensemble_mode == "moe" and context is not None:
            if len(context) != len(xgb_proba):
                raise ValueError(
                    f"context has {len(context)} rows but probabilities have "
                    f"{len(xgb_proba)} — they must align row-for-row"
                )
            W = self.gate.predict_weights(context)          # (n, 3)
            proba = (
                W[:, [0]] * xgb_proba
                + W[:, [1]] * lgbm_proba
                + W[:, [2]] * lstm_proba
            )
        else:
            proba = (
                self.weights[0] * xgb_proba
                + self.weights[1] * lgbm_proba
                + self.weights[2] * lstm_proba
            )
        # Re-normalise to handle any floating-point drift
        row_sums = proba.sum(axis=1, keepdims=True)
        return proba / np.where(row_sums == 0, 1.0, row_sums)

    def predict(
        self,
        xgb_proba:  np.ndarray,
        lgbm_proba: np.ndarray,
        lstm_proba: np.ndarray,
        context: Optional["pd.DataFrame"] = None,
    ) -> np.ndarray:
        """Return argmax class labels (0=DOWN, 1=FLAT, 2=UP)."""
        return np.argmax(
            self.predict_proba(xgb_proba, lgbm_proba, lstm_proba, context=context),
            axis=1,
        )

    def signal_frame(
        self,
        xgb_proba:  np.ndarray,
        lgbm_proba: np.ndarray,
        lstm_proba: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Return a dict with keys:
          'proba'       : (n, 3) ensemble probability matrix
          'signal'      : (n,) integer class {0,1,2}
          'signal_name' : (n,) string {DOWN, FLAT, UP}
          'confidence'  : (n,) max probability ∈ [0,1]
        """
        proba      = self.predict_proba(xgb_proba, lgbm_proba, lstm_proba)
        signal     = np.argmax(proba, axis=1)
        confidence = proba.max(axis=1)
        names      = np.array([_CLASS_NAMES[s] for s in signal])
        return {
            "proba":       proba,
            "signal":      signal,
            "signal_name": names,
            "confidence":  confidence,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(arr: np.ndarray, name: str) -> None:
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{name} must have shape (n, 3), got {arr.shape}")
        if not np.allclose(arr.sum(axis=1), 1.0, atol=1e-4):
            raise ValueError(f"{name} rows do not sum to 1.0 (got min={arr.sum(axis=1).min():.4f})")

    @classmethod
    def from_val_f1(
        cls,
        f1_xgb: float,
        f1_lgbm: float,
        f1_lstm: float,
    ) -> "EnsembleClassifier":
        """Factory: weight each model proportionally to its validation F1 score."""
        return cls(weights=[f1_xgb, f1_lgbm, f1_lstm])
