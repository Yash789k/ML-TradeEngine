"""
Phase 03 — Ensemble Classifier (Soft-Vote)

Combines probability outputs from XGBoost, LightGBM, and LSTM via a
weighted soft-vote average.

  P_ensemble[t, c] = Σ_m  w_m * P_m[t, c]   for class c ∈ {0,1,2}

Default weights are equal (1/3 each).  Weights can be overridden to reflect
per-model out-of-sample performance (e.g., weight by validation F1 score).

All input arrays must have shape (n_rows, 3) and probabilities in [0, 1].
The ensemble also exposes a signal column — the argmax class — and an
optional confidence score (max probability).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


_CLASS_NAMES = {0: "DOWN", 1: "FLAT", 2: "UP"}


class EnsembleClassifier:
    """
    Soft-vote ensemble over three model outputs.

    Parameters
    ----------
    weights : optional (3,) weight vector [w_xgb, w_lgbm, w_lstm].
              Defaults to equal weights.  Weights are L1-normalized internally.
    """

    def __init__(
        self,
        weights: Optional[list[float]] = None,
    ) -> None:
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
    ) -> np.ndarray:
        """
        Weighted average of the three probability matrices.

        Parameters
        ----------
        xgb_proba, lgbm_proba, lstm_proba : ndarray of shape (n, 3)

        Returns
        -------
        ensemble_proba : ndarray of shape (n, 3), rows sum to 1.0
        """
        self._validate(xgb_proba,  "xgb_proba")
        self._validate(lgbm_proba, "lgbm_proba")
        self._validate(lstm_proba, "lstm_proba")

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
    ) -> np.ndarray:
        """Return argmax class labels (0=DOWN, 1=FLAT, 2=UP)."""
        return np.argmax(self.predict_proba(xgb_proba, lgbm_proba, lstm_proba), axis=1)

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
