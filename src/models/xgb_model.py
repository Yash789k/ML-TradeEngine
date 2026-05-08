"""
Phase 03 — XGBoost Classifier with Optuna Hyperparameter Tuning

Wraps XGBClassifier in a sklearn-like interface with an integrated Optuna
study that searches the hyperparameter space on a hold-out validation split.

Training flow per fold
----------------------
  1. Optuna runs `n_trials` quick XGB fits on (X_train_sub, X_val_sub)
     to find the best hyperparameters.
  2. The best params are used to retrain on the FULL (X_train, y_train).
  3. Final model is returned and probability predictions made on X_test.

Predict contract
----------------
  predict_proba(X) → ndarray of shape (n, 3)  — [p_DOWN, p_FLAT, p_UP]
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def _xgb_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 50, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
        "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


# ---------------------------------------------------------------------------
# XGBModel
# ---------------------------------------------------------------------------

class XGBModel:
    """
    XGBoost 3-class directional classifier.

    Parameters
    ----------
    n_trials     : Optuna trials for hyperparameter search (default 50)
    val_fraction : fraction of X_train used for Optuna objective (default 0.2)
    random_state : reproducibility seed
    """

    def __init__(
        self,
        n_trials:     int   = 50,
        val_fraction: float = 0.2,
        random_state: int   = 42,
    ) -> None:
        self.n_trials     = n_trials
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.best_params_: dict[str, Any] = {}
        self.model_: xgb.XGBClassifier | None = None
        self.scaler_ = RobustScaler()

    # ------------------------------------------------------------------
    # Optuna objective
    # ------------------------------------------------------------------

    def _objective(
        self,
        trial: optuna.Trial,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weights: np.ndarray,
    ) -> float:
        params = _xgb_search_space(trial)
        model  = xgb.XGBClassifier(
            **params,
            objective    = "multi:softprob",
            num_class    = 3,
            eval_metric  = "mlogloss",
            random_state = self.random_state,
            verbosity    = 0,
            n_jobs       = 1,
        )
        model.fit(
            X_tr, y_tr,
            sample_weight = class_weights[y_tr],
            eval_set      = [(X_val, y_val)],
            verbose       = False,
        )
        preds = model.predict(X_val)
        return f1_score(y_val, preds, average="macro", zero_division=0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        class_weight_map: dict[int, float] | None = None,
    ) -> "XGBModel":
        """
        Tune then retrain on full training data.

        Parameters
        ----------
        X_train          : feature matrix (n_rows, n_features)
        y_train          : labels {0, 1, 2}
        class_weight_map : {class_int: weight} from labels.compute_class_weights()
        """
        X_train = self.scaler_.fit_transform(X_train)

        # Build per-sample weight array
        cw = class_weight_map or {0: 1.0, 1: 1.0, 2: 1.0}
        w_arr = np.array([cw.get(int(c), 1.0) for c in y_train])

        # Inner validation split for Optuna
        n_val  = max(10, int(len(X_train) * self.val_fraction))
        X_tr, X_val = X_train[:-n_val], X_train[-n_val:]
        y_tr, y_val = y_train[:-n_val], y_train[-n_val:]
        w_tr        = w_arr[:-n_val]

        # Build a weight array compatible with the objective signature
        w_vec = np.array([cw.get(int(c), 1.0) for c in range(3)])

        study = optuna.create_study(
            direction    = "maximize",
            sampler      = optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(
            lambda t: self._objective(t, X_tr, y_tr, X_val, y_val, w_vec),
            n_trials     = self.n_trials,
            show_progress_bar = False,
        )
        self.best_params_ = study.best_params
        log.info(
            "XGB best params (F1=%.4f): %s",
            study.best_value, self.best_params_,
        )

        # Retrain on full training set with best params
        self.model_ = xgb.XGBClassifier(
            **self.best_params_,
            objective    = "multi:softprob",
            num_class    = 3,
            eval_metric  = "mlogloss",
            random_state = self.random_state,
            verbosity    = 0,
            n_jobs       = 1,
        )
        self.model_.fit(X_train, y_train, sample_weight=w_arr)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability matrix of shape (n, 3)."""
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the XGBoost model to a JSON file (scaler saved separately)."""
        import json
        from pathlib import Path
        import joblib

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.model_.save_model(str(p))
        joblib.dump(self.scaler_, str(p).replace(".json", "_scaler.pkl"))

    @classmethod
    def load(cls, path: str) -> "XGBModel":
        import joblib
        obj = cls()
        obj.model_ = xgb.XGBClassifier()
        obj.model_.load_model(path)
        obj.scaler_ = joblib.load(path.replace(".json", "_scaler.pkl"))
        return obj
