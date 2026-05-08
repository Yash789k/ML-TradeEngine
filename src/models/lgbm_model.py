"""
Phase 03 — LightGBM Classifier with Optuna Hyperparameter Tuning

Same contract as XGBModel — soft-probability output, Optuna search,
RobustScaler normalisation, save/load.

LightGBM brings ensemble diversity versus XGBoost through its histogram-based
leaf-wise tree growth and different regularisation defaults.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def _lgbm_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 50, 500),
        "max_depth":        trial.suggest_int("max_depth", 3, 9),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 16, 128),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_samples":trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


# ---------------------------------------------------------------------------
# LGBMModel
# ---------------------------------------------------------------------------

class LGBMModel:
    """
    LightGBM 3-class directional classifier.

    Parameters
    ----------
    n_trials     : Optuna trials for hyperparameter search (default 50)
    val_fraction : fraction of X_train used for Optuna objective
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
        self.model_: lgb.LGBMClassifier | None = None
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
        class_weights: dict[int, float],
    ) -> float:
        params = _lgbm_search_space(trial)
        w_arr  = np.array([class_weights.get(int(c), 1.0) for c in y_tr])
        model  = lgb.LGBMClassifier(
            **params,
            objective    = "multiclass",
            num_class    = 3,
            random_state = self.random_state,
            verbosity    = -1,
            n_jobs       = -1,
        )
        model.fit(
            X_tr, y_tr,
            sample_weight  = w_arr,
            eval_set       = [(X_val, y_val)],
            callbacks      = [lgb.early_stopping(20, verbose=False),
                              lgb.log_evaluation(0)],
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
    ) -> "LGBMModel":
        X_train = self.scaler_.fit_transform(X_train)

        cw    = class_weight_map or {0: 1.0, 1: 1.0, 2: 1.0}
        w_arr = np.array([cw.get(int(c), 1.0) for c in y_train])

        n_val  = max(10, int(len(X_train) * self.val_fraction))
        X_tr, X_val = X_train[:-n_val], X_train[-n_val:]
        y_tr, y_val = y_train[:-n_val], y_train[-n_val:]

        study = optuna.create_study(
            direction = "maximize",
            sampler   = optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(
            lambda t: self._objective(t, X_tr, y_tr, X_val, y_val, cw),
            n_trials          = self.n_trials,
            show_progress_bar = False,
        )
        self.best_params_ = study.best_params
        log.info(
            "LGBM best params (F1=%.4f): %s",
            study.best_value, self.best_params_,
        )

        self.model_ = lgb.LGBMClassifier(
            **self.best_params_,
            objective    = "multiclass",
            num_class    = 3,
            random_state = self.random_state,
            verbosity    = -1,
            n_jobs       = -1,
        )
        self.model_.fit(X_train, y_train, sample_weight=w_arr)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler_.transform(X)
        # When loaded from disk the raw Booster is used directly (avoids sklearn
        # fitted-state checks that aren't restored by the load path).
        if hasattr(self, "_raw_booster"):
            return self._raw_booster.predict(X_scaled)
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        return self.model_.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        import joblib
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.model_.booster_.save_model(str(p))
        joblib.dump(self.scaler_, str(p).replace(".txt", "_scaler.pkl"))

    @classmethod
    def load(cls, path: str) -> "LGBMModel":
        import joblib
        obj = cls()
        obj._raw_booster = lgb.Booster(model_file=path)
        obj.scaler_ = joblib.load(path.replace(".txt", "_scaler.pkl"))
        return obj
