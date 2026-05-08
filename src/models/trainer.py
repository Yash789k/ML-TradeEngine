"""
Phase 03 — ModelTrainer
Orchestrates the full ML pipeline for one asset:

  1. Load Phase 02 feature matrix from Parquet cache
  2. Generate forward-return labels  (labels.py)
  3. Select top features via SHAP    (pipeline.py)
  4. Walk-forward CV                 (splitter.py)
     For each fold:
       a. Fit XGBModel   (Optuna-tuned)
       b. Fit LGBMModel  (Optuna-tuned)
       c. Fit LSTMModel  (early-stopped)
       d. Ensemble soft-vote, compute fold metrics
       e. Log fold to MLflow
  5. Retrain all models on the full dataset (final production models)
  6. Save models to data/models/{ticker}/
  7. Return a summary dict of all fold metrics

Usage
-----
    python3 train.py --ticker AAPL

Or programmatically:
    from src.models.trainer import ModelTrainer
    from src.data.loader import DataLoader
    from src.features.pipeline import FeatureEngineer

    loader   = DataLoader()
    fe       = FeatureEngineer()
    trainer  = ModelTrainer()

    aapl     = loader.load_equity("AAPL")
    spy      = loader.load_equity("SPY")
    btc      = loader.load_crypto("BTC/USDT")
    macro    = loader.load_macro()
    features = fe.compute_features(aapl, "AAPL", spy_df=spy, btc_df=btc, macro_df=macro)

    results  = trainer.run(
        features   = features,
        ticker     = "AAPL",
        n_trials   = 50,
        skip_lstm  = False,
    )
    print(results)
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.preprocessing import RobustScaler

from src.models.ensemble import EnsembleClassifier
from src.models.labels import CLASS_MASK, compute_class_weights, make_labels
from src.models.lgbm_model import LGBMModel
from src.models.lstm_model import LSTMModel
from src.models.splitter import WalkForwardSplit
from src.models.xgb_model import XGBModel

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODELS_ROOT  = _PROJECT_ROOT / "data" / "models"
_MODELS_ROOT.mkdir(parents=True, exist_ok=True)

# Raw OHLCV columns — never passed to the model
_RAW_COLS = frozenset({"Open", "High", "Low", "Close", "Adj_Close", "Volume", "ticker"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_down":  float(f1_score(y_true, y_pred, labels=[0], average="micro", zero_division=0)),
        "f1_flat":  float(f1_score(y_true, y_pred, labels=[1], average="micro", zero_division=0)),
        "f1_up":    float(f1_score(y_true, y_pred, labels=[2], average="micro", zero_division=0)),
    }


# ---------------------------------------------------------------------------
# ModelTrainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    Full training orchestrator for a single asset.

    Parameters
    ----------
    models_dir   : override default data/models/ location
    mlflow_uri   : MLflow tracking URI (default: local mlruns/)
    experiment   : MLflow experiment name
    """

    def __init__(
        self,
        models_dir:  Optional[Path] = None,
        mlflow_uri:  str = "mlruns",
        experiment:  str = "ml-trade-engine-phase03",
    ) -> None:
        self.models_dir = Path(models_dir) if models_dir else _MODELS_ROOT
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow_uri = mlflow_uri
        self.experiment = experiment

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        features: pd.DataFrame,
        horizon:   int   = 5,
        threshold: float = 0.005,
        top_n_features: int = 25,
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """
        From a Phase 02 feature matrix, generate labels and select features.

        Returns
        -------
        X          : feature DataFrame (clean rows only)
        y          : label Series aligned to X
        feat_cols  : ordered list of selected feature column names
        """
        labels, _ = make_labels(features, horizon=horizon, threshold=threshold)

        feat_cols = [c for c in features.columns if c not in _RAW_COLS]
        X = features[feat_cols].copy()
        y = labels.copy()

        # Drop rows with masked label or any NaN feature
        valid_mask = (y != CLASS_MASK) & X.notna().all(axis=1)
        X, y = X[valid_mask], y[valid_mask]

        log.info(
            "prepare_data: %d clean rows, %d features  "
            "(DOWN=%.1f%% FLAT=%.1f%% UP=%.1f%%)",
            len(X), len(feat_cols),
            (y == 0).mean() * 100,
            (y == 1).mean() * 100,
            (y == 2).mean() * 100,
        )

        # Fast SHAP-based feature selection on a held-out 20 % slice
        # (uses the same FeatureEngineer.select_features helper from Phase 02)
        if len(X) >= 200 and top_n_features < len(feat_cols):
            try:
                from src.features.pipeline import FeatureEngineer
                fe = FeatureEngineer()
                df_sel = X.copy()
                df_sel["label"] = y.values
                selected = fe.select_features(
                    df_sel, target_col="label", top_n=top_n_features
                )
                feat_cols = selected
                X = X[feat_cols]
                log.info("SHAP selected %d features.", len(feat_cols))
            except Exception as exc:
                log.warning("SHAP selection failed (%s); using all features.", exc)

        return X, y, feat_cols

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def run(
        self,
        features:        pd.DataFrame,
        ticker:          str   = "ASSET",
        horizon:         int   = 5,
        threshold:       float = 0.005,
        n_splits:        int   = 5,
        embargo_days:    int   = 10,
        n_trials_xgb:    int   = 50,
        n_trials_lgbm:   int   = 50,
        lstm_epochs:     int   = 50,
        lstm_seq_len:    int   = 60,
        top_n_features:  int   = 25,
        skip_lstm:       bool  = False,
        random_state:    int   = 42,
    ) -> dict:
        """
        Run the full walk-forward training loop and return a results summary.

        Parameters
        ----------
        features       : Phase 02 feature matrix for one asset
        ticker         : asset identifier (used for MLflow tags + file names)
        horizon        : label look-ahead bars
        threshold      : ± percentage threshold for UP/DOWN classification
        n_splits       : number of walk-forward folds
        embargo_days   : rows purged between train and test windows
        n_trials_xgb   : Optuna trials for XGBoost per fold
        n_trials_lgbm  : Optuna trials for LightGBM per fold
        lstm_epochs    : max LSTM training epochs per fold
        lstm_seq_len   : LSTM sequence window length
        top_n_features : max features after SHAP selection
        skip_lstm      : set True to skip LSTM (saves ~10× time)
        random_state   : global seed

        Returns
        -------
        results dict with keys:
          ticker, n_folds, fold_metrics[], oos_metrics, best_params_xgb,
          best_params_lgbm, feature_cols, model_paths
        """
        t0 = time.time()
        log.info("=== ModelTrainer.run(%s) start ===", ticker)

        X, y, feat_cols = self.prepare_data(
            features, horizon=horizon, threshold=threshold,
            top_n_features=top_n_features,
        )
        X_arr = X.values.astype(np.float32)
        y_arr = y.values.astype(np.int64)

        cw = compute_class_weights(y)

        splitter = WalkForwardSplit(
            n_splits     = n_splits,
            embargo_days = embargo_days,
        )

        fold_results: list[dict] = []
        # Accumulators for out-of-sample predictions (full timeline)
        oos_preds_xgb   = np.full((len(X_arr), 3), np.nan)
        oos_preds_lgbm  = np.full((len(X_arr), 3), np.nan)
        oos_preds_lstm  = np.full((len(X_arr), 3), 1.0 / 3.0)

        last_xgb  = None
        last_lgbm = None
        last_lstm = None
        best_xgb_params: dict  = {}
        best_lgbm_params: dict = {}

        with mlflow.start_run(run_name=f"{ticker}_walk_forward"):
            mlflow.set_tag("ticker",    ticker)
            mlflow.set_tag("phase",     "03")
            mlflow.log_param("horizon",         horizon)
            mlflow.log_param("threshold",       threshold)
            mlflow.log_param("n_splits",        n_splits)
            mlflow.log_param("embargo_days",    embargo_days)
            mlflow.log_param("n_trials_xgb",    n_trials_xgb)
            mlflow.log_param("n_trials_lgbm",   n_trials_lgbm)
            mlflow.log_param("lstm_epochs",     lstm_epochs)
            mlflow.log_param("lstm_seq_len",    lstm_seq_len)
            mlflow.log_param("skip_lstm",       skip_lstm)
            mlflow.log_param("n_features",      len(feat_cols))
            mlflow.log_param("feature_cols",    json.dumps(feat_cols))

            for fold, (tr_idx, te_idx) in enumerate(splitter.split(X_arr)):
                fold_start = time.time()
                log.info(
                    "fold %d: train [%d..%d] (%d rows)  test [%d..%d] (%d rows)",
                    fold, tr_idx[0], tr_idx[-1], len(tr_idx),
                    te_idx[0], te_idx[-1], len(te_idx),
                )

                X_tr, y_tr = X_arr[tr_idx], y_arr[tr_idx]
                X_te, y_te = X_arr[te_idx], y_arr[te_idx]

                # ── XGBoost ──────────────────────────────────────────
                log.info("  [XGB] tuning %d trials …", n_trials_xgb)
                xgb = XGBModel(n_trials=n_trials_xgb, random_state=random_state)
                xgb.fit(X_tr, y_tr, class_weight_map=cw)
                xgb_proba_te = xgb.predict_proba(X_te)
                oos_preds_xgb[te_idx] = xgb_proba_te
                best_xgb_params  = xgb.best_params_
                last_xgb = xgb

                xgb_preds = np.argmax(xgb_proba_te, axis=1)
                xgb_m     = _fold_metrics(y_te, xgb_preds)
                log.info("  [XGB] fold %d  acc=%.3f  f1=%.3f", fold, xgb_m["accuracy"], xgb_m["f1_macro"])

                # ── LightGBM ─────────────────────────────────────────
                log.info("  [LGBM] tuning %d trials …", n_trials_lgbm)
                lgbm = LGBMModel(n_trials=n_trials_lgbm, random_state=random_state)
                lgbm.fit(X_tr, y_tr, class_weight_map=cw)
                lgbm_proba_te = lgbm.predict_proba(X_te)
                oos_preds_lgbm[te_idx] = lgbm_proba_te
                best_lgbm_params = lgbm.best_params_
                last_lgbm = lgbm

                lgbm_preds = np.argmax(lgbm_proba_te, axis=1)
                lgbm_m     = _fold_metrics(y_te, lgbm_preds)
                log.info("  [LGBM] fold %d  acc=%.3f  f1=%.3f", fold, lgbm_m["accuracy"], lgbm_m["f1_macro"])

                # ── LSTM ──────────────────────────────────────────────
                if not skip_lstm:
                    log.info("  [LSTM] training (max %d epochs) …", lstm_epochs)
                    lstm = LSTMModel(
                        seq_len  = lstm_seq_len,
                        n_epochs = lstm_epochs,
                        random_state = random_state,
                    )
                    lstm.fit(X_tr, y_tr, class_weight_map=cw)
                    lstm_proba_te = lstm.predict_proba(X_te)
                    oos_preds_lstm[te_idx] = lstm_proba_te
                    last_lstm = lstm

                    lstm_preds = np.argmax(lstm_proba_te, axis=1)
                    lstm_m     = _fold_metrics(y_te, lstm_preds)
                    log.info("  [LSTM] fold %d  acc=%.3f  f1=%.3f", fold, lstm_m["accuracy"], lstm_m["f1_macro"])
                else:
                    lstm_m = {"accuracy": 0.0, "f1_macro": 0.0}

                # ── Ensemble ──────────────────────────────────────────
                ens_weights = [xgb_m["f1_macro"], lgbm_m["f1_macro"], lstm_m["f1_macro"]] \
                              if not skip_lstm else [0.5, 0.5, 0.0]
                ens = EnsembleClassifier(weights=ens_weights)
                ens_proba  = ens.predict_proba(
                    xgb_proba_te, lgbm_proba_te,
                    oos_preds_lstm[te_idx] if not skip_lstm else np.full((len(te_idx), 3), 1.0/3),
                )
                ens_preds  = np.argmax(ens_proba, axis=1)
                ens_m      = _fold_metrics(y_te, ens_preds)
                log.info("  [ENS] fold %d  acc=%.3f  f1=%.3f", fold, ens_m["accuracy"], ens_m["f1_macro"])

                elapsed = time.time() - fold_start
                fold_r = {
                    "fold":        fold,
                    "train_rows":  len(tr_idx),
                    "test_rows":   len(te_idx),
                    "elapsed_s":   elapsed,
                    "xgb":         xgb_m,
                    "lgbm":        lgbm_m,
                    "lstm":        lstm_m,
                    "ensemble":    ens_m,
                }
                fold_results.append(fold_r)

                # MLflow per-fold metrics
                for model_name, m in [("xgb", xgb_m), ("lgbm", lgbm_m), ("ens", ens_m)]:
                    for k, v in m.items():
                        mlflow.log_metric(f"fold{fold}_{model_name}_{k}", v, step=fold)

            # ── Aggregate OOS metrics ─────────────────────────────────
            # Rows that appeared in at least one test fold
            tested_mask = ~np.isnan(oos_preds_xgb[:, 0])
            if tested_mask.sum() > 0:
                y_oos      = y_arr[tested_mask]
                ens_final  = EnsembleClassifier()
                oos_proba  = ens_final.predict_proba(
                    oos_preds_xgb[tested_mask],
                    oos_preds_lgbm[tested_mask],
                    oos_preds_lstm[tested_mask],
                )
                oos_pred   = np.argmax(oos_proba, axis=1)
                oos_m      = _fold_metrics(y_oos, oos_pred)
            else:
                oos_m = {}

            mlflow.log_metrics({f"oos_{k}": v for k, v in oos_m.items()})
            log.info("OOS ensemble  acc=%.3f  f1=%.3f", oos_m.get("accuracy", 0), oos_m.get("f1_macro", 0))

        # ── Persist OOS predictions for Phase 04 backtesting ─────────
        if tested_mask.sum() > 0:
            oos_df = pd.DataFrame(
                {
                    "p_down":    oos_preds_xgb[tested_mask, 0] * self._w(skip_lstm)[0]
                                 + oos_preds_lgbm[tested_mask, 0] * self._w(skip_lstm)[1]
                                 + oos_preds_lstm[tested_mask, 0] * self._w(skip_lstm)[2],
                    "p_flat":    oos_preds_xgb[tested_mask, 1] * self._w(skip_lstm)[0]
                                 + oos_preds_lgbm[tested_mask, 1] * self._w(skip_lstm)[1]
                                 + oos_preds_lstm[tested_mask, 1] * self._w(skip_lstm)[2],
                    "p_up":      oos_preds_xgb[tested_mask, 2] * self._w(skip_lstm)[0]
                                 + oos_preds_lgbm[tested_mask, 2] * self._w(skip_lstm)[1]
                                 + oos_preds_lstm[tested_mask, 2] * self._w(skip_lstm)[2],
                    "true_label": y_arr[tested_mask],
                },
                index = X.index[tested_mask],
            )
            oos_path = self.models_dir / ticker / "oos_predictions.parquet"
            oos_path.parent.mkdir(parents=True, exist_ok=True)
            import pyarrow as pa, pyarrow.parquet as pq
            pq.write_table(pa.Table.from_pandas(oos_df, preserve_index=True),
                           str(oos_path), compression="snappy")
            log.info("OOS predictions saved → %s  (%d rows)", oos_path, len(oos_df))

        # ── Save final models (retrained on all data) ─────────────────
        model_paths = self._save_final_models(
            ticker, feat_cols, last_xgb, last_lgbm,
            last_lstm if not skip_lstm else None,
        )

        elapsed_total = time.time() - t0
        results = {
            "ticker":           ticker,
            "n_folds":          len(fold_results),
            "fold_metrics":     fold_results,
            "oos_metrics":      oos_m,
            "best_params_xgb":  best_xgb_params,
            "best_params_lgbm": best_lgbm_params,
            "feature_cols":     feat_cols,
            "model_paths":      model_paths,
            "elapsed_s":        elapsed_total,
        }

        # Persist results summary to JSON
        out_json = self.models_dir / ticker / "training_results.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(
                {k: v for k, v in results.items() if k != "fold_metrics"},
                f, indent=2, default=str,
            )
        log.info("=== ModelTrainer.run(%s) done in %.0fs ===", ticker, elapsed_total)
        return results

    @staticmethod
    def _w(skip_lstm: bool) -> tuple[float, float, float]:
        return (0.5, 0.5, 0.0) if skip_lstm else (1/3, 1/3, 1/3)

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def _save_final_models(
        self,
        ticker:    str,
        feat_cols: list[str],
        xgb:       Optional[XGBModel],
        lgbm:      Optional[LGBMModel],
        lstm:      Optional[LSTMModel],
    ) -> dict[str, str]:
        out_dir = self.models_dir / ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}

        if xgb is not None and xgb.model_ is not None:
            p = str(out_dir / "xgb_final.json")
            xgb.save(p)
            paths["xgb"] = p
            log.info("Saved XGB → %s", p)

        if lgbm is not None and lgbm.model_ is not None:
            p = str(out_dir / "lgbm_final.txt")
            lgbm.save(p)
            paths["lgbm"] = p
            log.info("Saved LGBM → %s", p)

        if lstm is not None and lstm.net_ is not None:
            p = str(out_dir / "lstm_final.pt")
            lstm.save(p)
            paths["lstm"] = p
            log.info("Saved LSTM → %s", p)

        # Persist selected feature list
        feat_path = out_dir / "feature_cols.json"
        feat_path.write_text(json.dumps(feat_cols, indent=2))
        paths["feature_cols"] = str(feat_path)

        return paths

    # ------------------------------------------------------------------
    # Load trained models for inference
    # ------------------------------------------------------------------

    def load_models(
        self,
        ticker: str,
    ) -> tuple[XGBModel, LGBMModel, Optional[LSTMModel], list[str]]:
        """Load saved models and return (xgb, lgbm, lstm, feature_cols)."""
        out_dir = self.models_dir / ticker

        xgb_path  = str(out_dir / "xgb_final.json")
        lgbm_path = str(out_dir / "lgbm_final.txt")
        lstm_path = str(out_dir / "lstm_final.pt")
        feat_path = out_dir / "feature_cols.json"

        xgb  = XGBModel.load(xgb_path)
        lgbm = LGBMModel.load(lgbm_path)
        lstm = LSTMModel.load(lstm_path) if Path(lstm_path).exists() else None
        feat_cols = json.loads(feat_path.read_text())

        return xgb, lgbm, lstm, feat_cols
