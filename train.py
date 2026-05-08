"""
Phase 03 — Training Entry Point

Trains XGBoost + LightGBM + LSTM ensemble for one or more assets using the
Phase 02 feature matrices cached in data/features/.

Quick start
-----------
    # Full run (recommended, 1–2 h depending on hardware)
    python3 train.py --ticker AAPL

    # Skip LSTM to iterate faster (~10–15 min)
    python3 train.py --ticker AAPL --skip-lstm

    # Multiple tickers
    python3 train.py --ticker AAPL MSFT SPY

    # Reduce Optuna trials for a smoke-test run
    python3 train.py --ticker AAPL --n-trials 5 --lstm-epochs 5 --skip-lstm

MLflow UI
---------
    mlflow ui --port 5000
    # then open http://localhost:5000

Results
-------
Trained models saved under data/models/{ticker}/
MLflow run metadata in mlruns/
Per-ticker JSON summary in data/models/{ticker}/training_results.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH regardless of how the script is invoked
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 03 — ML Trade Engine training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Asset selection
    p.add_argument(
        "--ticker", nargs="+", default=["AAPL"],
        help="One or more ticker symbols to train on (must exist in data/features/).",
    )
    p.add_argument(
        "--all-tickers", action="store_true",
        help="Train all default tickers: AAPL MSFT GOOGL SPY QQQ",
    )

    # Label parameters
    p.add_argument("--horizon",    type=int,   default=5,     help="Forward-return horizon (bars)")
    p.add_argument("--threshold",  type=float, default=0.005, help="±pct threshold for UP/DOWN label")

    # Cross-validation
    p.add_argument("--n-splits",     type=int, default=5,  help="Walk-forward CV folds")
    p.add_argument("--embargo-days", type=int, default=10, help="Purge gap between train and test")

    # XGBoost / LightGBM tuning
    p.add_argument("--n-trials",    type=int, default=50, help="Optuna trials (applied to both XGB and LGBM)")

    # LSTM
    p.add_argument("--lstm-epochs",  type=int, default=50, help="Max LSTM training epochs per fold")
    p.add_argument("--lstm-seq-len", type=int, default=60, help="LSTM sequence window length (bars)")
    p.add_argument("--skip-lstm",    action="store_true",  help="Skip LSTM (much faster; tree ensemble only)")

    # Feature selection
    p.add_argument("--top-n-features", type=int, default=25, help="Max features after SHAP selection")

    # Misc
    p.add_argument("--random-state",   type=int, default=42)
    p.add_argument("--mlflow-uri",     default="mlruns",    help="MLflow tracking URI")

    return p.parse_args()


def _load_or_build_features(fe, ticker: str, max_age_hours: float = 8760.0):
    """
    Return the Phase 02 feature matrix for `ticker`.

    Loads from Parquet cache if the file exists (any age is acceptable for
    training — daily features don't change intra-day).  If the file is missing,
    auto-builds it by fetching raw data first.
    """
    from src.features.pipeline import _feature_path, _is_stale, _read_parquet

    path = _feature_path(ticker)

    # Treat any existing cache as valid for training (1-year TTL)
    if path.exists() and not _is_stale(path, max_age_hours):
        return _read_parquet(path)

    # Cache missing or truly ancient — rebuild from raw data
    log.info("Feature cache not found for %s — building now …", ticker)
    from src.data.loader import DataLoader
    loader = DataLoader()

    raw   = loader.load_equity(ticker)
    spy   = loader.load_equity("SPY")
    btc   = loader.load_crypto("BTC/USDT", limit=1825)
    macro = loader.load_macro()

    return fe.compute_features(
        df         = raw,
        asset_slug = ticker,
        spy_df     = spy,
        btc_df     = btc,
        macro_df   = macro,
    )


def main() -> None:
    args = parse_args()

    from src.features.pipeline import FeatureEngineer
    from src.models.trainer    import ModelTrainer

    _DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]
    tickers = _DEFAULT_TICKERS if args.all_tickers else args.ticker

    fe      = FeatureEngineer()
    trainer = ModelTrainer(mlflow_uri=args.mlflow_uri)

    for ticker in tickers:
        log.info("══════════════════════════════════════════")
        log.info("  Training: %s", ticker)
        log.info("══════════════════════════════════════════")

        features = _load_or_build_features(fe, ticker)
        log.info("Loaded feature matrix: %d rows × %d cols", *features.shape)

        results = trainer.run(
            features        = features,
            ticker          = ticker,
            horizon         = args.horizon,
            threshold       = args.threshold,
            n_splits        = args.n_splits,
            embargo_days    = args.embargo_days,
            n_trials_xgb    = args.n_trials,
            n_trials_lgbm   = args.n_trials,
            lstm_epochs     = args.lstm_epochs,
            lstm_seq_len    = args.lstm_seq_len,
            top_n_features  = args.top_n_features,
            skip_lstm       = args.skip_lstm,
            random_state    = args.random_state,
        )

        oos = results["oos_metrics"]
        log.info(
            "  OOS results  acc=%.3f  f1_macro=%.3f  "
            "(DOWN=%.3f  FLAT=%.3f  UP=%.3f)",
            oos.get("accuracy",  0),
            oos.get("f1_macro",  0),
            oos.get("f1_down",   0),
            oos.get("f1_flat",   0),
            oos.get("f1_up",     0),
        )
        log.info("  Models saved to: data/models/%s/", ticker)

    log.info("Training complete. Launch MLflow UI with: mlflow ui --port 5000")


if __name__ == "__main__":
    main()
