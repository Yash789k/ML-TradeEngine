"""Bootstrap schema-checked bundles for local signal dry runs; never place orders."""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.loader import DataLoader
from src.features.pipeline import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.models.bundle import validate_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]
    )
    args = parser.parse_args()
    loader = DataLoader()
    fe = FeatureEngineer()
    spy = loader.load_equity("SPY", force_refresh=True)
    trainer = None
    for ticker in args.tickers:
        raw = loader.load_equity(ticker, force_refresh=True)
        feat = fe.compute_features(raw, ticker, spy_df=spy, force_refresh=True)
        try:
            metadata = validate_bundle(ROOT / "data/models" / ticker)
            if all(
                metadata.get(k) == feat.attrs.get(k)
                for k in ["feature_history_start", "hmm_calibration_sha256"]
            ):
                continue
        except (ValueError, FileNotFoundError):
            pass
        if trainer is None:
            trainer = ModelTrainer(
                mlflow_uri="sqlite:///" + str(ROOT / "data/bootstrap-tracking.db")
            )
        # Leave new observations after training for live inference. This is a
        # functional bootstrap, not research selection or promotion to trading.
        trainer.run(
            feat.iloc[:-6],
            ticker=ticker,
            n_splits=2,
            n_trials_xgb=1,
            n_trials_lgbm=1,
            lstm_epochs=3,
        )
        validate_bundle(ROOT / "data/models" / ticker)


if __name__ == "__main__":
    main()
