"""Versioned model packaging. Legacy bundles must be explicitly retrained."""

import hashlib
import json
from pathlib import Path


def seal_bundle(directory):
    directory = Path(directory)
    names = [
        "feature_cols.json",
        "training_results.json",
        "xgb_final.json",
        "xgb_final_scaler.pkl",
        "lgbm_final.txt",
        "lgbm_final_scaler.pkl",
    ]
    meta = json.loads((directory / "training_results.json").read_text())
    if not meta.get("skip_lstm"):
        names += ["lstm_final.pt", "lstm_final_scaler.pkl"]
    names += [
        name
        for name in ["oos_predictions.parquet", "oos_predictions_per_model.parquet"]
        if (directory / name).exists()
    ]
    hashes = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in names
    }
    (directory / "bundle.json").write_text(
        json.dumps({"version": 2, "files": hashes}, indent=2)
    )


def validate_bundle(directory):
    directory = Path(directory)
    if not (directory / "bundle.json").exists():
        raise ValueError(
            f"{directory}: missing versioned bundle; retrain using repaired pipeline"
        )
    bundle = json.loads((directory / "bundle.json").read_text())
    if bundle["version"] != 2:
        raise ValueError("Unsupported model bundle version")
    for name, digest in bundle["files"].items():
        path = directory / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError(f"Incomplete or altered model artifact: {name}")
    meta = json.loads((directory / "training_results.json").read_text())
    if meta.get("causality_version") != 2:
        raise ValueError("Model was not produced by the repaired training pipeline")
    return meta
