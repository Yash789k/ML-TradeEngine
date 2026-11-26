"""Reconstruct a held-out forecast from locally trained model files."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.special import softmax

from src.models.gating import GatingModel
from .experts import SequenceExpert
from .panel import NUMERIC, TEXT


def predict(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    data = np.load(folder / "example.npz")
    x = ((data["window"] - data["center"]) / data["scale"]).clip(-10, 10)
    width = len(NUMERIC)
    language = dict(
        zip(
            (
                "Qwen statement",
                "OpenAI asset digest",
                "OpenAI Decisions",
                "Granite MoE statement",
            ),
            data["language"],
        )
    )
    probs = []
    for name in manifest["experts"]:
        if name in language:
            raw = language[name]
        elif (folder / f"{name}.pt").exists():
            network = SequenceExpert(name, width)
            network.load_state_dict(
                torch.load(folder / f"{name}.pt", weights_only=True, map_location="cpu")
            )
            network.eval()
            with torch.inference_mode():
                raw = network(torch.tensor(x[None, :, :width])).softmax(-1)[0].numpy()
        else:
            features = x[-1, :width]
            if name == "JKP factors":
                features = x[-1, width + len(TEXT) :]
            elif name == "FinBERT logistic":
                features = x[-1, width : width + len(TEXT)]
            raw = joblib.load(folder / f"{name}.joblib").predict_proba(features[None])[
                0
            ]
        probs.append(
            softmax(np.log(np.clip(raw, 1e-12, 1)) / manifest["temperature"][name])
        )
    weights = GatingModel.load(str(folder / "gate.joblib")).predict_weights(
        data["context"][None]
    )[0]
    output = weights @ np.stack(probs)
    if not np.allclose(output, manifest["expected"], atol=2e-6):
        raise ValueError("Reloaded forecast differs from the historical run")
    return dict(
        symbol=manifest["symbol"],
        date=manifest["decision"],
        probabilities=dict(zip(manifest["class_order"], output.tolist())),
        weights=dict(zip(manifest["experts"], weights.tolist())),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    print(json.dumps(predict(parser.parse_args().folder), indent=2))
