"""Export a few recorded context-adapter decisions for the research website."""

import csv
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "data/ceci2-demo/colab-t4-v1/custom-gates"
ASSETS = ROOT / "site/assets"
METHODS = ("Fixed mixture", "Context adapter")
POOLS = ("sequence networks", "trees and linear", "all experts")
DAYS = ("2024-01-09", "2025-12-22")
SYMBOLS = ("AAPL", "GOOGL")
CLASSES = ("down", "flat", "up")


def read(path):
    return json.loads(path.read_text())


def export():
    fits = read(ARTIFACTS / "checkpoints.json")
    with gzip.open(ARTIFACTS / "forecasts.csv.gz", "rt", newline="") as file:
        archive = {
            (row["decision"], row["symbol"], row["model"]): [
                float(row[label]) for label in CLASSES
            ]
            for row in csv.DictReader(file)
            if row["decision"] in DAYS and row["symbol"] in SYMBOLS
        }

    examples = []
    for day in DAYS:
        year = day[:4]
        records = read(ASSETS / f"detail-{year}.json")
        day_index = records["dates"].index(day)
        for symbol in SYMBOLS:
            symbol_index = records["symbols"].index(symbol)
            context = records["context"][day_index][symbol_index]
            results = {}
            for pool in POOLS:
                results[pool] = {}
                for method in METHODS:
                    name = f"{method} / {pool}"
                    fit = fits[f"{year}: {name}"]
                    if fit["context"] != records["contextNames"]:
                        raise ValueError("Context column order differs from the training data")
                    scores = np.asarray(fit["intercept"])
                    if fit["coef"]:
                        scaled = np.clip((np.asarray(context) - fit["mean"]) / fit["scale"], -10, 10)
                        scores = scores + scaled @ np.asarray(fit["coef"])
                    weights = np.exp(scores - np.max(scores))
                    weights /= weights.sum()
                    channels = fit["channels"]
                    indices = [records["experts"].index(channel) for channel in channels]
                    forecast = np.array(records["probabilities"][day_index][symbol_index])[indices]
                    forecast /= forecast.sum(axis=1, keepdims=True)
                    recorded = archive[(day, symbol, name)]
                    if not np.allclose(weights @ forecast, recorded, rtol=0, atol=1e-6):
                        raise ValueError(f"Replayed forecast differs from archive for {day} {symbol} {name}")
                    results[pool][method] = {
                        "weights": dict(zip(channels, weights.tolist())),
                        "forecast": recorded,
                    }
            examples.append(dict(day=day, symbol=symbol, context=context, results=results))
    payload = {"contextNames": records["contextNames"], "examples": examples}
    (ASSETS / "adapter-examples.json").write_text(json.dumps(payload, separators=(",", ":")) + "\n")

    with gzip.open(ARTIFACTS.parent.parent / "moe-v2/forecasts.csv.gz", "rt", newline="") as file:
        classic = [{"day": row["decision"][:10], "symbol": row["symbol"],
                    "forecast": [float(row[label]) for label in CLASSES]}
                   for row in csv.DictReader(file)
                   if row["decision"][:10] in DAYS and row["symbol"] in SYMBOLS
                   and row["model"] == "Classic MoE"]
    if len(classic) != len(examples):
        raise ValueError("The classic mixture is missing a decision from the example set")
    (ASSETS / "classic-votes.json").write_text(json.dumps({"examples": classic}) + "\n")


if __name__ == "__main__":
    export()
