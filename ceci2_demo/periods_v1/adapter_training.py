"""Fit the registered static and contextual gates on saved out-of-time forecasts."""
import importlib.util
import json
import os
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import torch
from torch import nn


REVISION = "59853149b3fe9c19bba043e9a61965ed268c89b4"
BASE = f"https://raw.githubusercontent.com/Yash789k/ML-TradeEngine/{REVISION}/"
POOLS = {"all experts": None, "trees and linear": ["Logistic", "XGBoost", "LightGBM"],
         "sequence networks": ["LSTM", "GRU", "TCN", "Patch Transformer"]}


def source(relative, checkout, cache):
    if checkout is not None:
        return checkout / relative
    path = cache / REVISION / relative
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(BASE + relative, temporary)
        temporary.replace(path)
    return path


def training_data(checkout, cache):
    tape = source("data/ceci2-demo/five-session/predictions.csv.gz", checkout, cache)
    # Read only one execution tape per instrument, independently of pool selection.
    with pd.read_csv(tape, chunksize=50000, parse_dates=["decision", "endpoint", "mark"]) as chunks:
        tape = pd.concat([c[c.model == "HMM gate"] for c in chunks]).set_index(["decision", "symbol"])
    panels, inputs, forecasts = [], [], []
    expert_names = contexts = None
    for year in (2023, 2024, 2025):
        path = source(f"site/assets/detail-{year}.json", checkout, cache)
        detail = json.loads(path.read_text())
        if expert_names is not None and (expert_names != detail["experts"] or contexts != detail["contextNames"]):
            raise ValueError("Expert or context schema changed between years")
        expert_names, contexts = detail["experts"], detail["contextNames"]
        keys = pd.MultiIndex.from_product([pd.to_datetime(detail["dates"]), detail["symbols"]], names=tape.index.names)
        panel = tape.loc[keys].reset_index()
        if len(panel) != len(keys) or not keys.is_unique:
            raise ValueError("Ambiguous case identifiers")
        panels.append(panel)
        inputs.append(np.array(detail["context"], dtype=np.float32).reshape(len(keys), len(contexts)))
        probs = np.array(detail["probabilities"], dtype=np.float32).reshape(len(keys), len(expert_names), 3)
        if not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(2), 1, atol=3e-6):
            raise ValueError("Invalid saved expert distributions")
        forecasts.append(probs / probs.sum(2, keepdims=True))
    return pd.concat(panels, ignore_index=True), np.concatenate(inputs), np.concatenate(forecasts), expert_names, contexts


class MixtureGate(nn.Module):
    def __init__(self, channels, context_width):
        super().__init__()
        self.intercept = nn.Parameter(torch.zeros(channels))
        self.coef = nn.Parameter(torch.zeros(context_width, channels))

    def forward(self, x, experts):
        logits = self.intercept + x @ self.coef if self.coef.numel() else self.intercept.expand(len(x), -1)
        weights = logits.softmax(dim=-1)
        return (weights.unsqueeze(-1) * experts).sum(dim=1), weights


def fit_one(context, probabilities, label, train, test, channels, adaptive, device):
    center, scale = context[train].mean(0), context[train].std(0).clip(1e-6)
    x = torch.as_tensor(((context - center) / scale).clip(-10, 10), device=device)
    expert = torch.as_tensor(probabilities[:, channels], device=device)
    split = torch.tensor(train, device=device)
    hold = torch.tensor(test, device=device)
    x_fit, p_fit = x[split], expert[split]
    truth = torch.as_tensor(label[train], dtype=torch.long, device=device)
    gate = MixtureGate(len(channels), x.shape[1] if adaptive else 0).to(device)
    optimizer = torch.optim.Adam(gate.parameters(), lr=.03)
    for _ in range(80):
        optimizer.zero_grad()
        mixed, _ = gate(x_fit, p_fit)
        nll = -mixed.gather(1, truth[:, None]).clamp_min(1e-15).log().mean()
        penalty = .001 * gate.coef.square().mean() if adaptive else 0
        (nll + penalty).backward()
        optimizer.step()
    with torch.no_grad():
        mixed, weights = gate(x[hold], expert[hold])
    model = {"mean": center.tolist(), "scale": scale.tolist(), "coef": gate.coef.detach().cpu().tolist(),
             "intercept": gate.intercept.detach().cpu().tolist()}
    return mixed.cpu().numpy(), weights.cpu().numpy(), model


def bootstrap(daily):
    change = daily["Context adapter / all experts"] - daily["Fixed mixture / all experts"]
    rng = np.random.default_rng(28)
    total = np.zeros(2000)
    for _, group in change.groupby(change.index.year):
        n = len(group)
        starts = rng.integers(n, size=(2000, (n+20)//21))
        samples = ((starts[:, :, None] + np.arange(21)) % n).reshape(2000, -1)[:, :n]
        total += group.to_numpy()[samples].sum(1)
    return {"context_minus_fixed_log_loss": float(change.mean()),
            "paired_95_interval": np.quantile(total / len(change), [.025, .975]).tolist()}


def experiment(output, checkout=None):
    output = Path(output)
    output.mkdir(exist_ok=False, parents=True)
    cache = Path(os.environ.get("CECI2_CACHE", "/tmp/ceci2-inputs"))
    if checkout is None:
        script = source("ceci2_demo/evaluate.py", None, cache)
        spec = importlib.util.spec_from_file_location("recorded_replay", script)
        ledger_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ledger_module)
        replay = ledger_module.replay
    else:
        from ceci2_demo.evaluate import replay
    torch.manual_seed(28)
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hardware = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print("Training 12 custom gates on", hardware, flush=True)
    frame, context, probabilities, names, contexts = training_data(checkout, cache)
    if not np.isfinite(context).all():
        raise ValueError("Invalid context")
    label = frame.label.to_numpy()
    fits, cases, cutoffs = {}, [], {}
    for year in (2024, 2025):
        test = (frame.decision.dt.year == year).to_numpy()
        start = frame.loc[test, "decision"].min()
        train = ((frame.decision < start) & (frame.endpoint < start)).to_numpy()
        cutoffs[year] = {"last_training_endpoint": str(frame.loc[train, "endpoint"].max().date()),
                         "first_test_decision": str(start.date()), "train_cases": int(train.sum()),
                         "test_cases": int(test.sum())}
        for pool, wanted in POOLS.items():
            chosen = wanted or names
            channels = [names.index(s) for s in chosen]
            for adaptive in (False, True):
                name = ("Context adapter" if adaptive else "Fixed mixture") + " / " + pool
                preds, gate_weights, fitted = fit_one(context, probabilities, label, train, test, channels, adaptive, device)
                cases_year = frame.loc[test].copy()
                cases_year["model"] = name
                cases_year[["down", "flat", "up"]] = preds
                cases_year["loss"] = -np.log(preds[np.arange(len(preds)), label[test]].clip(1e-15))
                cases.append(cases_year)
                fits[f"{year}: {name}"] = dict(fitted, channels=chosen, context=contexts,
                                                  mean_test_weight=gate_weights.mean(0).tolist())
        print("Finished", year, cutoffs[year], flush=True)
    cases = pd.concat(cases)
    ledgers = pd.concat([replay(g, name) for name, g in cases.groupby("model")])
    ledgers.to_csv(output/"daily.csv.gz", index=False)
    cases[["decision","symbol","model","label","down","flat","up","loss"]].to_csv(output/"forecasts.csv.gz",index=False)
    (output/"checkpoints.json").write_text(json.dumps(fits, indent=2))
    daily = cases.groupby(["decision","model"]).loss.mean().unstack()
    summary = []
    for name, g in ledgers.groupby("model"):
        for window, part in [("all", g), *g.groupby(g.decision.dt.year)]:
            curve = np.r_[1., (1+part.net_return).cumprod()]
            summary.append({"model":name,"period":str(window),"sessions":len(part),"return":curve[-1]-1,
                            "drawdown":(1-curve/np.maximum.accumulate(curve)).max(),"exposure":part.exposure.mean(),
                            "log_loss":daily.loc[part.decision, name].mean(),"fees":part.fees.sum()})
    table = pd.DataFrame(summary)
    table.to_csv(output/"comparison.csv",index=False)
    receipt = dict(source=REVISION, device=hardware, torch=str(torch.__version__), cutoffs=cutoffs, **bootstrap(daily))
    (output/"receipt.json").write_text(json.dumps(receipt, indent=2))
    print(table.loc[table.period == "all", ["model","return","drawdown","log_loss"]].to_string(index=False))
    print(json.dumps(receipt, indent=2))
    return ledgers, table


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, default=Path.cwd())
    args = parser.parse_args()
    experiment(args.output, args.checkout)
