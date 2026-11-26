"""Train three new neural experts and a contextual mixture on recorded as-of inputs."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import torch
from torch import nn

REVISION = "ffb3c92c79b9cf19b9a93c2b7d5b68e6a70ce43e"
BASE = f"https://raw.githubusercontent.com/Yash789k/ML-TradeEngine/{REVISION}/"
KEY = ["decision", "symbol"]
PROBS = ["down", "flat", "up"]
HEADS = ["Numerical sequence", "Event sentiment", "Market context"]
MODELS = ["Joint neural mixture", "Joint experts uniform"]
NUMERIC = ["vol", "range", "drawdown", "r21", "atr_ratio", "log_volume"]
EVENT = ["fed_positive", "headline_tone", "annual_report_tone"]
MARKET = ["market", "spy_vol", "spy_r21", "positive_breadth"]


def source(relative, checkout, cache):
    if checkout is not None:
        return Path(checkout) / relative
    path = cache / REVISION / relative
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(BASE + relative, temporary)
        temporary.replace(path)
    return path


def inputs(checkout, cache):
    path = source("data/ceci2-demo/five-session/predictions.csv.gz", checkout, cache)
    cols = KEY + [
        "model",
        "label",
        "endpoint",
        "mark",
        "entry",
        "exit",
        "vol",
        "atr",
        "close",
        "volume",
        "r21",
    ]
    with pd.read_csv(
        path,
        usecols=cols,
        chunksize=50000,
        parse_dates=["decision", "endpoint", "mark"],
    ) as chunks:
        tape = pd.concat(
            [chunk[chunk.model == "HMM gate"] for chunk in chunks]
        ).set_index(KEY)
        if not tape.index.is_unique:
            raise ValueError("Replayed instrument decisions must be unique")
    parts = []
    for year in (2023, 2024, 2025):
        detail = json.loads(
            source(f"site/assets/detail-{year}.json", checkout, cache).read_text()
        )
        key = pd.MultiIndex.from_product(
            [pd.to_datetime(detail["dates"]), detail["symbols"]], names=KEY
        )
        if (
            not key.is_unique
            or len(key) != (tape.index.get_level_values("decision").year == year).sum()
        ):
            raise ValueError("Input grid differs between the two source artifacts")
        block = tape.loc[key].copy()
        context = np.asarray(detail["context"]).reshape(
            len(key), len(detail["contextNames"])
        )
        for field in ["range", "drawdown", "market", *EVENT]:
            block[field] = context[:, detail["contextNames"].index(field)]
        parts.append(block)
    frame = pd.concat(parts).reset_index().sort_values(KEY, ignore_index=True)
    frame["atr_ratio"] = frame.atr / frame.close
    frame["log_volume"] = np.log1p(frame.volume)
    spy = frame[frame.symbol == "SPY"].set_index("decision")
    frame["spy_vol"] = frame.decision.map(spy.vol)
    frame["spy_r21"] = frame.decision.map(spy.r21)
    frame["positive_breadth"] = (
        (frame.r21 > 0).groupby(frame.decision).transform("mean")
    )
    if (
        not np.isfinite(frame[NUMERIC + EVENT + MARKET].to_numpy()).all()
        or not (frame.endpoint > frame.decision).all()
    ):
        raise ValueError("Invalid as-of features or label dates")
    return frame


def sequences(frame, values):
    result = np.empty((len(frame), 8, values.shape[1]), dtype=np.float32)
    # Do not cross the gaps between calendar-year tapes.
    for _, part in frame.groupby(["symbol", frame.decision.dt.year], sort=False):
        idx = part.index.to_numpy()
        if not part.decision.is_monotonic_increasing:
            raise ValueError("Unsorted instrument history")
        back = np.maximum(0, np.arange(len(idx))[:, None] - np.arange(7, -1, -1))
        result[idx] = values[idx[back]]
    return result


class JointMixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence = nn.GRU(len(NUMERIC), 8, batch_first=True)
        self.sequence_head = nn.Linear(8, 3)
        self.event = nn.Sequential(nn.Linear(len(EVENT), 8), nn.Tanh(), nn.Linear(8, 3))
        self.market = nn.Linear(len(MARKET), 3)
        self.gate = nn.Sequential(
            nn.Linear(len(NUMERIC) + len(EVENT) + len(MARKET), 8),
            nn.Tanh(),
            nn.Linear(8, 3),
        )

    def forward(self, numeric, event, market):
        _, hidden = self.sequence(numeric)
        heads = torch.stack(
            [self.sequence_head(hidden[-1]), self.event(event), self.market(market)],
            dim=1,
        ).log_softmax(-1)
        context = torch.cat([numeric[:, -1], event, market], dim=1)
        log_weights = self.gate(context).log_softmax(-1)
        mixture = torch.logsumexp(heads + log_weights.unsqueeze(-1), dim=1)
        return mixture, heads, log_weights


def fold(frame, year, output, device):
    test = frame.decision.dt.year == year
    first = frame.loc[test, "decision"].min()
    train = (frame.decision.dt.year < year) & (frame.endpoint < first)
    columns = NUMERIC + EVENT + MARKET
    fit_x = frame.loc[train, columns]
    mean = fit_x.mean().to_numpy()
    scale = fit_x.std(ddof=0).clip(lower=1e-6).to_numpy()
    scaled = (
        ((frame[columns].to_numpy() - mean) / scale).clip(-10, 10).astype("float32")
    )
    n, m = len(NUMERIC), len(NUMERIC) + len(EVENT)
    x = [
        torch.as_tensor(z, device=device)
        for z in (sequences(frame, scaled[:, :n]), scaled[:, n:m], scaled[:, m:])
    ]
    label = torch.as_tensor(
        frame.label.to_numpy(copy=True), dtype=torch.long, device=device
    )
    train_idx = torch.as_tensor(np.flatnonzero(train), device=device)
    test_idx = torch.as_tensor(np.flatnonzero(test), device=device)
    torch.manual_seed(28)
    net = JointMixture().to(device)
    initial = {k: v.detach().clone() for k, v in net.named_parameters()}
    opt = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.01)
    history = []
    for _ in range(12):
        total = 0.0
        for i in train_idx[torch.randperm(len(train_idx), device=device)].split(256):
            opt.zero_grad()
            mixed, heads, _ = net(*(v[i] for v in x))
            head_loss = nn.functional.nll_loss(
                heads.reshape(-1, 3), label[i].repeat_interleave(3)
            )
            loss = nn.functional.nll_loss(mixed, label[i]) + 0.2 * head_loss
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(i)
        history.append(total / len(train_idx))
    moved = {
        k: float((v.detach() - initial[k]).norm().cpu())
        for k, v in net.named_parameters()
    }
    if not all(np.isfinite(list(moved.values()))) or min(moved.values()) <= 0:
        raise ValueError("An expert or gate parameter failed to train")
    torch.save(
        {
            "parameters": {k: v.cpu() for k, v in net.state_dict().items()},
            "features": columns,
            "mean": mean.tolist(),
            "scale": scale.tolist(),
        },
        output / f"{year}-joint.pt",
    )
    net.eval()
    with torch.inference_mode():
        logmix, logheads, logw = net(*(v[test_idx] for v in x))
        forecasts = [logmix.exp().cpu().numpy(), logheads.exp().mean(1).cpu().numpy()]
        heads, weights = logheads.exp().cpu().numpy(), logw.exp().cpu().numpy()
    if not np.allclose((heads * weights[:, :, None]).sum(1), forecasts[0], atol=1e-6):
        raise ValueError("Mixture failed to reconstruct from its experts")
    return (
        frame.loc[test].copy(),
        forecasts,
        heads,
        weights,
        {
            "first_decision": str(first.date()),
            "last_training_endpoint": str(frame.loc[train, "endpoint"].max().date()),
            "training_cases": int(train.sum()),
            "test_cases": int(test.sum()),
            "epoch_train_loss": history,
            "parameter_change_l2": moved,
        },
    )


def paired_interval(delta):
    rng = np.random.default_rng(28)
    total = np.zeros(2000)
    for _, block in delta.groupby(delta.index.year):
        n = len(block)
        start = rng.integers(n, size=(2000, (n + 20) // 21))
        idx = ((start[:, :, None] + np.arange(21)) % n).reshape(2000, -1)[:, :n]
        total += block.to_numpy()[idx].sum(1)
    return np.quantile(total / len(delta), [0.025, 0.975]).tolist()


def experiment(output, device=None, checkout=None):
    """Run both folds; checkout=None downloads the pinned public source artifacts."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cache = Path(os.environ.get("CECI2_CACHE", "/tmp/ceci2-inputs"))
    script = source("ceci2_demo/evaluate.py", checkout, cache)
    spec = importlib.util.spec_from_file_location("shared_execution", script)
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_num_threads(1)
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print("Training three experts and a gate from scratch on", hardware, flush=True)
    frame = inputs(checkout, cache)
    metadata = {
        "revision": REVISION,
        "device": str(device),
        "hardware": hardware,
        "torch": str(torch.__version__),
        "folds": {},
    }
    cases, expert_rows = [], []
    for year in (2024, 2025):
        test, forecasts, heads, weights, fit = fold(frame, year, output, device)
        fit["mean_expert_weights"] = dict(
            zip(HEADS, weights.mean(axis=0, dtype=np.float64).tolist())
        )
        metadata["folds"][year] = fit
        for name, pred in zip(MODELS, forecasts):
            chosen = test.copy()
            chosen["model"] = name
            chosen[PROBS] = pred
            cases.append(chosen)
        for i, expert in enumerate(HEADS):
            summary = test[KEY + ["label"]].copy()
            summary["expert"] = expert
            summary["weight"] = weights[:, i]
            summary[PROBS] = heads[:, i]
            expert_rows.append(summary)
        print("Completed", year, "on", hardware, flush=True)
    all_cases = pd.concat(cases)
    all_cases[KEY + ["model", "label", "endpoint"] + PROBS].to_csv(
        output / "forecasts.csv.gz", index=False
    )
    pd.concat(expert_rows).to_csv(output / "experts.csv.gz", index=False)
    ledgers, losses = [], {}
    for name, group in all_cases.groupby("model"):
        ledgers.append(shared.replay(group, name))
        pred = group[PROBS].to_numpy()
        nll = -np.log(pred[np.arange(len(group)), group.label.to_numpy()].clip(1e-12))
        losses[name] = pd.Series(nll, index=group.decision).groupby(level=0).mean()
    daily = pd.concat(ledgers, ignore_index=True)
    daily.to_csv(output / "daily.csv.gz", index=False)
    losses = pd.DataFrame(losses)
    losses.to_csv(output / "daily-loss.csv")
    delta = losses[MODELS[0]] - losses[MODELS[1]]
    metadata.update(
        {
            "sessions": len(losses),
            "log_loss": losses.mean().to_dict(),
            "context_minus_uniform_log_loss": delta.mean(),
            "paired_95_interval": paired_interval(delta),
        }
    )
    (output / "result.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k != "folds"}, indent=2))
    return metadata


if __name__ == "__main__" and "__file__" in globals():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()
    experiment(args.output, device=args.device, checkout=args.checkout)
