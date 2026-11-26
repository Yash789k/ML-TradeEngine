"""Fit a model-level reinforcement learner and send its expert forecasts to the shared simulator."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .adapter_training import POOLS, REVISION, training_data
from ceci2_demo.evaluate import replay


def mature_feedback(frame, probs):
    """Each symbol's rolling paid-off forecast losses, observed strictly before a decision."""
    loss = -np.log(np.take_along_axis(probs, frame.label.to_numpy()[:, None, None], axis=2)[:, :, 0].clip(1e-15))
    feedback = np.full_like(loss, np.log(3))
    for _, subset in frame.groupby("symbol", sort=False):
        rows = subset.index.to_numpy()
        endpoints = subset.endpoint.to_numpy()
        if np.any(endpoints[1:] < endpoints[:-1]):
            raise ValueError("Maturity order changed")
        available = np.searchsorted(endpoints, subset.decision.to_numpy(), side="left")
        start = np.maximum(available - 21, 0)
        prefix = np.vstack([np.zeros(loss.shape[1]), np.cumsum(loss[rows], axis=0, dtype=np.float64)])
        mask = available > 0
        feedback[rows[mask]] = (prefix[available[mask]] - prefix[start[mask]]) / (available-start)[mask, None]
    return loss, feedback


def fit_q(frame, state, loss, train, device):
    """Eight Bellman projections. Choosing expert a makes it the previous action at the next close."""
    selected = frame.loc[train, ["symbol","decision"]].reset_index(drop=True)
    n, actions = loss[train].shape
    next_row = np.arange(n)
    terminal = np.ones(n, dtype=bool)
    for _, block in selected.groupby(["symbol", selected.decision.dt.year]):
        ids = block.index.to_numpy()
        next_row[ids[:-1]] = ids[1:]
        terminal[ids[:-1]] = False
    center = state[train].mean(0, dtype=np.float64)
    scale = state[train].std(0, dtype=np.float64).clip(1e-6)
    source = ((state-center)/scale).clip(-10, 10)
    x = torch.as_tensor(np.column_stack([np.ones(len(state)), source]), dtype=torch.float64, device=device)
    train_x = x[torch.tensor(train, device=device)]
    reward = torch.as_tensor(-loss[train], dtype=torch.float64, device=device)
    next_row = torch.tensor(next_row, device=device)
    alive = torch.tensor(~terminal, device=device)
    reg = torch.eye(train_x.shape[1], dtype=x.dtype, device=device) * 100
    reg[0, 0] = 0
    matrix = train_x.T @ train_x + reg
    coefficients = torch.zeros((train_x.shape[1], actions), dtype=x.dtype, device=device)
    for _ in range(8):
        future = train_x[next_row] @ coefficients
        # Keeping the current expert costs zero. All alternative experts cost the same.
        next_best = torch.maximum(future, future.max(1, keepdim=True).values-.01)
        target = reward + .9 * alive[:, None] * next_best
        coefficients = torch.linalg.solve(matrix, train_x.T @ target)
    return (x @ coefficients).cpu().numpy(), {
        "center":center.tolist(),"scale":scale.tolist(),"coefficients":coefficients.cpu().tolist()
    }


def choose(frame, scores):
    # Each name carries its own previous expert. Annual folds start unconstrained.
    actions = np.zeros(len(frame), dtype=int)
    for _, group in frame.groupby("symbol"):
        prev = None
        for i in group.index:
            values = scores[i].copy()
            if prev is not None:
                values[prev] += .01
            prev = int(values.argmax())
            actions[i] = prev
    return actions


def mean_interval(delta):
    rng = np.random.default_rng(28)
    total = np.zeros(2000)
    for _, group in delta.groupby(delta.index.year):
        n = len(group)
        starts = rng.integers(n, size=(2000, (n+20)//21))
        idx = ((starts[:,:,None] + np.arange(21)) % n).reshape(2000, -1)[:,:n]
        total += group.to_numpy()[idx].sum(1)
    return np.quantile(total / len(delta), [.025,.975]).tolist()


def run(output, checkout):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame, context, probabilities, names, contexts = training_data(checkout, Path("/tmp/ceci2-inputs"))
    loss, feedback = mature_feedback(frame, probabilities)
    # Changing future labels may affect future feedback, but none at or before the changed date.
    cutoff = pd.Timestamp("2025-07-01")
    mutated = frame.copy()
    future = mutated.decision >= cutoff
    mutated.loc[future, "label"] = (mutated.loc[future, "label"] + 1) % 3
    changed = mature_feedback(mutated, probabilities)[1]
    if not np.array_equal(changed[frame.decision <= cutoff], feedback[frame.decision <= cutoff]):
        raise ValueError("Future labels affected prior model-routing inputs")
    cases, fits, cutoffs = [], {}, {}
    for year in (2024,2025):
        test = (frame.decision.dt.year == year).to_numpy()
        first = frame.loc[test, "decision"].min()
        train = ((frame.decision < first) & (frame.endpoint < first)).to_numpy()
        cutoffs[year] = {"latest_training_endpoint":str(frame.loc[train,"endpoint"].max().date()),
                         "first_test_decision":str(first.date()), "train_cases":int(train.sum()), "test_cases":int(test.sum())}
        for pool, wanted in POOLS.items():
            experts = names if wanted is None else wanted
            indices = [names.index(s) for s in experts]
            state = np.column_stack([context, feedback[:, indices]])
            scores, model = fit_q(frame, state, loss[:,indices], train, device)
            forecast = frame[test].copy().reset_index(drop=True)
            action = choose(forecast, scores[test])
            idx = np.array(indices)[action]
            p = probabilities[test][np.arange(len(idx)),idx]
            forecast[["down","flat","up"]] = p
            forecast["selected_expert"] = np.array(names)[idx]
            forecast["model"] = "RL model router / "+pool
            forecast["loss"] = -np.log(p[np.arange(len(idx)), forecast.label.to_numpy()].clip(1e-15))
            cases.append(forecast)
            fits[f"{year}: {pool}"] = dict(model,expert_names=experts,state=contexts+["mature_loss_"+x for x in experts])
    cases = pd.concat(cases)
    ledgers = pd.concat([replay(g,name) for name,g in cases.groupby("model")])
    ledgers.to_csv(output/"daily.csv.gz", index=False)
    cases[["decision","symbol","model","label","down","flat","up","selected_expert","loss"]].to_csv(output/"forecasts.csv.gz",index=False)
    (output/"checkpoints.json").write_text(json.dumps(fits,indent=2))
    summary = []
    for name, group in ledgers.groupby("model"):
        for period, seg in [("all",group),*group.groupby(group.decision.dt.year)]:
            curve = np.r_[1,(1+seg.net_return).cumprod()]
            forecast = cases[(cases.model==name)&cases.decision.isin(seg.decision)]
            summary.append(dict(model=name,period=str(period),sessions=len(seg),net_return=curve[-1]-1,
                                max_drawdown=(1-curve/np.maximum.accumulate(curve)).max(),exposure=seg.exposure.mean(),
                                fees=seg.fees.sum(),log_loss=forecast.loss.mean()))
    summary = pd.DataFrame(summary)
    summary.to_csv(output/"comparison.csv",index=False)
    receipt = dict(source=REVISION,device=torch.cuda.get_device_name(0) if device=="cuda" else "CPU",cutoffs=cutoffs,
                   past_contexts_unchanged_after_future_label_mutation=int((frame.decision<=cutoff).sum()))
    fixed_path = checkout/"data/ceci2-demo/adapter-pools-v1/forecasts.csv.gz" if checkout is not None else None
    if fixed_path is not None and fixed_path.exists():
        ref = pd.read_csv(fixed_path,parse_dates=["decision"])
        fixed = ref[ref.model=="Fixed mixture / all experts"].groupby("decision").loss.mean()
        result = cases[cases.model=="RL model router / all experts"].groupby("decision").loss.mean()
        if not fixed.index.equals(result.index):
            raise ValueError("Fixed-mixture reference covers different dates")
        delta = result-fixed
        receipt.update(primary_loss_change=float(delta.mean()),paired_95_interval=mean_interval(delta))
    (output/"receipt.json").write_text(json.dumps(receipt,indent=2))
    print(summary.loc[summary.period=="all"].to_string(index=False))
    print(json.dumps(receipt,indent=2))
    return ledgers, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkout", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.output,args.checkout)
