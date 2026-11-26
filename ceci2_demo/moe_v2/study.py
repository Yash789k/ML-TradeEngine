"""Run the registered classic mixture and offline allocation comparison."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn

from ceci2_demo.evaluate import replay
from ceci2_demo.panel import NUMERIC, TEXT, panel
from ceci2_demo.train import observations

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("protocol.json")
PROBS = ["down", "flat", "up"]
KEY = ["decision", "symbol"]
FEE = .0005


class DenseMixture(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate = nn.Linear(width, 3)
        self.experts = nn.Linear(width, 9)

    def forward(self, x):
        log_weights = self.gate(x).log_softmax(-1)
        log_expert = self.experts(x).reshape(-1, 3, 3).log_softmax(-1)
        return torch.logsumexp(log_weights.unsqueeze(-1) + log_expert, dim=1)


def fit_mixture(rows, year, features, output, name):
    train = rows.endpoint.dt.year < year - 2
    eligible = rows.endpoint.dt.year == rows.decision.dt.year
    for part in (year-2, year-1, year):
        dates = np.sort(rows.loc[rows.decision.dt.year == part, "decision"].unique())
        eligible &= (rows.decision.dt.year != part) | (rows.decision >= dates[5])
    scaler = StandardScaler().fit(rows.loc[train, features])
    x = torch.tensor(scaler.transform(rows[features]).clip(-10, 10), dtype=torch.float32)
    labels = torch.tensor(rows.label.values, dtype=torch.long)
    idx = torch.from_numpy(np.flatnonzero(train))
    torch.manual_seed(28)
    model = DenseMixture(len(features))
    opt = torch.optim.AdamW(model.parameters(), lr=.003)
    for _ in range(12):
        for batch in idx[torch.randperm(len(idx))].split(512):
            opt.zero_grad()
            nn.functional.nll_loss(model(x[batch]), labels[batch]).backward()
            opt.step()
    with torch.no_grad():
        logits = model(x).numpy()
    cal = eligible & (rows.decision.dt.year == year-2)
    def loss(temp):
        prob = softmax(logits[cal] / temp, axis=1)
        return -np.log(prob[np.arange(len(prob)), rows.loc[cal, "label"]]).mean()
    temp = float(minimize_scalar(loss, bounds=(.5, 5), method="bounded").x)
    torch.save(dict(features=features, mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                    temperature=temp, state=model.state_dict()), output / f"{year}-{name}.pt")
    keep = eligible & rows.decision.dt.year.isin([year-1, year])
    chosen = rows.loc[keep].copy()
    chosen[PROBS] = softmax(logits[keep] / temp, axis=1)
    return chosen


def environments(frame):
    frame = frame.sort_values(KEY)
    symbols = sorted(frame.symbol.unique())
    def mat(key):
        return frame.pivot(index="decision", columns="symbol", values=key)[symbols].to_numpy()
    score = mat("up") - mat("down")
    vol = mat("vol")
    top = np.argsort(-score, axis=1, kind="stable")[:, :5]
    selected = np.zeros_like(score, dtype=bool)
    np.put_along_axis(selected, top, True, axis=1)
    inv = np.where(selected & (score > .05), 1 / vol, 0)
    basket = np.minimum(.15, inv / np.maximum(inv.sum(1, keepdims=True), 1e-12))
    spy = symbols.index("SPY")
    weights = np.zeros((len(score), 3, len(symbols)))
    weights[:, 1, spy] = 1
    weights[:, 2] = basket
    state = np.column_stack((score.mean(1), score.std(1), mat("r21")[:, spy], vol[:, spy]))
    returns = mat("exit") / mat("entry") - 1
    if not np.isfinite(state).all() or not np.isfinite(returns).all():
        raise ValueError("Incomplete portfolio state")
    return frame, state, weights, returns


def transition_rewards(weights, returns):
    # From each preceding target action, mark its shares to the current opening.
    gross = (weights * returns[:, None, :]).sum(-1)
    drift = weights * (1 + returns[:, None, :]) / (1 + gross[:, :, None])
    previous = np.zeros_like(weights)
    previous[1:] = drift[:-1]
    before = previous[:, :, None, :]
    target = weights[:, None, :, :]
    after = np.ones((len(weights), 3, 3))
    for _ in range(6):
        after = 1 - FEE * np.abs(target * after[..., None] - before).sum(-1)
    wealth = after * (1 + gross[:, None, :])
    wealth[-1] -= FEE * after[-1] * (target[-1] * (1 + returns[-1])).sum(-1)
    return np.log(wealth)


def offline_policy(training, testing, output, year):
    _, state, weights, returns = environments(training)
    scaler = StandardScaler().fit(state)
    features = scaler.transform(state)
    rewards = transition_rewards(weights, returns)
    q = np.zeros_like(rewards)
    for _ in range(8):
        continuation = np.zeros_like(q[:, 0, :])
        continuation[:-1] = q[1:].max(2)
        target = rewards + .9 * continuation[:, None, :]
        reg = Ridge(alpha=10).fit(features, target.reshape(len(state), 9))
        q = reg.predict(features).reshape(-1, 3, 3)
    np.savez(output / f"{year}-policy.npz", mean=scaler.mean_, scale=scaler.scale_,
             coef=reg.coef_, intercept=reg.intercept_)
    frame, state, weights, _ = environments(testing)
    values = reg.predict(scaler.transform(state)).reshape(-1, 3, 3)
    last = 0
    actions = []
    for estimates in values:
        last = int(estimates[last].argmax())
        actions.append(last)
    selected = weights[np.arange(len(actions)), actions]
    target = pd.Series(selected.ravel(), index=frame.set_index(KEY).index)
    return target, pd.Series(actions, index=sorted(frame.decision.unique()), name="action")


def interval(delta):
    rng = np.random.default_rng(28)
    n = len(delta)
    starts = rng.integers(n, size=(2000, (n+20)//21))
    idx = ((starts[:, :, None] + np.arange(21)) % n).reshape(2000, -1)[:, :n]
    return np.quantile(delta[idx].mean(1), [.025, .975]).tolist()


def run(cache, out):
    out.mkdir(parents=True, exist_ok=False)
    (out/"protocol.json").write_bytes(PROTOCOL.read_bytes())
    torch.set_num_threads(1)
    frames, _ = panel(cache, 5)
    _, rows = observations(frames)
    rows = rows.rename_axis("decision").reset_index()
    base = NUMERIC + list(rows.filter(like="jkp_"))
    archive = pd.read_csv(ROOT/"data/ceci2-demo/five-session/predictions.csv.gz", parse_dates=["decision","mark"])
    incumbent = archive[archive.model == "HMM gate"].sort_values(KEY)
    ledgers, forecasts, targets, actions = [], [], [], []
    for year in (2023, 2024, 2025):
        for name, cols in [("Classic MoE", base+TEXT), ("MoE without text", base)]:
            chosen = fit_mixture(rows, year, cols, out, name)
            test = chosen[chosen.decision.dt.year == year].sort_values(KEY).copy()
            expected = incumbent[incumbent.decision.dt.year == year]
            if not test.set_index(KEY).index.equals(expected.set_index(KEY).index):
                raise ValueError("Dates or instruments differ from archived comparison")
            test["model"] = name
            forecasts.append(test)
            if name == "Classic MoE":
                target, action = offline_policy(chosen[chosen.decision.dt.year < year], test, out, year)
                targets.append(target)
                actions.append(action)
    new = pd.concat(forecasts)
    all_forecasts = pd.concat([new, incumbent, archive[archive.model == "Uniform vote"]])
    daily_loss = {}
    for name, frame in all_forecasts.groupby("model"):
        prob = frame[PROBS].to_numpy()
        nll = -np.log(prob[np.arange(len(frame)), frame.label.to_numpy()].clip(1e-15))
        daily_loss[name] = pd.Series(nll, index=frame.decision).groupby(level=0).mean()
        ledgers.append(replay(frame, name))
        if name in ("HMM gate", "Classic MoE"):
            ledgers.append(replay(frame, name+" every five sessions", cadence=5))
    classic = new[new.model=="Classic MoE"]
    rl = replay(classic, "MoE + offline RL", targets=pd.concat(targets))
    ledgers.append(rl)
    ledgers.append(replay(classic, "SPY at RL exposure", exposure=rl.set_index("decision").exposure))
    for existing in ["Classic MoE", "HMM gate"]:
        reference = next(v for v in ledgers if v.model.iloc[0] == existing)
        ledgers.append(replay(classic, "SPY at "+existing+" exposure",
                              exposure=reference.set_index("decision").exposure))
    for name in ["Cash","SPY buy and hold","Equal weight"]:
        ledgers.append(replay(classic, name))
    ledgers = pd.concat(ledgers)
    ledgers.to_csv(out/"daily.csv.gz", index=False)
    pd.concat(actions).rename_axis("decision").to_csv(out/"actions.csv")
    new[KEY+["model","label"]+PROBS+["news_available","filing_available"]].to_csv(out/"forecasts.csv.gz", index=False)
    loss = pd.DataFrame(daily_loss)
    loss.to_csv(out/"daily-loss.csv")
    delta = loss["HMM gate"]-loss["Classic MoE"]
    result = {"decisions":len(classic), "sessions":len(delta), "primary_difference":delta.mean(),
              "primary_95_interval":interval(delta.to_numpy()), "log_loss":loss.mean().to_dict(),
              "text_coverage":classic[["news_available","filing_available"]].mean().to_dict()}
    (out/"result.json").write_text(json.dumps(result, indent=2))
    rows = []
    for name, g in ledgers.groupby("model"):
        for period, segment in [("all",g),*g.groupby(g.decision.dt.year)]:
            values = np.r_[1, (1+segment.net_return).cumprod()]
            rows.append([name,period,values[-1]-1,(1-values/np.maximum.accumulate(values)).max(),
                         segment.exposure.mean(),segment.fees.sum(),segment.turnover.sum()])
    pd.DataFrame(rows, columns=["model","period","return","drawdown","exposure","fees","turnover"]).to_csv(out/"comparison.csv",index=False)
    fig, ax = plt.subplots(figsize=(9,4.5))
    for name in ["HMM gate","Classic MoE","Classic MoE every five sessions","MoE + offline RL","SPY buy and hold"]:
        g = ledgers[ledgers.model==name]
        ax.plot(g.mark, g.equity, label=name)
    ax.set_ylabel("Simulated portfolio value (USD)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/"comparison.png",dpi=155)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT/"data/raw/ceci2-v1")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.cache,args.output)
