"""Replay the frozen classic mixture and offline RL study on its 2022 extension."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from ceci2_demo.moe_v2.study import KEY, PROBS, fit_mixture, interval, offline_policy
from ceci2_demo.evaluate import replay
from ceci2_demo.panel import NUMERIC, TEXT, panel
from ceci2_demo.train import observations

ROOT = Path(__file__).resolve().parents[2]
YEAR = 2022


def evaluate(cache, out):
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__).with_name("protocol.json"), out / "protocol.json")
    torch.set_num_threads(1)
    frames, _ = panel(cache, 5)
    _, raw = observations(frames)
    rows = raw.rename_axis("decision").reset_index()
    numerical = NUMERIC + list(rows.filter(like="jkp_"))
    forecasts = {}
    for name, features in [
        ("Classic MoE", numerical + TEXT),
        ("MoE without text", numerical),
    ]:
        chosen = fit_mixture(rows, YEAR, features, out, name)
        forecasts[name] = chosen[chosen.decision.dt.year == YEAR].sort_values(KEY)
        if name == "Classic MoE":
            train_policy = chosen[chosen.decision.dt.year == YEAR - 1]
            test = forecasts[name]
            targets, actions = offline_policy(train_policy, test, out, YEAR)
    test = forecasts["Classic MoE"]
    edge = {}
    for role, part in [
        ("expert", rows[rows.endpoint.dt.year < YEAR - 2]),
        (
            "calibration",
            rows[
                (rows.endpoint.dt.year == YEAR - 2)
                & (rows.decision.dt.year == YEAR - 2)
            ],
        ),
        ("policy", train_policy),
    ]:
        edge[role] = str(part.endpoint.max().date())
    if not (
        edge["expert"]
        < "2020-01-01"
        <= edge["calibration"]
        < "2021-01-01"
        <= edge["policy"]
        < str(test.decision.min().date())
    ):
        raise ValueError("Observed outcomes overlap the next annual stage")

    archive = pd.read_csv(
        ROOT / "data/ceci2-demo/robustness/signals.csv.gz", parse_dates=["decision"]
    )
    for name in ["Uniform vote", "HMM gate"]:
        observed = archive[
            (archive.scenario == "original")
            & (archive.decision.dt.year == YEAR)
            & (archive.model == name)
        ].sort_values(KEY)
        if not observed.set_index(KEY).index.equals(test.set_index(KEY).index):
            raise ValueError(f"Dates or instrument coverage differ for {name}")
        forecasts[name] = test.drop(columns=PROBS).merge(
            observed[KEY + PROBS], on=KEY, validate="one_to_one"
        )
    losses, ledgers = {}, {}
    for name, predicted in forecasts.items():
        probabilities = predicted[PROBS].to_numpy()
        nll = -np.log(
            probabilities[np.arange(len(predicted)), predicted.label.to_numpy()].clip(
                1e-15
            )
        )
        losses[name] = pd.Series(nll, index=predicted.decision).groupby(level=0).mean()
        ledgers[name] = replay(predicted, name)
        if name in ["Classic MoE", "HMM gate"]:
            weekly = name + " every five sessions"
            ledgers[weekly] = replay(predicted, weekly, cadence=5)
    ledgers["MoE + offline RL"] = replay(test, "MoE + offline RL", targets=targets)
    for source, label in [
        ("MoE + offline RL", "RL"),
        ("Classic MoE", "Classic MoE"),
        ("HMM gate", "HMM gate"),
    ]:
        name = "SPY at " + label + " exposure"
        exposure = ledgers[source].set_index("decision").exposure
        ledgers[name] = replay(test, name, exposure=exposure)
    for name in ["Cash", "SPY buy and hold", "Equal weight"]:
        ledgers[name] = replay(test, name)

    daily = pd.concat(ledgers.values(), ignore_index=True)
    expected = pd.read_csv(ROOT / "data/ceci2-demo/moe-v2/daily.csv.gz", nrows=0)
    if not daily.columns.equals(expected.columns):
        raise ValueError("Extension ledger differs from the parent contract")
    daily.to_csv(out / "daily.csv.gz", index=False)
    actions.rename_axis("decision").to_csv(out / "actions.csv")
    pd.concat([frame.assign(model=name) for name, frame in forecasts.items()])[
        KEY + ["model", "label"] + PROBS + ["news_available", "filing_available"]
    ].to_csv(out / "forecasts.csv.gz", index=False)
    loss = pd.DataFrame(losses)
    loss.to_csv(out / "daily-loss.csv")
    delta = loss["HMM gate"] - loss["Classic MoE"]
    results = dict(
        year=YEAR,
        decisions=len(test),
        sessions=len(delta),
        first_decision=str(test.decision.min().date()),
        last_decision=str(test.decision.max().date()),
        maximum_label_endpoints=edge,
        primary_difference=delta.mean(),
        primary_95_interval=interval(delta.to_numpy()),
        log_loss=loss.mean().to_dict(),
        text_coverage=test[["news_available", "filing_available"]].mean().to_dict(),
        policy_action_counts=actions.map({0: "cash", 1: "SPY", 2: "classic basket"})
        .value_counts()
        .to_dict(),
    )
    (out / "result.json").write_text(json.dumps(results, indent=2) + "\n")
    comparison = []
    for name, ledger in ledgers.items():
        returns = ledger.net_return
        equity = np.r_[1, (1 + returns).cumprod()]
        comparison.append(
            dict(
                model=name,
                period=YEAR,
                net_return=equity[-1] - 1,
                drawdown=(1 - equity / np.maximum.accumulate(equity)).max(),
                exposure=ledger.exposure.mean(),
                fees=ledger.fees.sum(),
                turnover=ledger.turnover.sum(),
                sharpe=np.sqrt(252) * returns.mean() / returns.std()
                if returns.std()
                else 0,
            )
        )
    pd.DataFrame(comparison).to_csv(out / "comparison.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/raw/ceci2-v1")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    evaluate(args.cache, args.output)
