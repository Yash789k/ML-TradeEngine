"""Evaluate the registered controller stress study on cached inputs only."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize, minimize_scalar
from scipy.special import softmax
from sklearn.metrics import log_loss

from src.models.gating import GatingModel
from .evaluate import replay
from .experts import SequenceExpert, predict_experts
from .panel import NUMERIC, TEXT, panel
from .train import market_states, observations

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "data/ceci2-demo/five-session"
OUT = ROOT / "data/ceci2-demo/robustness"
LANGUAGE = {
    "Qwen statement": ["qdown", "qflat", "qup"],
    "OpenAI asset digest": ["mdown", "mflat", "mup"],
    "OpenAI Decisions": ["odown", "oflat", "oup"],
    "Granite MoE statement": ["gdown", "gflat", "gup"],
}
METHODS = ["Adaptive router", "HMM gate", "Uniform vote"]


def existing_expert(folder, name, values):
    width = len(NUMERIC)
    if (folder / f"{name}.pt").exists():
        net = SequenceExpert(name, width)
        net.load_state_dict(
            torch.load(folder / f"{name}.pt", map_location="cpu", weights_only=True)
        )
        net.eval()
        with torch.inference_mode():
            x = torch.from_numpy(values[:, :, :width].copy())
            return torch.cat([net(batch).softmax(-1) for batch in x.split(512)]).numpy()
    if name == "FinBERT logistic":
        features = values[:, -1, width : width + len(TEXT)]
    elif name == "JKP factors":
        features = values[:, -1, width + len(TEXT) :]
    else:
        features = values[:, -1, :width]
    return joblib.load(folder / f"{name}.joblib").predict_proba(features)


def probabilities(windows, rows, year, output):
    train = (rows.index.year < year - 2) & (rows.endpoint.dt.year < year - 2)
    eligible = np.zeros(len(rows), dtype=bool)
    for part in (year - 2, year - 1, year):
        dates = rows.index[rows.index.year == part].unique().sort_values()
        eligible |= (
            (rows.index.year == part)
            & (rows.index >= dates[5])
            & (rows.endpoint.dt.year == part)
        )
    selected = rows.loc[eligible].copy()
    folder = STUDY / f"models-{year}"
    if folder.is_dir():
        info = json.loads((folder / "manifest.json").read_text())
        params = np.load(folder / "example.npz")
        x = ((windows[eligible] - params["center"]) / params["scale"]).clip(-10, 10)
        raw = {
            name: existing_expert(folder, name, x)
            for name in info["experts"]
            if name not in LANGUAGE
        }
        temperatures = info["temperature"]
    else:
        center = windows[train, -1].mean(0)
        scale = windows[train, -1].std(0).clip(1e-6)
        x = ((windows - center) / scale).clip(-10, 10)
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            raw = predict_experts(x, rows.label.values, train, eligible, Path(tmp))
        temperatures = {}
    raw.update(
        {name: selected[columns].to_numpy() for name, columns in LANGUAGE.items()}
    )
    cal = selected.index.year == year - 2
    calibrated = []
    for name, p in raw.items():
        logits = np.log(p.clip(1e-12, 1))
        if name not in temperatures:

            def objective(temp):
                return log_loss(
                    selected.label[cal],
                    softmax(logits[cal] / temp, axis=1),
                    labels=[0, 1, 2],
                )

            temperatures[name] = minimize_scalar(
                objective, bounds=(0.5, 5), method="bounded"
            ).x
        calibrated.append(softmax(logits / temperatures[name], axis=1))
    return selected, list(raw), np.stack(calibrated, axis=1), temperatures


def simplex_fit(target_prob, importance):
    count = target_prob.shape[1]
    uniform = np.full(count, 1 / count)
    sample_weight = importance / importance.sum()

    def objective(weights):
        mixture = (target_prob @ weights).clip(1e-12)
        delta = weights - uniform
        return (
            -sample_weight @ np.log(mixture) + 0.02 * (delta @ delta),
            -(sample_weight / mixture) @ target_prob + 0.04 * delta,
        )

    fitted = minimize(
        objective,
        uniform,
        jac=True,
        method="SLSQP",
        bounds=[(0, 1)] * count,
        constraints={
            "type": "eq",
            "fun": lambda w: w.sum() - 1,
            "jac": lambda w: np.ones_like(w),
        },
        options={"maxiter": 200, "ftol": 1e-10},
    )
    if not fitted.success:
        raise RuntimeError(f"Simplex fit failed: {fitted.message}")
    result = fitted.x.clip(0, 1)
    return result / result.sum()


def target_probabilities(probs, labels):
    return np.take_along_axis(probs, labels[:, None, None], axis=-1).squeeze(-1)


def original_context(frame, probabilities, states):
    correct = probabilities.argmax(axis=-1) == frame.label.values[:, None]
    correct = np.where(
        np.isfinite(probabilities).all(axis=-1), correct.astype(float), np.nan
    )
    mem = (
        pd.DataFrame(correct, index=frame.endpoint)
        .groupby(level=0)
        .mean()
        .rolling(63, min_periods=1)
        .mean()
    )
    aligned = mem.reindex(frame.index - pd.Timedelta(days=1), method="ffill").fillna(
        1 / 3
    )
    ctx = frame[["vol", "range", "drawdown", "market"]].copy()
    ctx[[f"accuracy_{j}" for j in range(probabilities.shape[1])]] = aligned.to_numpy()
    ctx[["fed_positive", "headline_tone", "annual_report_tone"]] = frame[
        ["positive", "news_tone", "filing_tone"]
    ].to_numpy()
    ctx[["bear", "middle", "bull"]] = states
    return ctx


def router_weights(frame, probs, states, test, prior_weights, state_weights):
    loss = -np.log(target_probabilities(probs, frame.label.to_numpy()).clip(1e-12))
    # Regime attribution belongs to the decision that generated the forecast.
    # The feedback index belongs to the later opening that resolves its label.
    history = frame.index.year >= frame.index[test].year.min() - 1
    dates = frame.loc[history, "endpoint"].to_numpy()
    daily = pd.DataFrame(loss[history], index=dates).groupby(level=0).mean()
    regime_at_issue = pd.DataFrame(states[history], index=dates).groupby(level=0).mean()
    days = frame.index[test]
    feedback_cutoff = days - pd.Timedelta(days=1)
    global_loss = (
        daily.rolling(63, min_periods=1)
        .mean()
        .reindex(feedback_cutoff, method="ffill")
        .fillna(np.log(3))
        .to_numpy()
    )
    posterior = states[test]
    recent = np.zeros_like(global_loss)
    for k in range(3):
        state = regime_at_issue.iloc[:, k]
        weighted = daily.mul(state, axis=0).rolling(63, min_periods=1).sum()
        mass = daily.notna().mul(state, axis=0).rolling(63, min_periods=1).sum()
        sums = weighted.reindex(feedback_cutoff, method="ffill").fillna(0).to_numpy()
        masses = mass.reindex(feedback_cutoff, method="ffill").fillna(0).to_numpy()
        recent += posterior[:, k, None] * (sums + 10 * global_loss) / (masses + 10)
    prior = (0.5 * prior_weights + 0.5 * posterior @ state_weights).clip(1e-12)
    return 0.9 * softmax(np.log(prior) - 4 * recent, axis=1) + 0.1 / probs.shape[1]


def perturb(original, names, frame, year, scenario):
    probs = original.copy()
    shock = np.flatnonzero((frame.index.year == year) & (frame.index.month == 7))
    if scenario == "language_outage":
        channels = [names.index(name) for name in [*LANGUAGE, "FinBERT logistic"]]
        probs[np.ix_(shock, channels)] = np.nan
    elif scenario == "reversed_tree":
        channels = [names.index(name) for name in ("XGBoost", "LightGBM")]
        probs[np.ix_(shock, channels)] = probs[np.ix_(shock, channels)][:, :, [2, 1, 0]]
    return probs


def run_fold(
    windows, rows, market, year, scenarios, weight_rows, signal_rows, states_map, output
):
    print(f"Replaying {year}", flush=True)
    frame, names, original, temps = probabilities(windows, rows, year, output)
    gate = frame.index.year == year - 1
    test = frame.index.year == year
    fit = (rows.index.year < year - 2) & (rows.endpoint.dt.year < year - 2)
    calibration = frame.index.year == year - 2
    boundaries = {}
    for role, last, first in (
        ("expert", rows.loc[fit, "endpoint"].max(), frame.index[calibration].min()),
        (
            "calibration",
            frame.loc[calibration, "endpoint"].max(),
            frame.index[gate].min(),
        ),
        ("gate", frame.loc[gate, "endpoint"].max(), frame.index[test].min()),
    ):
        if not last < first:
            raise ValueError(
                f"{role} outcomes overlap next evaluation stage for {year}"
            )
        boundaries[role] = {
            "last_outcome": last.date().isoformat(),
            "next_stage": first.date().isoformat(),
        }
    regimes = market_states(market, year - 1)
    states = regimes.reindex(frame.index).to_numpy()
    regime = regimes.idxmax(axis=1)
    transition = regime.ne(regime.shift()).rolling(5, min_periods=1).max().astype(bool)
    full_test = frame.loc[test].copy()
    full_test["regime"] = regime.reindex(full_test.index).values
    full_test["transition"] = transition.reindex(full_test.index).values
    state_rows = full_test[["regime", "transition"]].groupby(level=0).first()
    states_map.append(state_rows)
    labels = frame.label.to_numpy()
    mature_probs = target_probabilities(original[gate], labels[gate])
    prior = simplex_fit(mature_probs, np.ones(gate.sum()))
    state_weights = np.stack(
        [simplex_fit(mature_probs, states[gate, i]) for i in range(3)]
    )
    original_gate_path = STUDY / f"models-{year}/gate.joblib"
    if original_gate_path.exists():
        original_gate = GatingModel.load(str(original_gate_path))
    else:
        correct = original[gate].argmax(-1) == labels[gate, None]
        original_gate = GatingModel(n_experts=len(names)).fit(
            original_context(frame, original, states)[gate], correct
        )
    observed = router_weights(frame, original, states, test, prior, state_weights)
    counterfactual = frame.copy()
    cutoff = pd.Timestamp(year, 7, 1)
    later = counterfactual.endpoint >= cutoff
    counterfactual.loc[later, "label"] = (counterfactual.loc[later, "label"] + 1) % 3
    altered = router_weights(
        counterfactual, original, states, test, prior, state_weights
    )
    earlier = frame.index[test] <= cutoff
    if not np.array_equal(observed[earlier], altered[earlier]):
        raise ValueError("Changing future outcomes altered earlier controller weights")
    for scenario in scenarios:
        probs = perturb(original, names, frame, year, scenario)
        current = probs[test]
        valid = np.isfinite(current).all(-1)
        choices = [
            router_weights(frame, probs, states, test, prior, state_weights),
            original_gate.predict_weights(original_context(frame, probs, states)[test]),
            np.ones(valid.shape),
        ]
        for method, raw in zip(METHODS, choices):
            w = np.where(valid, raw, 0.0)
            w /= w.sum(1, keepdims=True)
            mixed = np.einsum("ij,ijk->ik", w, np.nan_to_num(current))
            if not np.isfinite(mixed).all() or not np.allclose(mixed.sum(1), 1):
                raise ValueError("Invalid controller output")
            predictions = full_test[
                [
                    "symbol",
                    "label",
                    "entry",
                    "exit",
                    "vol",
                    "mark",
                    "regime",
                    "transition",
                ]
            ].copy()
            predictions[["down", "flat", "up"]] = mixed
            predictions["model"] = method
            predictions["scenario"] = scenario
            signal_rows.append(predictions.rename_axis("decision").reset_index())
            avg = (
                pd.DataFrame(w, index=full_test.index, columns=names)
                .groupby(level=0)
                .mean()
            )
            weight_rows.append(
                avg.rename_axis("decision")
                .reset_index()
                .melt(id_vars="decision", var_name="expert", value_name="weight")
                .assign(model=method, scenario=scenario)
            )
    return {
        "year": year,
        "experts": names,
        "global_simplex": prior.tolist(),
        "regime_simplex": state_weights.tolist(),
        "temperature": temps,
        "stage_boundaries": boundaries,
        "future_label_perturbation": {
            "cutoff": cutoff.date().isoformat(),
            "unchanged_weight_rows": int(earlier.sum()),
        },
        "latest_gate_outcome": str(frame.loc[gate, "endpoint"].max().date()),
        "first_evaluation_date": str(frame.index[test].min().date()),
    }


def portfolio_summary(block, period):
    r = block.net_return.to_numpy()
    path = np.r_[1.0, np.cumprod(1 + r)]
    return dict(
        period=period,
        sessions=len(r),
        log_loss=block.log_loss.mean(),
        net_return=path[-1] - 1,
        max_drawdown=(1 - path / np.maximum.accumulate(path)).max(),
        sharpe=r.mean() / r.std(ddof=1) * np.sqrt(252) if r.std(ddof=1) else 0,
        mean_exposure=block.exposure.mean(),
    )


def uncertainty(daily):
    results = []
    for scenario, sub in daily[
        daily.model.isin(["Adaptive router", "Uniform vote"])
    ].groupby("scenario"):
        a = sub[sub.model == "Adaptive router"].set_index("decision").sort_index()
        b = sub[sub.model == "Uniform vote"].set_index("decision").loc[a.index]
        delta = (a.log_loss - b.log_loss).to_numpy()
        returns = np.stack([a.net_return, b.net_return], axis=1)
        rg = np.random.default_rng(28)
        blocks = []
        for y in a.index.year.unique():
            idx = np.flatnonzero(a.index.year == y)
            starts = rg.integers(len(idx), size=(1000, (len(idx) + 19) // 20))
            blocks.append(
                idx[
                    ((starts[:, :, None] + np.arange(20)) % len(idx)).reshape(1000, -1)[
                        :, : len(idx)
                    ]
                ]
            )
        idx = np.concatenate(blocks, axis=1)
        rr = np.prod(1 + returns[idx], axis=1)
        for name, obs, draws in (
            ("log_loss", delta.mean(), delta[idx].mean(1)),
            (
                "net_return",
                a.equity.iloc[-1] / 10000 - b.equity.iloc[-1] / 10000,
                rr[:, 0] - rr[:, 1],
            ),
        ):
            low, high = np.quantile(draws, [0.025, 0.975])
            results.append(
                dict(
                    scenario=scenario, endpoint=name, difference=obs, low=low, high=high
                )
            )
    return pd.DataFrame(results)


def plot_controller(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ledger = pd.read_csv(output / "daily.csv.gz", parse_dates=["mark"])
    ledger = ledger.loc[ledger.scenario == "original"].pivot(
        index="mark", columns="model", values="equity"
    )
    weights = pd.read_csv(output / "weights.csv.gz", parse_dates=["decision"])
    routed = weights[
        (weights.model == "Adaptive router") & (weights.scenario == "original")
    ]
    routed = routed.pivot(index="decision", columns="expert", values="weight")
    leaders = routed.mean().nlargest(4).index
    plot_weights = routed[leaders].copy()
    plot_weights["Remaining experts"] = 1 - plot_weights.sum(axis=1)
    fig, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 6.8), layout="constrained"
    )
    for model, color in zip(
        [*METHODS, "SPY buy and hold", "Equal weight"],
        ["#204c74", "#a75c2e", "#559584", "#797e84", "#b4914a"],
    ):
        top.plot(ledger.index, ledger[model], color=color, label=model, linewidth=1.5)
    top.set(title="Simulated portfolio value after costs", ylabel="US dollars")
    top.legend(ncols=3, frameon=False, fontsize=8)
    bottom.stackplot(
        plot_weights.index,
        plot_weights.T,
        labels=plot_weights.columns,
        colors=["#204c74", "#397a92", "#61a296", "#d4a15b", "#dedcd6"],
    )
    bottom.set(
        title="Adaptive controller weights averaged across instruments",
        ylabel="Share of vote",
        ylim=(0, 1),
    )
    bottom.legend(
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        frameon=False,
        fontsize=8,
    )
    for ax in (top, bottom):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "controller.png", dpi=140)
    plt.close(fig)


def export_site(output):
    """Package price paths and robustness metrics without browser-side recomputation."""
    daily = pd.read_csv(output / "daily.csv.gz")
    comparison = pd.read_csv(output / "comparison.csv")
    conditional = pd.read_csv(output / "conditional.csv")
    intervals = pd.read_csv(output / "uncertainty.csv")
    days = daily.drop_duplicates("decision").sort_values("decision")
    bundle = {
        "decision": days.decision.tolist(),
        "mark": days.mark.tolist(),
        "regime": days.regime.tolist(),
        "transition": days.transition.tolist(),
        "scenarios": {},
        "summary": json.loads(comparison.to_json(orient="records")),
        "conditional": json.loads(conditional.to_json(orient="records")),
        "uncertainty": json.loads(intervals.to_json(orient="records")),
    }
    for (scenario, model), group in daily.groupby(["scenario", "model"]):
        group = group.set_index("decision").loc[days.decision]
        bundle["scenarios"].setdefault(scenario, {})[model] = group.equity.round(
            2
        ).tolist()
    (output / "site.json").write_text(
        json.dumps(bundle, separators=(",", ":"), allow_nan=False)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/raw/ceci2-v1")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Download public prices and JKP factors; read all language decisions from repository evidence",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="An empty directory for a new study"
    )
    args = parser.parse_args()
    output = args.output
    if output.exists() and any(output.iterdir()):
        parser.error("Choose an empty output directory to preserve completed studies")
    output.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        from .download import fetch

        args.cache.mkdir(parents=True, exist_ok=True)
        receipts = Path(__file__).with_name("receipts")
        for name in ("statements.json", "company-scores.csv", "finbert-scores.csv"):
            if not (args.cache / name).exists():
                shutil.copyfile(receipts / name, args.cache / name)
        fetch(args.cache)
    protocol = json.loads((OUT / "protocol.json").read_text())
    required = [
        "statements.json",
        "company-scores.csv",
        "finbert-scores.csv",
        "jkp.zip",
    ]
    from .download import SYMBOLS

    required.extend(f"{symbol}.json" for symbol in SYMBOLS)
    if not all(
        (args.cache / name).is_file()
        for name in required
        if name != "jkp.zip" or not args.prepare
    ):
        raise ValueError(
            "Populate the local CECI2 historical cache with the main study first"
        )
    torch.set_num_threads(1)
    statements = json.loads((args.cache / "statements.json").read_text())
    sentiment = pd.read_csv(args.cache / "finbert-scores.csv")
    if sentiment.url.tolist() != [
        s["url"] for s in statements
    ] or sentiment.text.tolist() != [s["text"] for s in statements]:
        raise ValueError("Cached text scores do not match the recorded corpus")
    shutil.copyfile(OUT / "protocol.json", output / "protocol.json")
    frames, _ = panel(args.cache)
    windows, rows = observations(frames)
    weight_rows, signal_rows, state_rows = [], [], []
    fits = [
        run_fold(
            windows,
            rows,
            frames["SPY"],
            y,
            protocol["scenarios"],
            weight_rows,
            signal_rows,
            state_rows,
            output,
        )
        for y in range(2022, 2026)
    ]
    (output / "fits.json").write_text(json.dumps(fits, indent=2) + "\n")
    pd.concat(weight_rows).to_csv(output / "weights.csv.gz", index=False)
    signals = pd.concat(signal_rows, ignore_index=True)
    signals[["scenario", "model", "decision", "symbol", "down", "flat", "up"]].to_csv(
        output / "signals.csv.gz", index=False
    )
    states = pd.concat(state_rows).rename_axis("decision").reset_index()
    ledgers = []
    for (scenario, model), group in signals.groupby(["scenario", "model"]):
        probs = group[["down", "flat", "up"]].to_numpy()
        losses = -np.log(
            probs[np.arange(len(group)), group.label.to_numpy()].clip(1e-12)
        )
        grouped = (
            group.assign(log_loss=losses)
            .groupby("decision")
            .log_loss.mean()
            .reset_index()
        )
        ledgers.append(
            replay(group, model).merge(grouped, on="decision").assign(scenario=scenario)
        )
    base = signals[(signals.model == "Uniform vote") & (signals.scenario == "original")]
    for model in ("Cash", "SPY buy and hold", "Equal weight"):
        ledgers.append(replay(base, model).assign(scenario="original", log_loss=np.nan))
    daily = pd.concat(ledgers, ignore_index=True).merge(
        states, on="decision", validate="many_to_one"
    )
    daily.to_csv(output / "daily.csv.gz", index=False)
    summary, groups = [], []
    for (sc, model), block in daily.groupby(["scenario", "model"]):
        slices = [("all", block)] + [
            (str(y), g) for y, g in block.groupby(block.decision.dt.year)
        ]
        summary.extend(
            dict(scenario=sc, model=model, **portfolio_summary(g, period))
            for period, g in slices
        )
        regimes = [(f"state-{k}", g) for k, g in block.groupby("regime")]
        for name, g in regimes + [
            ("transition", block[block.transition]),
            ("stable", block[~block.transition]),
            ("july", block[block.decision.dt.month == 7]),
        ]:
            groups.append(
                dict(
                    scenario=sc,
                    model=model,
                    group=name,
                    sessions=len(g),
                    log_loss=g.log_loss.mean(),
                    mean_net_daily_return=g.net_return.mean(),
                    daily_return_p05=g.net_return.quantile(0.05),
                    mean_exposure=g.exposure.mean(),
                )
            )
    comparison = pd.DataFrame(summary)
    comparison.to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(groups).to_csv(output / "conditional.csv", index=False)
    uncertainty(daily).to_csv(output / "uncertainty.csv", index=False)
    export_site(output)
    plot_controller(output)
    print(comparison.query("period == 'all'").to_string(index=False))


if __name__ == "__main__":
    main()
