"""Classifier statistics from saved forecasts. No model is fitted here.

Three separate bodies of evidence, never pooled:

1. ``history``      out-of-fold forecasts in the sealed adaptive release,
                    2 January 2024 to 17 September 2026 (previously inspected).
2. ``prospective``  the 88 forecast rows the scheduled collector recorded in
                    October 2026 and later matched to outcomes.
3. ``replay``       a simulated continuous replay after the release date
                    (written by ``review_v1.replay``; scored there).

Outputs go to ``data/review-v1/classifier/``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from review_v1 import receipts, registration, regime_report
from review_v1.paths import ADAPTIVE_RELEASE, CLASSES, CONTROLLER_V1, OUT, RECORDED, RESEARCH_V2, read_json, rel, write_json
from review_v1.registration import BLOCK, PERIODS, REPS, SYMBOLS
from review_v1.stats import block_effect, classification_report, holm, row_log_loss

DIR = OUT / "classifier"
MODELS = ["linear", "tree", "raw_mix", "hybrid", "prior"]
MODEL_TEXT = {
    "linear": "Logistic regression (C = 0.1) on 17 standardised price and volume features",
    "tree": "Histogram gradient-boosted trees (100 rounds, 7 leaves)",
    "raw_mix": "Equal average of the two, before calibration",
    "hybrid": "Equal average after temperature calibration: the forecast the pilot records",
    "prior": "Class-frequency prior: each class's share of the training window, same answer every day until the next refit",
}


def load_history() -> pd.DataFrame:
    """One row per instrument and decision date with a matured label."""
    pieces = []
    for symbol in SYMBOLS:
        frame = pd.read_parquet(ADAPTIVE_RELEASE / f"experiment/{symbol}_predictions.parquet")
        frame = regime_report.naive(frame)
        frame["symbol"] = symbol
        pieces.append(frame)
    data = pd.concat(pieces).rename_axis("date").reset_index()
    data = data.loc[data.label >= 0].copy()
    for c in range(3):
        data[f"raw_mix_{c}"] = (data[f"linear_{c}"] + data[f"tree_{c}"]) / 2
    return data.sort_values(["date", "symbol"]).reset_index(drop=True)


def probabilities(frame: pd.DataFrame, model: str) -> np.ndarray:
    return frame[[f"{model}_{c}" for c in range(3)]].to_numpy(float)


def metric_row(frame: pd.DataFrame, model: str) -> dict:
    report = classification_report(frame.label.to_numpy(), probabilities(frame, model))
    predicted = np.asarray(report["confusion"]).sum(axis=0)
    return dict(
        model=model,
        sessions=int(frame.date.nunique()),
        n=report["n"],
        accuracy=report["accuracy"],
        balanced_accuracy=report["balanced_accuracy"],
        macro_f1=report["macro_f1"],
        log_loss=report["log_loss"],
        brier=report["brier"],
        predicted_down=int(predicted[0]),
        predicted_flat=int(predicted[1]),
        predicted_up=int(predicted[2]),
    ), report


def daily_difference(frame: pd.DataFrame, model: str, comparator: str = "prior", endpoint: str = "log_loss") -> pd.Series:
    """Cross-sectional mean of (model minus comparator) per decision date."""
    y = frame.label.to_numpy()
    if endpoint == "log_loss":
        value = row_log_loss(y, probabilities(frame, model)) - row_log_loss(y, probabilities(frame, comparator))
    elif endpoint == "accuracy":
        value = (probabilities(frame, model).argmax(1) == y).astype(float) - (probabilities(frame, comparator).argmax(1) == y).astype(float)
    else:
        raise ValueError(endpoint)
    return pd.Series(value, index=frame.date.to_numpy()).groupby(level=0).mean()


def paired(frame: pd.DataFrame, model: str, block: int = BLOCK) -> dict:
    loss = block_effect(daily_difference(frame, model).to_numpy(), block, REPS)
    accuracy = block_effect(daily_difference(frame, model, endpoint="accuracy").to_numpy(), block, REPS)
    return dict(
        model=model,
        comparator="prior",
        sessions=loss["sessions"],
        block_length=block,
        log_loss_difference=loss["effect"],
        log_loss_ci_low=loss["ci_low"],
        log_loss_ci_high=loss["ci_high"],
        log_loss_p=loss["p_two_sided"],
        log_loss_mde80=loss["mde80_approx"],
        accuracy_difference=accuracy["effect"],
        accuracy_ci_low=accuracy["ci_low"],
        accuracy_ci_high=accuracy["ci_high"],
        accuracy_p=accuracy["p_two_sided"],
        reading="negative log-loss difference means the model beat the prior",
    )


def verdict(row: dict) -> str:
    if row["log_loss_ci_low"] > 0:
        return "worse than the prior"
    if row["log_loss_ci_high"] < 0:
        return "better than the prior"
    return "not distinguishable from the prior"


def label_shares(frame: pd.DataFrame) -> dict:
    counts = np.bincount(frame.label.to_numpy(), minlength=3)
    return {CLASSES[c]: float(counts[c] / counts.sum()) for c in range(3)}


def prior_profile(frame: pd.DataFrame) -> dict:
    p = probabilities(frame, "prior")
    picks = np.bincount(p.argmax(1), minlength=3)
    return dict(
        mean_probability={CLASSES[c]: float(p[:, c].mean()) for c in range(3)},
        predicted_class_share={CLASSES[c]: float(picks[c] / picks.sum()) for c in range(3)},
        mean_top_probability=float(p.max(axis=1).mean()),
    )


def history_report(data: pd.DataFrame) -> dict:
    by_period, class_rows, confusion_rows, paired_rows, profile = [], [], [], [], {}
    scopes = dict(PERIODS, all=(str(data.date.min().date()), str(data.date.max().date())))
    for period, (start, end) in scopes.items():
        frame = data.loc[(data.date >= start) & (data.date <= end)]
        profile[period] = dict(
            first=str(frame.date.min().date()), last=str(frame.date.max().date()), sessions=int(frame.date.nunique()), n=int(len(frame)),
            label_share=label_shares(frame), prior=prior_profile(frame),
        )
        for model in MODELS:
            row, report = metric_row(frame, model)
            by_period.append(dict(period=period, **row))
            for item in report["classes"]:
                class_rows.append(dict(period=period, model=model, **item))
            for actual in range(3):
                for predicted in range(3):
                    confusion_rows.append(dict(period=period, model=model, actual=CLASSES[actual], predicted=CLASSES[predicted], count=report["confusion"][actual][predicted]))
            if model != "prior":
                paired_rows.append(dict(period=period, **paired(frame, model)))
    paired_frame = pd.DataFrame(paired_rows)
    paired_frame["verdict"] = [verdict(r) for r in paired_rows]
    return dict(
        by_period=pd.DataFrame(by_period),
        class_report=pd.DataFrame(class_rows),
        confusion=pd.DataFrame(confusion_rows),
        paired=paired_frame,
        profile=profile,
    )


def cross_check(by_period: pd.DataFrame) -> dict:
    """Same numbers as the independently generated research-v2 table?"""
    reference = pd.read_csv(RESEARCH_V2 / "tables/diag_forecast_summary.csv")
    merged = by_period.merge(reference, on=["period", "model"], suffixes=("", "_reference"))
    columns = ["accuracy", "balanced_accuracy", "macro_f1", "log_loss", "brier"]
    difference = max(float((merged[c] - merged[f"{c}_reference"]).abs().max()) for c in columns)
    counts = bool((merged.n == merged.n_reference).all() and (merged.sessions == merged.sessions_reference).all())
    return dict(
        reference=rel(RESEARCH_V2 / "tables/diag_forecast_summary.csv"),
        rows_compared=int(len(merged)),
        max_abs_metric_difference=difference,
        counts_equal=counts,
        passed=bool(len(merged) == 15 and counts and difference < 1e-12),
    )


def regime_report_rows(data: pd.DataFrame) -> dict:
    """Same forecasts split by the market regime known at each decision close."""
    posterior = regime_report.load_posterior("history")
    frame = data.merge(posterior[["p_turbulent", "state"]], left_on="date", right_index=True, how="left", validate="many_to_one")
    if frame.state.isna().any():
        raise ValueError("A forecast date has no regime probability")
    rows, contrasts, shares = [], [], {}
    for state, g in frame.groupby("state"):
        shares[state] = dict(sessions=int(g.date.nunique()), n=int(len(g)), label_share=label_shares(g), prior=prior_profile(g))
        for model in MODELS:
            row, _ = metric_row(g, model)
            rows.append(dict(regime=state, **row))
        contrasts.append(dict(regime=state, **paired(g, "hybrid")))
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame["holm_p"] = holm(contrast_frame.log_loss_p.to_numpy())
    contrast_frame["verdict"] = [verdict(r) for r in contrasts]
    contrast_frame["note"] = "Regime dates are not contiguous: a block can join sessions from different spells."
    return dict(by_regime=pd.DataFrame(rows), contrasts=contrast_frame, shares=shares)


def by_symbol(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol, g in data.groupby("symbol"):
        hybrid, _ = metric_row(g, "hybrid")
        prior, _ = metric_row(g, "prior")
        rows.append(
            dict(
                symbol=symbol, n=hybrid["n"],
                hybrid_log_loss=hybrid["log_loss"], prior_log_loss=prior["log_loss"], log_loss_difference=hybrid["log_loss"] - prior["log_loss"],
                hybrid_accuracy=hybrid["accuracy"], prior_accuracy=prior["accuracy"],
                share_down=float((g.label == 0).mean()), share_flat=float((g.label == 1).mean()), share_up=float((g.label == 2).mean()),
            )
        )
    return pd.DataFrame(rows).sort_values("log_loss_difference")


def prospective_report() -> dict:
    """Outcome rows recorded by the scheduled collector; small and dependent."""
    rows = receipts.recorded_outcomes()
    if rows.empty:
        return dict(available=False)
    y = rows.label.to_numpy(int)
    hybrid = np.asarray(rows.hybrid.tolist(), float)
    prior = np.asarray(rows.prior.tolist(), float)
    out = dict(available=True, provenance=RECORDED, n=int(len(rows)), decision_sessions=sorted(rows.decision_session.unique().tolist()), symbols=int(rows.symbol.nunique()))
    for name, p in [("hybrid", hybrid), ("prior", prior)]:
        report = classification_report(y, p)
        out[name] = {k: report[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "log_loss", "brier", "confusion"]}
    out["label_share"] = {CLASSES[c]: float(np.mean(y == c)) for c in range(3)}
    difference = pd.Series(row_log_loss(y, hybrid) - row_log_loss(y, prior), index=rows.decision_session.to_numpy()).groupby(level=0).mean()
    out["daily_log_loss_difference"] = {k: float(v) for k, v in difference.items()}
    recorded = {m["model"]: m for m in receipts.outcome_report().get("metrics", [])}
    if recorded:
        flat = {f"{m}.{k}": abs(out[m][k] - recorded[m][k]) for m in ["hybrid", "prior"] for k in ["accuracy", "log_loss", "brier"]}
        out["matches_recorded_report"] = dict(compared=sorted(flat), max_abs_difference=max(flat.values()))
    out["reading"] = (
        "Four decision dates from two weeks: effectively four dependent observations, far too few for any "
        "inference. The pilot's own closeout classed this collection as incomplete."
    )
    return out


def earlier_studies() -> dict:
    """The two results quoted in the progress summary, read from their own outputs."""
    predictive = pd.read_csv(RESEARCH_V2 / "tables/exp_predictive.csv")
    base = predictive.loc[predictive.variant == "base"].set_index("model")
    contrasts = pd.read_csv(RESEARCH_V2 / "tables/exp_contrasts.csv")
    calibrated = contrasts.loc[(contrasts.id == "R4-calibrated") & (contrasts.block == 10)].iloc[0]
    study = dict(
        source=rel(RESEARCH_V2 / "tables/exp_predictive.csv"),
        name="Controlled 848-trial study (research-v2)",
        dates=int(base.loc["prior", "n_dates"]), observations=int(base.loc["prior", "n_rows"]),
        window="four 63-session blocks in 2024 and 2025",
        prior=dict(log_loss=float(base.loc["prior", "log_loss"]), accuracy=float(base.loc["prior", "accuracy"])),
        mixture=dict(log_loss=float(base.loc["calibrated", "log_loss"]), accuracy=float(base.loc["calibrated", "accuracy"])),
        difference=dict(effect=float(calibrated.effect), ci_low=float(calibrated.ci_low), ci_high=float(calibrated.ci_high), p_holm=float(calibrated.p_holm), block=10),
    )
    screen = read_json(CONTROLLER_V1 / "20261030-screen-v1/summary.json")
    metrics = {m["model"]: m for m in screen["metrics"] if m["period"] == "2025-all"}
    contrast = {c["id"]: c for c in screen["contrasts"] if c.get("block_length", 10) == 10}
    controller = dict(
        source=rel(CONTROLLER_V1 / "20261030-screen-v1/summary.json"),
        name="Controller screen of 30 October 2026",
        dates=int(metrics["prior"]["n_dates"]), observations=int(metrics["prior"]["n_symbol_dates"]), window="calendar 2025",
        models={k: dict(log_loss=float(v["log_loss"]), accuracy=float(v["accuracy"])) for k, v in metrics.items()},
        versus_prior={
            name: dict(effect=float(contrast[f"{name}-vs-prior"]["effect"]), ci_low=float(contrast[f"{name}-vs-prior"]["ci_low"]),
                       ci_high=float(contrast[f"{name}-vs-prior"]["ci_high"]), p_holm=float(contrast[f"{name}-vs-prior"]["p_holm"]))
            for name in ["hedge", "stack"]
        },
    )
    return dict(study_848=study, controller_screen=controller)


def build() -> dict:
    reg = registration.require()
    data = load_history()
    history = history_report(data)
    regimes = regime_report_rows(data)
    DIR.mkdir(parents=True, exist_ok=True)
    history["by_period"].to_csv(DIR / "by_period.csv", index=False)
    history["class_report"].to_csv(DIR / "class_report.csv", index=False)
    history["confusion"].to_csv(DIR / "confusion.csv", index=False)
    history["paired"].to_csv(DIR / "paired_vs_prior.csv", index=False)
    regimes["by_regime"].to_csv(DIR / "by_regime.csv", index=False)
    regimes["contrasts"].to_csv(DIR / "regime_contrasts.csv", index=False)
    by_symbol(data).to_csv(DIR / "by_symbol.csv", index=False)
    summary = dict(
        registration_sha256=reg["sha256"],
        status="exploratory: every history date was inspected before this review",
        label_contract=reg["registration"]["label_contract"],
        models=MODEL_TEXT,
        history=dict(source=rel(ADAPTIVE_RELEASE / "experiment"), first=str(data.date.min().date()), last=str(data.date.max().date()),
                     sessions=int(data.date.nunique()), n=int(len(data)), symbols=int(data.symbol.nunique())),
        profile=history["profile"],
        regime_shares=regimes["shares"],
        cross_check=cross_check(history["by_period"]),
        prospective=prospective_report(),
        earlier_studies=earlier_studies(),
        inference=f"{REPS} circular date blocks of length {BLOCK}; all instruments on a date stay together; percentile interval",
    )
    write_json(DIR / "summary.json", summary)
    return summary


if __name__ == "__main__":
    out = build()
    print(out["cross_check"])
    print(pd.read_csv(DIR / "by_period.csv").round(4).to_string())
    print(pd.read_csv(DIR / "paired_vs_prior.csv").round(4).to_string())
    print(pd.read_csv(DIR / "by_regime.csv").round(4).to_string())
    print(pd.read_csv(DIR / "regime_contrasts.csv").round(4).to_string())
    print(out["prospective"])
    print(out["earlier_studies"])
