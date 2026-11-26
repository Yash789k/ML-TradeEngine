"""Repository audit for the progress review: what works, what is unproven, what is missing.

Each row names its evidence and a way to check it. Statuses:

* ``works``        the component runs and its output was verified in this review;
* ``unvalidated``  the code exists, but what it is meant to deliver has not been shown;
* ``diagnostic``   built and checked, but it does not feed any recorded decision;
* ``missing``      specified in the FYP proposal and not in the repository.

Numbers are not typed here. ``rows()`` fills them from the stage outputs so the
audit, the dashboard and the report cannot drift apart.
"""

from __future__ import annotations

import pandas as pd

from review_v1.paths import OUT, read_json

STATUS_TEXT = {
    "works": "Works (verified)",
    "unvalidated": "Exists, not validated",
    "diagnostic": "Diagnostic only",
    "missing": "Missing",
}


def interval_verdict(low, high, lower_is_better: bool = True) -> str:
    """Plain reading of a 95% interval for a difference: better, worse or not distinguishable."""
    if low is None or high is None:
        return "not assessed"
    if low > 0:
        return "worse" if lower_is_better else "better"
    if high < 0:
        return "better" if lower_is_better else "worse"
    return "not distinguishable"


def sentiment_ablation_text(ablation: dict) -> str:
    """One sentence on the registered sentiment ablation, from its pooled row."""
    if ablation.get("status") != "run":
        return "The classifier ablation with sentiment columns was not run."
    row = ablation["rows"][-1]
    reading = {
        "worse": "the forecasts got worse",
        "better": "the forecasts improved",
        "not distinguishable": "no difference could be detected",
    }[interval_verdict(row["ci_low"], row["ci_high"])]
    return (
        f"Adding six FinBERT columns to the classifier for {row['symbol'].replace('+', ' and ')} changed log loss by {row['difference']:+.4f} "
        f"(95% interval {row['ci_low']:+.4f} to {row['ci_high']:+.4f}; lower is better): {reading} on these two tickers."
    )


def sentiment_link_text(rows: list) -> list:
    """One clause per relationship: the correlations and which intervals exclude zero."""
    out = []
    for key, label in [("model_score", "with the classifier's score"), ("trend_flag", "with the trend flag"), ("forward_return", "with the next five-session return")]:
        values = ", ".join(f"{r['symbol']} {r[f'tone_vs_{key}_rho']:+.2f}" for r in rows)
        verdicts = {r["symbol"]: interval_verdict(r[f"tone_vs_{key}_ci_low"], r[f"tone_vs_{key}_ci_high"]) for r in rows}
        clear = [symbol for symbol, verdict in verdicts.items() if verdict in ("better", "worse")]
        unassessed = [symbol for symbol, verdict in verdicts.items() if verdict == "not assessed"]
        note = f"interval excludes zero for {' and '.join(clear)}" if clear else "no interval excludes zero"
        if unassessed:  # too few sessions for an interval: say so instead of reporting a null result
            note = f"no interval could be computed for {' and '.join(unassessed)}" + (f"; {note}" if len(unassessed) < len(rows) else "")
        out.append(f"{label}: {values} ({note})")
    return out


def _load() -> dict:
    return dict(
        classifier=read_json(OUT / "classifier/summary.json", {}),
        regime=read_json(OUT / "regime/summary.json", {}),
        sandbox=read_json(OUT / "sandbox/summary.json", {}),
        replay=read_json(OUT / "replay/summary.json", {}),
        sentiment=read_json(OUT / "sentiment/summary.json", {}),
        inventory=read_json(OUT / "inventory.json", {}),
        live=read_json(OUT / "live/last_run.json", {}),
        index=read_json(OUT / "walkthrough/index.json", {}),
    )


def rows() -> list:
    d = _load()
    metrics = pd.read_csv(OUT / "sandbox/metrics.csv") if (OUT / "sandbox/metrics.csv").exists() else pd.DataFrame()
    periods = pd.read_csv(OUT / "classifier/by_period.csv") if (OUT / "classifier/by_period.csv").exists() else pd.DataFrame()

    def metric(period, arm, column):
        row = metrics.loc[(metrics.period == period) & (metrics.arm == arm)]
        return float(row[column].iloc[0]) if len(row) else float("nan")

    def loss(model):
        row = periods.loc[(periods.period == "all") & (periods.model == model)]
        return float(row.log_loss.iloc[0]) if len(row) else float("nan")

    pilot = d["inventory"].get("pilot", {})
    cross = d["classifier"].get("cross_check", {})
    parity = d["sandbox"].get("history", {}).get("parity", {})
    checks = d["regime"].get("checks", {})
    prospective = d["classifier"].get("prospective", {})
    replay = d["replay"]
    ablation = d["sentiment"].get("ablation", {})
    decisions = pilot.get("decisions", 0)
    fills = pilot.get("fills", 0)
    attempts = pilot.get("attempts", [])
    refits = d["regime"].get("refits", {})
    replay_refits = replay.get("refits", [])
    contrasts = pd.read_csv(OUT / "sandbox/contrasts.csv") if (OUT / "sandbox/contrasts.csv").exists() else pd.DataFrame()
    benchmark = pd.read_csv(OUT / "sandbox/versus_benchmark.csv") if (OUT / "sandbox/versus_benchmark.csv").exists() else pd.DataFrame()
    overlay_supported = bool(len(contrasts)) and bool((contrasts.verdict != "not established").any())
    trend_edge = benchmark.loc[benchmark.arm == "trend"] if len(benchmark) else benchmark
    edge = interval_verdict(float(trend_edge.sharpe_ci_low.iloc[0]), float(trend_edge.sharpe_ci_high.iloc[0]), lower_is_better=False) if len(trend_edge) else "not assessed"
    longest_complete = max((a["required_sessions"] for a in attempts if a["valid_sessions"] == a["required_sessions"]), default=0)
    samples = [(loss("hybrid"), loss("prior")), (prospective.get("hybrid", {}).get("log_loss", float("nan")), prospective.get("prior", {}).get("log_loss", float("nan")))]
    behind = sum(model > prior for model, prior in samples)
    trend_return, equal_return = metric("history", "trend", "net_return"), metric("history", "equal_weight", "net_return")
    return [
        # ---------------------------------------------------------------- works
        dict(area="Data", component="Daily market data collection and validation", status="works",
             finding=f"Public daily bars for 22 instruments are downloaded into immutable, hashed snapshots and checked against the exchange calendar. {pilot.get('snapshots', 0)} snapshots were recorded by the scheduler in October; every file hash still matches.",
             verify="Dashboard section 2; `python -m review_v1.live` downloads and validates a new snapshot", code="runtime_v2/market.py"),
        dict(area="Models", component="Feature pipeline (17 price and volume features)", status="works",
             finding=f"Features recomputed from the recorded bars reproduce the recorded trend flag for all 22 instruments on all {decisions} decision dates.",
             verify="`python -m review_v1.walkthrough` stops if any value differs; `pytest tests/test_review_v1_walkthrough.py`", code="runtime_v2/model.py"),
        dict(area="Models", component="Classifier software (logistic + boosted trees, calibration, purged rolling refits)", status="works",
             finding=(
                 f"Saved models reload with checksum verification, and the replay of all {decisions} recorded decisions stops unless every recorded probability is reproduced. "
                 + (f"The scheduled monthly refit ran in the simulated replay ({', '.join(replay_refits)})." if replay_refits else "No scheduled refit fell inside the simulated replay.")
             ),
             verify="Dashboard section 3, step 3; walkthrough `verification` block; `pytest tests/test_review_v1_walkthrough.py`", code="runtime_v2/model.py"),
        dict(area="Decision", component="Decision rule and portfolio risk constraints", status="works",
             finding="The trend rule and the four portfolio constraints reproduce every recorded target weight exactly; the ledger's turnover cap reproduces every recorded order quantity.",
             verify="Dashboard section 3, step 4; `pytest tests/test_review_v1_walkthrough.py` re-derives each order from the recorded account", code="runtime_v2/policy.py, runtime_v2/risk.py"),
        dict(area="Execution", component="Paper ledger (orders, next-close fills, costs, limits, reconciliation)", status="works",
             finding=f"{fills} fills were recorded by the scheduler. The review harness reproduces the sealed September accounts to the cent (maximum equity difference ${max((v['max_abs_equity_difference_usd'] for v in parity.get('arms', {}).values()), default=float('nan')):.2f}).",
             verify="Dashboard section 7, 'Checks'; `pytest tests/test_review_v1_sandbox.py`", code="runtime_v2/account.py"),
        dict(area="Operations", component="Receipts and audit trail", status="works",
             finding=f"All {decisions} decision receipts match the SHA-256 hashes registered on 27 October, each binds to its plan and its market snapshot.",
             verify="`python -m review_v1.walkthrough` re-checks every hash before replaying; `pytest tests/test_review_v1_walkthrough.py` shows an edited receipt is refused", code="runtime_v2/storage.py, review_v1/receipts.py"),
        dict(area="Evaluation", component="Classifier statistics (new in this review)", status="works",
             finding=f"Accuracy, balanced accuracy, macro F1, log loss, Brier score, per-class precision and recall, confusion matrices and paired date-block intervals against the prior. Agrees with the independently generated research table to {cross.get('max_abs_metric_difference', float('nan')):.0e}.",
             verify="Dashboard section 4; `python -m review_v1.classifier`; `pytest tests/test_review_v1_stats.py` checks the measures against hand-computed values", code="review_v1/classifier.py, review_v1/stats.py"),
        dict(area="Evaluation", component="Trading statistics (new in this review)", status="works",
             finding="Sharpe ratio with a date-block interval, probabilistic and deflated Sharpe ratio, paired Sharpe-difference bootstrap with Holm adjustment. The deflated Sharpe ratio is tested against the worked example in Bailey and López de Prado (2014).",
             verify="`pytest tests/test_review_v1_stats.py tests/test_review_v1_sandbox.py`", code="review_v1/stats.py, review_v1/sandbox.py"),
        dict(area="Demo", component="Review dashboard and decision replay (new in this review)", status="works",
             finding=f"Nine sections; {len(d['index'].get('cases', []))} recorded decisions can be traced step by step for any of 22 instruments; one button runs the current pipeline.",
             verify="`streamlit run app_review.py --server.port 8510`; `pytest tests/test_review_v1_app.py`", code="app_review.py"),
        # ---------------------------------------------------------- diagnostic
        dict(area="Models", component="Hidden Markov model of market regimes (new in this review)", status="diagnostic",
             finding=f"Causal two-state Gaussian HMM on SPY, refit every 21 sessions ({refits.get('converged', 0)} of {refits.get('count', 0)} refits converged). Removing later data {'leaves earlier probabilities unchanged' if checks.get('truncation_passed') else 'CHANGED earlier probabilities: the causality check failed'}. "
                     "It was not an input to any recorded decision. As an exposure overlay in the sandbox "
                     + ("at least one registered contrast was supported on history." if overlay_supported else "neither registered contrast showed an improvement in the Sharpe ratio."),
             verify="Dashboard section 5; `pytest tests/test_review_v1_regime.py`", code="review_v1/regime.py"),
        dict(area="Models", component="News sentiment (FinBERT)", status="diagnostic",
             finding=(
                 "A FinBERT pipeline exists and runs, but it is not connected to the pilot: the pilot feature set has no news column, and every recorded decision reproduces from prices alone. "
                 + sentiment_ablation_text(ablation)
             ),
             verify="Dashboard section 6; `python -m review_v1.sentiment_link`; `pytest tests/test_review_v1_sentiment.py`", code="src/features/sentiment.py, review_v1/sentiment_link.py"),
        # --------------------------------------------------------- unvalidated
        dict(area="Operations", component="Unattended daily collection", status="unvalidated",
             finding=(
                 f"{len(pilot.get('attempts', []))} collection attempts were registered between 29 September and 22 October; together they scheduled {pilot.get('scheduled_decision_sessions', 0)} decision sessions and recorded {decisions}. "
                 f"{pilot.get('attempts_halted', 0)} attempts halted: {len(pilot.get('sessions_not_run', []))} windows passed with no run (the collector is a local launch agent that needs a logged-in, awake machine) "
                 f"and {len(pilot.get('sessions_rejected', []))} download was rejected by validation. The longest attempt that recorded every scheduled session was {longest_complete} session{'s' if longest_complete != 1 else ''} long, so operational reliability is not established."
             ),
             verify="Dashboard section 2, collection calendar", code="scripts/, launchd services"),
        dict(area="Models", component="Predictive value of the classifier", status="unvalidated",
             finding=(
                 f"On {d['classifier'].get('history', {}).get('n', 0):,} saved out-of-fold forecasts the calibrated mixture has log loss {loss('hybrid'):.4f} against {loss('prior'):.4f} for the class-frequency prior (lower is better). "
                 f"On the {prospective.get('n', 0)} recorded October rows it scored {prospective.get('hybrid', {}).get('log_loss', float('nan')):.4f} against {prospective.get('prior', {}).get('log_loss', float('nan')):.4f}. "
                 + {2: "It is behind the prior on both samples.", 1: "It is behind the prior on one of the two samples.", 0: "It is ahead of the prior on both samples; see section 4 for whether the intervals exclude zero."}[behind]
             ),
             verify="Dashboard section 4", code="runtime_v2/model.py"),
        dict(area="Strategy", component="Trading edge of the trend rule", status="unvalidated",
             finding=(
                 f"On the 2024 to September 2026 history the trend rule returned {trend_return:+.1%} (Sharpe {metric('history', 'trend', 'sharpe_zero_rf'):.2f}) against "
                 f"{equal_return:+.1%} (Sharpe {metric('history', 'equal_weight', 'sharpe_zero_rf'):.2f}) for equal weight, and {metric('post_release', 'trend', 'net_return'):+.1%} in the simulated replay since. "
                 + {"not distinguishable": "Its Sharpe ratio is not distinguishable from the benchmark's, so no edge is established.", "better": "Its Sharpe ratio is above the benchmark's on this previously inspected history; that is not prospective evidence.",
                    "worse": "Its Sharpe ratio is below the benchmark's.", "not assessed": "The comparison with the benchmark has not been run."}[edge]
             ),
             verify="Dashboard section 7", code="runtime_v2/policy.py"),
        dict(area="Legacy", component="Summer prototype stack (XGBoost, LightGBM, LSTM, gating network, Alpaca broker adapter, original dashboards)", status="unvalidated",
             finding="Present under `src/` with unit tests, built on a different label and split design. It is not part of the pilot contract and none of its results are used in this review.",
             verify="`pytest tests/` (legacy suites)", code="src/models, src/live, app.py"),
        dict(area="Models", component="Prototype HMM in `src/features/regime.py`", status="unvalidated",
             finding="Fits once on the first 252 rows and emits hard labels. Superseded for this review by the rolling, causal version; left unchanged.",
             verify="Compare with `review_v1/regime.py`", code="src/features/regime.py"),
        # ------------------------------------------------------------- missing
        dict(area="Proposal", component="Main experiment: learned gate versus static ensemble", status="missing",
             finding="The contextual mixture gate over four experts, evaluated through one trading policy with registered contrasts, has not been built. The October screen tested probability combination only.",
             verify="docs/research-controller-v1/PROPOSAL_ALIGNMENT_AND_GAPS.md", code="not present"),
        dict(area="Proposal", component="Proposal label (next open, cost- and volatility-scaled threshold)", status="missing",
             finding="Everything measured so far uses next-close entry and a fixed ±1% threshold. With a fixed threshold the class mix moves with volatility; whether the proposal's volatility-scaled label changes the comparison with the prior has not been tested.",
             verify="Dashboard section 4, label mix by regime", code="not present"),
        dict(area="Proposal", component="HMM state-count selection and use as a gate input", status="missing",
             finding="Two states are fixed by assumption here. Choosing among 2, 3 and 4 states on validation data, and testing the gate with and without regime inputs, are not done.",
             verify="data/review-v1/regime/summary.json, `assumptions`", code="not present"),
        dict(area="Proposal", component="Text data beyond headlines (filings, FOMC, LLM features)", status="missing",
             finding="Only Alpaca headlines for three tickers are cached, plus a one-month fetch for the pilot window made for this review.",
             verify="Dashboard section 2", code="not present"),
        dict(area="Data", component="Price history back to 2010", status="missing",
             finding="Not extended, on purpose. Every evaluation here starts in 2024 with a 1,008-session training window, and the HMM starts in 2019; none of it needs pre-2016 bars. The proposal wants 2010 only so the 2011 and 2015 to 2016 stress periods are in sample, which matters when the stress-period test is run.",
             verify="data/review-v1/inventory.json", code="not needed yet"),
        dict(area="Evidence", component="A prospective sample long enough to judge", status="missing",
             finding=f"{decisions} decision dates and {prospective.get('n', 0)} matured outcome rows exist. The simulated replay adds {replay.get('sessions', 0)} sessions but is not prospective evidence. At least one uninterrupted quarter of recorded sessions is needed before a Sharpe ratio means anything.",
             verify="Dashboard sections 4 and 7", code="operational"),
    ]


def summary(items: list | None = None) -> dict:
    items = rows() if items is None else items
    return {status: sum(r["status"] == status for r in items) for status in STATUS_TEXT}
