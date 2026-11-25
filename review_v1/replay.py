"""Simulated continuous replay of the pilot contract after the release date.

The scheduled collector produced only five decision receipts in October 2026.
This stage answers a different question: what would the *same* code have
decided on every session from the pilot anchor to the latest completed session?

It runs ``runtime_v2.policy.replay`` (the pilot's own reference replay) over a
fresh public snapshot, revealing one session at a time, with fresh isolated
accounts and the carried 25 September models (refit every 21 sessions exactly as
the pilot schedule says). Everything it writes is labelled **simulated**:

* prices are a November download, not the bars the collector saw each evening;
* fills are assumed at the next close, as in every backtest here;
* nothing is a pilot observation, and none of it counts as prospective evidence.

State lives in ``data/paper/review-v1/replay/<snapshot>/``; summaries go to
``data/review-v1/replay/``.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from review_v1 import fresh, quality, receipts, registration, regime_report
from review_v1.paths import CLASSES, OUT, SEGMENTS, SIMULATED, WORK, read_json, rel, write_json
from review_v1.registration import POST_RELEASE_START, SYMBOLS
from review_v1.stats import block_effect, classification_report, row_log_loss

DIR = OUT / "replay"
CONFIG = dict(anchor=POST_RELEASE_START, symbols=SYMBOLS, capital=10000)


def model_paths() -> dict:
    """Carried 25 September models from the first pilot release (same files in r2, r4 and r6)."""
    root = SEGMENTS["r2"]["release"] / "models"
    return {s: str(root / s) for s in SYMBOLS}


def sessions(frames) -> list:
    index = frames["SPY"].index
    return [d for d in index if d >= pd.Timestamp(POST_RELEASE_START, tz="UTC")]


def run(snapshot=None, force: bool = False) -> dict:
    """Run (or reuse) the replay for one snapshot and return its raw day results."""
    from runtime_v2 import policy

    snapshot = fresh.snapshot() if snapshot is None else snapshot
    frames, meta = fresh.load(snapshot)
    workspace = WORK / "replay" / snapshot.name
    done = workspace / "replay.json"
    if done.exists() and not force:
        return dict(read_json(done), frames=frames, workspace=workspace, snapshot=snapshot, meta=meta, reused=True)
    if workspace.exists():
        raise FileExistsError(f"Partial replay workspace {rel(workspace)}: remove it to rerun")
    workspace.mkdir(parents=True)
    dates = sessions(frames)
    started = time.perf_counter()
    days = policy.replay(frames, model_paths(), dates, CONFIG, workspace)
    payload = dict(days=days, seconds=time.perf_counter() - started, completed_at=pd.Timestamp.now(tz="UTC").isoformat())
    write_json(done, payload)
    return dict(read_json(done), frames=frames, workspace=workspace, snapshot=snapshot, meta=meta, reused=False)


def labels(frames, symbol: str, horizon: int = 5, threshold: float = 0.01) -> pd.DataFrame:
    """Forward return and class for every session whose outcome is already observed."""
    from runtime_v2.model import execution_labels

    close = frames[symbol].Close
    y, forward = execution_labels(close)
    return pd.DataFrame(dict(label=y, forward_return=forward, entry=close.shift(-1), exit=close.shift(-(horizon + 1))))


def signal_table(result: dict) -> pd.DataFrame:
    """One row per replay session and instrument, with the outcome once it is known."""
    from runtime_v2.model import AdaptiveModel

    frames = result["frames"]
    outcome = {s: labels(frames, s) for s in SYMBOLS}
    cache, rows = {}, []
    for day in result["days"]:
        date = pd.Timestamp(day["session"], tz="UTC")
        for signal in day["signals"]:
            symbol = signal["symbol"]
            path = day["models"][symbol]
            if path not in cache:
                cache[path] = AdaptiveModel.load(path).prior.tolist()
            item = outcome[symbol].loc[date]
            matured = bool(item.label >= 0)
            rows.append(
                dict(
                    session=day["session"], symbol=symbol,
                    p_down=signal["probabilities"][0], p_flat=signal["probabilities"][1], p_up=signal["probabilities"][2],
                    prior_down=cache[path][0], prior_flat=cache[path][1], prior_up=cache[path][2],
                    trend=signal["trend"], gate=signal["gate"], model_decision=str(pd.Timestamp(signal["model_decision"]).date()),
                    target_weight=day["shadow_weights"][symbol], rebalance=day["rebalance"],
                    matured=matured, label=int(item.label) if matured else None,
                    forward_return=float(item.forward_return) if matured else None,
                )
            )
    return pd.DataFrame(rows)


def day_table(result: dict) -> pd.DataFrame:
    rows = []
    for day in result["days"]:
        shadow = day["shadow"]
        rows.append(
            dict(
                session=day["session"], rebalance=day["rebalance"], refit=day["refit"],
                trend_count=int(sum(s["trend"] for s in day["signals"])),
                target_gross=float(sum(day["shadow_weights"].values())),
                equity=shadow["equity"], cash=shadow["cash"], reserved_cash=shadow["reserved_cash"],
                gross_exposure=shadow["gross_exposure"], fees=shadow["fees"], realized_pnl=shadow["realized_pnl"],
                unrealized_pnl=shadow["unrealized_pnl"], drawdown=shadow["drawdown"],
                positions=int(sum(abs(q) > 1e-9 for q in shadow["positions"].values())),
                deployment_equity=day["deployment"]["equity"],
            )
        )
    return pd.DataFrame(rows)


def classification(signals: pd.DataFrame) -> dict:
    scored = signals.loc[signals.matured].copy()
    if scored.empty:
        return dict(available=False)
    y = scored.label.to_numpy(int)
    hybrid = scored[["p_down", "p_flat", "p_up"]].to_numpy(float)
    prior = scored[["prior_down", "prior_flat", "prior_up"]].to_numpy(float)
    out = dict(
        available=True, provenance=SIMULATED,
        first=scored.session.min(), last=scored.session.max(), sessions=int(scored.session.nunique()), n=int(len(scored)),
        label_share={CLASSES[c]: float(np.mean(y == c)) for c in range(3)},
    )
    for name, p in [("hybrid", hybrid), ("prior", prior)]:
        report = classification_report(y, p)
        out[name] = {k: report[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "log_loss", "brier", "confusion", "classes"]}
    daily = pd.Series(row_log_loss(y, hybrid) - row_log_loss(y, prior), index=scored.session.to_numpy()).groupby(level=0).mean()
    effect = block_effect(daily.to_numpy(), 5, 2000)
    out["hybrid_minus_prior_log_loss"] = {k: effect[k] for k in ["effect", "ci_low", "ci_high", "p_two_sided", "sessions", "block_length"]}
    out["reading"] = (
        f"{out['sessions']} overlapping five-session outcomes from one two-month stretch are roughly "
        f"{max(1, out['sessions'] // 6)} independent windows. Descriptive only; the interval is wide and conditional on this stretch."
    )
    return out


def agreement_with_receipts(result: dict) -> dict:
    """Where a recorded receipt exists for a replay date, do the decisions match?

    Probabilities can differ slightly: the replay uses November-vintage adjusted
    prices, and a dividend paid since October changes every earlier adjusted bar.
    """
    if not receipts.available():
        return dict(available=False)
    by_session = {d["session"]: d for d in result["days"]}
    rows = []
    for case in receipts.registered_decisions():
        recorded = receipts.load_decision(case)["plan"]
        simulated = by_session.get(case["session"])
        if simulated is None:
            continue
        a = {s["symbol"]: s for s in recorded["signals"]}
        b = {s["symbol"]: s for s in simulated["signals"]}
        rows.append(
            dict(
                session=case["session"], case_id=case["case_id"],
                trend_flags_equal=bool(all(a[s]["trend"] == b[s]["trend"] for s in SYMBOLS)),
                max_abs_probability_difference=float(max(np.abs(np.subtract(a[s]["probabilities"], b[s]["probabilities"])).max() for s in SYMBOLS)),
                max_abs_weight_difference=float(max(abs(recorded["shadow_weights"][s] - simulated["shadow_weights"][s]) for s in SYMBOLS)),
            )
        )
    return dict(
        available=True, rows=rows,
        all_trend_flags_equal=bool(all(r["trend_flags_equal"] for r in rows)),
        max_abs_probability_difference=max(r["max_abs_probability_difference"] for r in rows),
        max_abs_weight_difference=max(r["max_abs_weight_difference"] for r in rows),
        note="Recorded receipts used the bars downloaded each evening; the replay uses one November download.",
    )


def build() -> dict:
    reg = registration.require()
    result = run()
    days, signals = day_table(result), signal_table(result)
    DIR.mkdir(parents=True, exist_ok=True)
    days.to_csv(DIR / "days.csv", index=False)
    signals.to_csv(DIR / "signals.csv", index=False)
    posterior = regime_report.combined_posterior()
    refits = [d["session"] for d in result["days"] if d["refit"]]
    summary = dict(
        registration_sha256=reg["sha256"],
        provenance=SIMULATED,
        warning="Simulated replay. Not a pilot observation and not prospective evidence. No order left this machine.",
        snapshot=dict(path=rel(result["snapshot"]), provider=result["meta"]["provider"], retrieved_at=result["meta"]["retrieved_at"], expected_session=result["meta"]["expected_session"]),
        workspace=rel(result["workspace"]),
        sessions=int(len(days)), first=days.session.iloc[0], last=days.session.iloc[-1],
        rebalances=int(days.rebalance.sum()), refits=refits,
        models="carried 25 September pilot models; refit on the scheduled sessions listed in `refits`",
        seconds=result["seconds"],
        account=dict(
            capital=CONFIG["capital"], terminal_equity=float(days.equity.iloc[-1]), net_return=float(days.equity.iloc[-1] / CONFIG["capital"] - 1),
            fees=float(days.fees.iloc[-1]), max_drawdown=float(days.drawdown.min()), mean_gross_exposure=float(days.gross_exposure.mean()),
            deployment_terminal_equity=float(days.deployment_equity.iloc[-1]),
        ),
        regime_at_last=dict(session=str(posterior.index[-1].date()), p_turbulent=float(posterior.p_turbulent.iloc[-1]), state=posterior.state.iloc[-1]),
        classification=classification(signals),
        agreement_with_receipts=agreement_with_receipts(result),
        data_quality=quality.report(result["frames"], result["snapshot"].name, CONFIG["anchor"]),
    )
    write_json(DIR / "summary.json", summary)
    return summary


if __name__ == "__main__":
    out = build()
    print({k: out[k] for k in ["sessions", "first", "last", "rebalances", "refits", "seconds", "account"]})
    print(out["classification"])
    print(out["agreement_with_receipts"])
