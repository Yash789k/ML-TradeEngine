"""Run the current pipeline once, end to end, in an isolated demo workspace.

This is what the dashboard's "Run the pipeline now" button calls. It uses the
pilot's own code for every step (download and validation, features, models,
trend rule, risk constraints, ledger) and adds the review's HMM regime as a
diagnostic. It differs from the scheduled pilot in three deliberate ways:

* it writes only under ``data/paper/review-v1/live/`` and never touches a pilot
  workspace, release or scheduler;
* its account starts from cash on every run, so the orders shown are a first
  purchase, not an incremental rebalance;
* it stops at pending orders. No fill is simulated, and nothing is sent to any
  broker. A demo run is never prospective evidence.

Each run saves a receipt to ``data/review-v1/live/last_run.json``.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

from review_v1 import fresh, regime_report
from review_v1.paths import CLASSES, OUT, SEGMENTS, SIMULATED, WORK, read_json, rel, write_json
from review_v1.regime import RegimeConfig, rolling_regime
from review_v1.registration import POST_RELEASE_START, SYMBOLS

RECEIPT = OUT / "live/last_run.json"
LIVE = WORK / "live"
CONFIG = dict(anchor=POST_RELEASE_START, symbols=SYMBOLS, capital=10000)
STEP_TITLES = [
    ("session", "Find the latest completed market session"),
    ("download", "Download daily bars from the public provider"),
    ("validate", "Validate the snapshot (hashes, calendar, freshness)"),
    ("models", "Load the current models"),
    ("signals", "Compute features, classifier forecasts and the trend rule"),
    ("regime", "Infer the market regime (HMM, diagnostic)"),
    ("risk", "Apply portfolio risk constraints"),
    ("orders", "Create orders in an isolated demo ledger"),
    ("receipt", "Write the run receipt"),
]


def current_models(session) -> tuple:
    """Latest scheduled models: from the simulated replay if it reached a refit, else the carried ones."""
    carried = {s: str(SEGMENTS["r2"]["release"] / "models" / s) for s in SYMBOLS}
    newest = None
    root = WORK / "replay"
    if root.exists():
        for done in sorted(root.glob("*/replay.json")):
            days = read_json(done)["days"]
            if days and pd.Timestamp(days[-1]["session"]) <= pd.Timestamp(session):
                newest = days[-1]
    if newest and all(Path(p).exists() for p in newest["models"].values()):
        return newest["models"], f"models after the scheduled refit(s) in the simulated replay ending {newest['session']}"
    return carried, "carried 25 September pilot models"


def run(progress=None, now=None) -> dict:
    """Execute every step; ``progress(step_id, title, status, detail)`` is called as it goes."""
    from runtime_v2 import market, policy
    from runtime_v2.account import Account
    from runtime_v2.risk import limits

    steps, started = [], time.perf_counter()

    def step(identifier, function):
        title = dict(STEP_TITLES)[identifier]
        if progress:
            progress(identifier, title, "running", "")
        t = time.perf_counter()
        try:
            value, detail = function()
        except Exception as exc:
            steps.append(dict(id=identifier, title=title, status="failed", seconds=time.perf_counter() - t, detail=f"{type(exc).__name__}: {exc}"))
            if progress:
                progress(identifier, title, "failed", steps[-1]["detail"])
            raise
        steps.append(dict(id=identifier, title=title, status="ok", seconds=time.perf_counter() - t, detail=detail))
        if progress:
            progress(identifier, title, "ok", detail)
        return value

    stamp = policy.utc(now).strftime("%Y%m%dT%H%M%S%f")
    workspace = LIVE / stamp
    try:
        session = step("session", lambda: (lambda s: (s, f"{s.date()} (closed {policy.calendar().session_close(s).strftime('%H:%M UTC')})"))(fresh.latest_completed_session(now)))
        meta = step("download", lambda: (lambda m: (m, f"{len(m['data'])} instruments, {m['attempted_requests']} requests, about ten years of daily bars each"))(market.download(workspace / "market", SYMBOLS, session)))
        frames = step("validate", lambda: (lambda f: (f, f"all file hashes match; last bar {f['SPY'].index[-1].date()} for every instrument; {len(f['SPY'])} sessions for SPY"))(market.load(workspace / "market", SYMBOLS, session, now=meta["finished_at"])[0]))
        models, model_note = step("models", lambda: (lambda m: (m, m[1]))(current_models(session)))
        plan = step("signals", lambda: (lambda p: (p, f"{sum(s['trend'] for s in p['signals'])} of {len(SYMBOLS)} instruments pass the trend rule; refit due today: {p['refit']}"))(policy.plan(frames, models, session, CONFIG, workspace / "models")))

        def regime():
            posterior, _ = rolling_regime(regime_report.naive(frames["SPY"].Close), RegimeConfig())
            row = posterior.iloc[-1]
            return dict(session=str(posterior.index[-1].date()), p_turbulent=float(row.p_turbulent), state=row.state, fit_date=str(pd.Timestamp(row.fit_date).date())), f"{row.state} (P(turbulent) = {row.p_turbulent:.3f})"

        regime_now = step("regime", regime)
        gross = float(sum(plan["shadow_weights"].values()))
        step("risk", lambda: (None, f"target gross exposure {gross:.1%} after the 10% / 35% / 80% caps and the 12% volatility target"))
        result = step("orders", lambda: (lambda r: (r, f"{sum(1 for e in Account(SYMBOLS, CONFIG['capital'], limits(), workspace / 'shadow.sqlite').events if e['kind'] == 'order')} pending orders in the demo shadow account; deployment account stays in cash"))(policy.commit(plan, workspace, CONFIG)))
        events = Account(SYMBOLS, CONFIG["capital"], limits(), workspace / "shadow.sqlite").events
        orders = [e for e in events if e["kind"] == "order"]
        signals = []
        for s in plan["signals"]:
            p = s["probabilities"]
            order = next((o for o in orders if o["symbol"] == s["symbol"]), None)
            signals.append(
                dict(symbol=s["symbol"], p_down=p[0], p_flat=p[1], p_up=p[2], top_class=CLASSES[int(np.argmax(p))], gate=s["gate"], trend=s["trend"],
                     target_weight=plan["shadow_weights"][s["symbol"]], overlay_weight=plan["shadow_weights"][s["symbol"]] * (1 - regime_now["p_turbulent"]),
                     close=plan["marks"][s["symbol"]], order_side=order["side"] if order else None, order_quantity=order["qty"] if order else None,
                     order_limit_price=order["limit_price"] if order else None, model_fitted=str(pd.Timestamp(s["model_decision"]).date()))
            )
        receipt = dict(
            schema=1, provenance=SIMULATED,
            warning="Demo run in an isolated account. Orders are pending only: no fill is simulated and nothing was sent to a broker. Not prospective evidence.",
            run_id=stamp, started_at=policy.utc(now).isoformat(), completed_at=policy.utc().isoformat(), seconds=time.perf_counter() - started,
            session=plan["session"], information_cutoff=plan["information_cutoff"], workspace=rel(workspace),
            data=dict(provider=meta["provider"], retrieved_at=meta["retrieved_at"], convention=meta["convention"], instruments=len(meta["data"])),
            models=model_note, refit_today=plan["refit"], regime=regime_now,
            decision=dict(
                rule="trend rule only; classifier probabilities and the HMM are recorded as diagnostics",
                trend_count=int(sum(s["trend"] for s in plan["signals"])), target_gross=gross,
                overlay_gross=gross * (1 - regime_now["p_turbulent"]),
                overlay_note="what the sandbox regime overlay would target today; shown for comparison, not executed",
            ),
            account=dict(
                shadow={k: result["shadow"][k] for k in ["equity", "cash", "reserved_cash", "gross_exposure", "fees"]},
                deployment={k: result["deployment"][k] for k in ["equity", "cash", "gross_exposure"]},
                orders=len(orders), external_orders=False,
                note="The demo account starts from cash on every run, so these orders are a first purchase scaled by the 40% turnover cap.",
            ),
            signals=signals, steps=steps,
        )
        step("receipt", lambda: (write_json(RECEIPT, receipt), rel(RECEIPT)))
        receipt["steps"] = steps
        write_json(RECEIPT, receipt)
        return receipt
    except Exception as exc:
        failure = dict(schema=1, provenance=SIMULATED, run_id=stamp, status="failed", error=f"{type(exc).__name__}: {exc}", steps=steps,
                       completed_at=policy.utc().isoformat(), note="The previous successful receipt, if any, was left untouched.")
        write_json(OUT / "live/last_failure.json", failure)
        raise


def last_run() -> dict | None:
    return read_json(RECEIPT)


def caption(run: dict | None) -> str:
    """One-sentence summary of a saved run. The dashboard and the narration file both use it, so they cannot disagree."""
    if not run:
        return "No demo run has been saved yet."
    return (
        f"Last saved run: session {run['session']}, {run['decision']['trend_count']} of {len(run['signals'])} instruments pass the trend rule, "
        f"{run['account']['orders']} pending orders in an isolated demo account, regime {run['regime']['state']}. "
        f"Completed in {run['seconds']:.0f} seconds. No fill is simulated and nothing is sent to a broker."
    )


if __name__ == "__main__":
    out = run(progress=lambda i, title, status, detail: print(f"[{status:7s}] {title} {detail}", flush=True) if status != "running" else None)
    print(out["seconds"], out["regime"], out["decision"], out["account"]["shadow"])
