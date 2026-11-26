"""Build every evidence file the review dashboard and report read.

    python -m review_v1.build all            # every stage, in order
    python -m review_v1.build classifier     # one stage
    python -m review_v1.build --list

Stages and what they need:

    regime       sealed market history + one public download
    classifier   sealed forecasts (+ recorded outcome rows if present)
    replay       public download + carried pilot models   [simulated]
    sandbox      sealed targets and prices + replay        [registered comparison]
    sentiment    Alpaca headlines + local FinBERT          [slow the first time]
    walkthrough  recorded receipts, snapshots and ledgers
    inventory    counts of everything collected
    narration    captions generated from the numbers above
    report       docs/review-v1/PROGRESS_REPORT.md, generated from the same numbers
    manifest     hashes and the list of checks

The registration (``python -m review_v1.registration``) must exist first. Sealed
releases and recorded receipts are gitignored local evidence: on a fresh clone the
committed files under ``data/review-v1/`` are what the dashboard shows, and the
stages that need local evidence say so instead of producing partial output.
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from review_v1 import audit, nav, receipts, registration
from review_v1.paths import ADAPTIVE_RELEASE, OUT, ROOT, SEGMENTS, WORK, read_json, rel, sha256, write_json
from review_v1.registration import SYMBOLS


COLLECTION_WINDOW = ("2026-09-29", "2026-10-30")  # first registered session to the end of October


def pct(value, digits=1) -> str:
    return f"{value:+.{digits}%}"


# --------------------------------------------------------------------------
# Inventory of collected data
# --------------------------------------------------------------------------

def collection_calendar() -> list:
    """One row per XNYS session of the October collection: what ran and what did not."""
    from runtime_v2.calendar import calendar

    sessions = [str(s.date()) for s in calendar().sessions_in_range(COLLECTION_WINDOW[0], COLLECTION_WINDOW[1])]
    daily, outcomes, segment_of = {}, {}, {}
    # Later attempts supersede earlier ones on a shared date (r3 was replaced by r4 on 14 October).
    for item in receipts.attempts():
        for s in sessions:
            if item["start"] <= s <= item["end"] and (s in item["daily"] or s not in segment_of):
                segment_of[s] = item["attempt"]
        daily.update(item["daily"])
        outcomes.update(item["outcomes"])
    for item in receipts.outcome_snapshots():
        if "outcomes-r7" in str(item["path"]):
            outcomes[item["session"]] = "outcome_recovery_read"
    text = dict(
        complete="Decision recorded", missed_deadline="Missed: collector did not run", halted="Halted after a missed session",
        failed="Failed: download rejected by validation",
        labels_attached_terminal_failed="Outcome prices read", missed_outcome_read="Outcome read missed",
        outcome_recovery_read="Outcome prices read (recovery collector)",
    )
    rows = []
    for s in sessions:
        rows.append(
            dict(
                session=s, weekday=pd.Timestamp(s).strftime("%a"), segment=segment_of.get(s),
                decision=text.get(daily.get(s), "No decision scheduled") if s in segment_of or s in daily else "No decision scheduled",
                decision_status=daily.get(s, "not_scheduled"),
                outcome_read=text.get(outcomes.get(s), "") if s in outcomes else "",
                valid_receipt=bool(daily.get(s) == "complete"),
            )
        )
    return rows


def inventory() -> dict:
    market = read_json(ADAPTIVE_RELEASE / "market/manifest.json", {})
    out = dict(
        universe=dict(symbols=SYMBOLS, count=len(SYMBOLS), description="12 large US stocks and 10 exchange-traded funds (index, sector, bond and gold)"),
        market_history=dict(
            available=bool(market), provider=market.get("provider"), retrieved_at=market.get("retrieved_at"), convention=market.get("price_convention"),
            instruments=len(market.get("data", {})), rows=int(sum(v["rows"] for v in market.get("data", {}).values())),
            first=str(min(pd.Timestamp(v["start"]) for v in market["data"].values()).date()) if market else None,
            last=str(max(pd.Timestamp(v["end"]) for v in market["data"].values()).date()) if market else None,
            path=rel(ADAPTIVE_RELEASE / "market"),
        ),
    )
    if receipts.available():
        cases = receipts.registered_decisions()
        events = {name: receipts.ledger_events(name) for name in SEGMENTS}
        segments = []
        for name in SEGMENTS:
            status = receipts.segment_status(name)
            segments.append(
                dict(
                    segment=name, start=status["config"]["start"], end=status["config"]["end"], required_sessions=status["config"]["sessions"],
                    valid_sessions=status["status"].get("valid_observed_sessions"), missing_sessions=status["status"].get("missing_sessions", []),
                    halted=bool(status["halt"]), halt_reason=(status["halt"] or {}).get("reason"),
                    orders=sum(e["kind"] == "order" for e in events[name]), fills=sum(e["kind"] == "fill" for e in events[name]),
                    last_equity=[e for e in events[name] if e["kind"] == "mark"][-1]["equity"], fees=receipts.ledger_state(name)["fees"],
                )
            )
        snapshots = []
        for case in cases:
            item = receipts.load_decision(case)
            snapshots.append(dict(kind="decision", session=case["session"], retrieved_at=item["manifest"]["retrieved_at"], instruments=len(item["manifest"]["data"]),
                                  rows=int(sum(v["rows"] for v in item["manifest"]["data"].values())), path=rel(item["snapshot"])))
        for item in receipts.outcome_snapshots():
            manifest = read_json(item["path"] / "manifest.json")
            snapshots.append(dict(kind="outcome read", session=item["session"], retrieved_at=item["retrieved_at"], instruments=len(manifest["data"]),
                                  rows=int(sum(v["rows"] for v in manifest["data"].values())), path=rel(item["path"])))
        outcomes = receipts.recorded_outcomes()
        attempts = [{k: v for k, v in a.items() if k not in ("daily", "outcomes")} for a in receipts.attempts()]
        out["pilot"] = dict(
            available=True, decisions=len(cases), decision_sessions=[c["session"] for c in cases], snapshots=len(snapshots),
            attempts=attempts, collection_window=list(COLLECTION_WINDOW),
            scheduled_decision_sessions=int(sum(a["required_sessions"] for a in attempts)),
            attempts_halted=int(sum(a["halted"] for a in attempts)),
            sessions_not_run=sorted(s for a in attempts for s in a["not_run"]),
            sessions_rejected=sorted(s for a in attempts for s in a["rejected"]),
            outcome_recovery={
                k: receipts.outcome_report().get("status_snapshot", {}).get(k)
                for k in ["required_outcome_reads", "completed_outcome_reads", "required_mature_decision_dates", "mature_decision_dates", "missing_new_sessions"]
            },
            orders=sum(s["orders"] for s in segments), fills=sum(s["fills"] for s in segments), outcome_rows=int(len(outcomes)),
            outcome_decision_sessions=sorted(outcomes.decision_session.unique().tolist()) if len(outcomes) else [],
            segments=segments, snapshot_list=sorted(snapshots, key=lambda r: r["retrieved_at"]), calendar=collection_calendar(),
            external_orders=False,
        )
    else:
        out["pilot"] = dict(available=False, reason="recorded receipts are gitignored local evidence and are not on this machine")
    news = []
    for symbol in ["AAPL", "NVDA", "SPY"]:
        path = ROOT / f"data/news/{symbol}_news.parquet"
        if path.exists():
            frame = pd.read_parquet(path, columns=["created_at"])
            created = pd.to_datetime(frame.created_at, utc=True)
            news.append(dict(symbol=symbol, headlines=int(len(frame)), first=str(created.min().date()), last=str(created.max().date()), source="Alpaca News API, cached July 2026"))
    window = []
    for symbol in SYMBOLS:
        path = WORK / f"news/pilot-window/{symbol}_news.parquet"
        if path.exists():
            window.append(dict(symbol=symbol, headlines=int(len(pd.read_parquet(path, columns=["created_at"])))))
    out["news"] = dict(history=news, pilot_window=window, pilot_window_total=int(sum(w["headlines"] for w in window)),
                       pilot_window_without_headlines=[w["symbol"] for w in window if w["headlines"] == 0],
                       note="Headline text is kept in local caches only; the repository holds daily aggregates.")
    replay = read_json(OUT / "replay/summary.json", {})
    out["fresh_download"] = replay.get("snapshot", {})
    models = SEGMENTS["r2"]["release"] / "models"
    if models.exists():
        manifest = read_json(models / "SPY/manifest.json")
        out["models"] = dict(count=len(list(models.glob("*/manifest.json"))), fitted_at=str(pd.Timestamp(manifest["decision"]).date()),
                             fit_rows=manifest["n_fit"], calibration_rows=manifest["n_calibration"], features=len(manifest["columns"]))
    write_json(OUT / "inventory.json", out)
    return out


# --------------------------------------------------------------------------
# Narration: captions generated from the same numbers the dashboard shows
# --------------------------------------------------------------------------

def narration() -> dict:
    classifier = read_json(OUT / "classifier/summary.json")
    regime = read_json(OUT / "regime/summary.json")
    sandbox = read_json(OUT / "sandbox/summary.json")
    sentiment = read_json(OUT / "sentiment/summary.json", {})
    stock = read_json(OUT / "inventory.json")
    from review_v1.live import caption as live_caption

    live = read_json(OUT / "live/last_run.json", {})
    index = read_json(OUT / "walkthrough/index.json")
    case = read_json(OUT / f"walkthrough/{index['default_case']}.json")
    item = case["symbols"][index["default_symbol"]]
    metrics = pd.read_csv(OUT / "sandbox/metrics.csv").set_index(["period", "arm"])
    contrasts = pd.read_csv(OUT / "sandbox/contrasts.csv").set_index("arm")
    periods = pd.read_csv(OUT / "classifier/by_period.csv").set_index(["period", "model"])
    paired = pd.read_csv(OUT / "classifier/paired_vs_prior.csv").set_index(["period", "model"])
    pilot = stock["pilot"]
    hist = lambda arm, column: float(metrics.loc[("history", arm), column])
    post = lambda arm, column: float(metrics.loc[("post_release", arm), column])
    hybrid, prior = periods.loc[("all", "hybrid")], periods.loc[("all", "prior")]
    gap = paired.loc[("all", "hybrid")]
    symbol = index["default_symbol"]
    probabilities = item["models"]["recorded"]["probabilities"]
    date = pd.Timestamp(case["session"]).strftime("%-d %B %Y")
    fill, outcome, position = item["fill"], item["outcome"], item["position"]
    turbulent = next(s for s in regime["states"] if s["state"] == "turbulent")
    calm = next(s for s in regime["states"] if s["state"] == "calm")
    ablation = sentiment.get("ablation", {})

    views = dict(
        overview=dict(
            headline="The pipeline runs end to end. Its forecasts and its strategy have not yet beaten simple baselines.",
            caption=(
                f"{pilot['decisions']} decisions and {pilot['fills']} paper fills were recorded in October 2026. "
                f"On {int(hybrid.n):,} saved forecasts the classifier's log loss is {hybrid.log_loss:.4f} against {prior.log_loss:.4f} for the class-frequency prior (lower is better). "
                f"The trend rule returned {pct(hist('trend', 'net_return'))} on the 2024 to September 2026 history against {pct(hist('equal_weight', 'net_return'))} for equal weight."
            ),
        ),
        data=dict(
            headline="What was collected, and where the collection broke.",
            caption=(
                f"{stock['universe']['count']} instruments, about ten years of daily bars each ({stock['market_history']['rows']:,} rows from {stock['market_history']['first']} to {stock['market_history']['last']}). "
                f"{len(pilot['attempts'])} collection attempts scheduled {pilot['scheduled_decision_sessions']} decision sessions and recorded {pilot['decisions']}, with {pilot['snapshots']} hashed snapshots. "
                f"{pilot['attempts_halted']} attempts halted after a missed or rejected session."
            ),
        ),
        walkthrough=dict(
            headline=f"One recorded decision: {symbol} at the close of {date}.",
            caption=(
                f"The system {item['action'].lower()}s {symbol} because the trend rule was on. "
                f"The classifier's most likely class was {item['models']['recorded']['top_class']} ({max(probabilities):.0%}), and it was not used. "
                f"Every recorded number on this page reproduces from the recorded inputs."
            ),
        ),
        classifiers=dict(
            headline="The classifier does not beat the class-frequency prior.",
            caption=(
                f"All {int(hybrid.sessions)} history dates: log loss {hybrid.log_loss:.4f} versus {prior.log_loss:.4f}, difference {gap.log_loss_difference:+.4f} "
                f"(95% interval {gap.log_loss_ci_low:+.4f} to {gap.log_loss_ci_high:+.4f}); accuracy {hybrid.accuracy:.1%} versus {prior.accuracy:.1%}. "
                f"On the {classifier['prospective'].get('n', 0)} recorded October rows: {classifier['prospective'].get('hybrid', {}).get('log_loss', float('nan')):.4f} versus {classifier['prospective'].get('prior', {}).get('log_loss', float('nan')):.4f}."
            ),
        ),
        regimes=dict(
            headline="A causal two-state HMM separates calm from turbulent markets. It is diagnostic.",
            caption=(
                f"{regime['refits']['count']} rolling refits on SPY, all converged. Calm state about {calm['median_annualised_volatility']:.0%} annualised volatility, turbulent about {turbulent['median_annualised_volatility']:.0%}. "
                f"It flags 2020, 2022 and April 2025 as turbulent. No recorded decision used it."
            ),
        ),
        sentiment=dict(
            headline="Sentiment is not connected to any decision.",
            caption=(
                "The pilot's 17 features contain no news input, and every recorded decision reproduces from prices alone. "
                + audit.sentiment_ablation_text(ablation)
            ),
        ),
        performance=dict(
            headline="No trading edge over the benchmark has been established.",
            caption=(
                f"History, {sandbox['history']['sessions']} sessions, 15 bp costs per side: trend {pct(hist('trend', 'net_return'))} (Sharpe {hist('trend', 'sharpe_zero_rf'):.2f}), "
                f"equal weight {pct(hist('equal_weight', 'net_return'))} (Sharpe {hist('equal_weight', 'sharpe_zero_rf'):.2f}). "
                f"The HMM overlay changed the trend rule's Sharpe by {contrasts.loc['trend_regime', 'sharpe_difference']:+.2f} "
                f"(95% interval {contrasts.loc['trend_regime', 'sharpe_ci_low']:+.2f} to {contrasts.loc['trend_regime', 'sharpe_ci_high']:+.2f}): not established. "
                f"Simulated replay since 29 September, {sandbox['post_release']['sessions']} sessions: trend {pct(post('trend', 'net_return'))}, equal weight {pct(post('equal_weight', 'net_return'))}; too short for a Sharpe ratio."
            ),
        ),
        live=dict(
            headline="The current pipeline, run on the latest completed session.",
            caption=live_caption(live or None),
        ),
        report=dict(
            headline="What the demo shows, and what it does not.",
            caption=(
                "Shows: data collection, validation, forecasting, the decision rule, risk limits, paper orders and fills all work, and every recorded decision can be reproduced from its inputs. "
                "Does not show: that the forecasts beat a naive baseline, that the strategy beats equal weight, or that the scheduler can run unattended. "
                f"Recorded evidence so far is {pilot['decisions']} decision dates out of {pilot['scheduled_decision_sessions']} scheduled."
            ),
        ),
    )
    regime_case = case["regime"]
    risk = case["risk"]["account"]
    steps = {
        "1": f"At {pd.Timestamp(case['collection']['retrieved_at']).strftime('%H:%M UTC')} on {date} the collector downloaded {item['inputs']['rows']:,} daily bars for {symbol} (and 21 other instruments) from {case['collection']['provider']}. The last bar closed at ${item['inputs']['last_bars'][-1]['close']:.2f}. No news data was collected.",
        "2": f"17 features are recomputed from those bars. Two drive the trend rule: price is {item['decision']['trend_inputs']['average_200']:+.1%} versus its 200-session average and 12-month momentum is {item['decision']['trend_inputs']['momentum_12_1']:+.1%}.",
        "3": f"The recorded forecast for {symbol}: DOWN {probabilities[0]:.0%}, FLAT {probabilities[1]:.0%}, UP {probabilities[2]:.0%}. The HMM, computed now from the same bars, reads {regime_case['state']} (P(turbulent) = {regime_case['p_turbulent']:.3f}). Neither was used to decide.",
        "4": f"Trend rule on, so the target is {item['weight']:.2%} of equity. {case['rule']['trend_count']} of 22 instruments qualified, for {case['rule']['final_gross']:.1%} gross exposure. The 40% turnover cap was {'binding' if risk['binding'] else 'not binding'}: every order was scaled to {risk['scale']:.1%} of its target.",
        "5": (f"A {item['order']['side']} order for {item['order']['qty']:.6f} units with a reserve price of ${item['order']['limit_price']:.2f} (close plus 3%). ${item['order']['reserved_cash']:.2f} of cash was reserved. The deployment account stayed in cash; no external order exists."
              if item["order"]["created"] else "No order was created for this instrument."),
        "6": (f"Filled at the next close, {pd.Timestamp(fill['asof']).strftime('%-d %B')}, at ${fill['price']:.2f} with a ${fill['fee']:.2f} cost (15 bp). After all {case['fills']['count']} fills the account was worth ${case['account']['after_fill']['equity']:,.2f} with {case['account']['after_fill']['gross_exposure']:.1%} invested."
              if fill["filled"] else "No fill was recorded: the collector did not run on the next session."),
        "7": (
            (f"Over the five sessions after entry {symbol} returned {pct(outcome['forward_return'])}: class {outcome['label']}. The classifier's top class was {outcome['predicted']}. " if outcome["available"] else "")
            + (f"Held to {position['latest']['session']}, the position would show {position['latest']['pnl']:+.2f} USD ({pct(position['latest']['pnl_percent'])}); " if position["opened"] else "")
            + f"the recorded portfolio, simply held, would be worth ${case['held_forward']['latest']['equity']:,.2f} ({pct(case['held_forward']['latest']['net_return'])}). One case proves nothing about skill."
        ),
    }
    out = dict(
        schema=1, generated_at=pd.Timestamp.now(tz="UTC").isoformat(), default_case=index["default_case"], default_symbol=symbol,
        views=views,
        walkthrough={str(s["n"]): dict(title=s["title"], caption=steps[str(s["n"])], provenance=s["provenance"]) for s in nav.STEPS},
    )
    write_json(OUT / "narration.json", out)
    return out


# --------------------------------------------------------------------------
# Manifest: hashes and checks
# --------------------------------------------------------------------------

def checks() -> list:
    classifier = read_json(OUT / "classifier/summary.json", {})
    regime = read_json(OUT / "regime/summary.json", {})
    sandbox = read_json(OUT / "sandbox/summary.json", {})
    replay = read_json(OUT / "replay/summary.json", {})
    sentiment = read_json(OUT / "sentiment/summary.json", {})
    index = read_json(OUT / "walkthrough/index.json", {})
    rows = []

    def add(name, passed, detail):
        rows.append(dict(check=name, passed=bool(passed), detail=detail))

    add("Registration written before evaluation", (OUT / "registration.json").exists(), f"sha256 {sha256(OUT / 'registration.json')[:16]}…; amendments: {len(read_json(OUT / 'registration_amendments.json', []))}")
    if index:
        cases = [read_json(OUT / f"walkthrough/{c['file']}") for c in index["cases"]]
        add("Decision receipts match registered hashes", all(c["verification"]["receipt_matches_registration"] for c in cases), f"{len(cases)} receipts")
        add("Recorded probabilities reproduce from recorded bars", all(c["verification"]["max_abs_probability_difference"] < 1e-12 for c in cases),
            f"maximum difference {max(c['verification']['max_abs_probability_difference'] for c in cases):.1e} over {len(cases) * len(SYMBOLS)} instrument-decisions")
        add("Recorded weights and order quantities reproduce", all(c["verification"]["weights_reproduced"] and c["verification"]["order_quantities_reproduced"] for c in cases), "trend rule, four portfolio constraints, turnover cap")
    if classifier:
        cross = classifier["cross_check"]
        add("Classifier statistics match the independent research table", cross["passed"], f"{cross['rows_compared']} rows, maximum difference {cross['max_abs_metric_difference']:.1e}")
        match = classifier["prospective"].get("matches_recorded_report")
        if match:
            add("Recorded outcome metrics match the collector's own report", match["max_abs_difference"] < 1e-12, f"maximum difference {match['max_abs_difference']:.1e}")
    if regime:
        add("HMM is causal (removing later data changes nothing earlier)", regime["checks"]["truncation_passed"],
            "; ".join(f"cut {c['cut']}: {c['rows_compared']} rows, max difference {c['max_abs_difference']:.1e}" for c in regime["checks"]["truncation"]))
        add("HMM stable across data vintages", regime["checks"]["vintage"]["label_agreement"] == 1.0,
            f"label agreement {regime['checks']['vintage']['label_agreement']:.1%}, max probability difference {regime['checks']['vintage']['max_abs_probability_difference']:.1e}")
        add("All HMM refits converged", regime["refits"]["converged"] == regime["refits"]["count"], f"{regime['refits']['converged']} of {regime['refits']['count']}")
    if sandbox:
        add("Sandbox harness reproduces the sealed September accounts", sandbox["history"]["parity"]["passed"],
            "; ".join(f"{k}: ${v['max_abs_equity_difference_usd']:.6f}" for k, v in sandbox["history"]["parity"]["arms"].items()))
        add("Sandbox harness equals the pilot's reference replay", sandbox["post_release"]["parity"]["passed"], f"maximum equity difference ${sandbox['post_release']['parity']['max_abs_equity_difference_usd']:.6f}")
    if replay.get("agreement_with_receipts", {}).get("available"):
        agreement = replay["agreement_with_receipts"]
        add("Simulated replay agrees with recorded decisions where both exist", agreement["all_trend_flags_equal"] and agreement["max_abs_weight_difference"] < 1e-12,
            f"trend flags and weights equal on {len(agreement['rows'])} dates; probabilities differ by at most {agreement['max_abs_probability_difference']:.3f} (price vintage)")
    second = replay.get("data_quality", {}).get("second_source", {})
    if second.get("available"):
        add("Replay prices agree with a second source", second["passed"],
            f"{second['rows_compared']} of {second['rows_expected']} daily returns compared, largest difference {second['max_abs_return_difference']:.2%} (tolerance {second['tolerance']:.1%})")
    if sentiment:
        add("Pilot features contain no sentiment input", not sentiment["pilot_features_contain_sentiment"], f"{len(sentiment['pilot_feature_columns'])} price and volume columns")
        if sentiment.get("ablation", {}).get("status") == "run":
            add("Sentiment ablation price-only arm reproduces sealed forecasts", sentiment["ablation"]["parity_passed"],
                "; ".join(f"{k}: {v:.1e}" for k, v in sentiment["ablation"]["parity_with_sealed_price_only"].items()))
    return rows


def manifest() -> dict:
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json")
    sources = sorted((ROOT / "review_v1").glob("*.py")) + [ROOT / "app_review.py"]
    items = audit.rows()
    out = dict(
        schema=1, built_at=pd.Timestamp.now(tz="UTC").isoformat(),
        registration_sha256=sha256(OUT / "registration.json"),
        checks=checks(), audit=items, audit_summary=audit.summary(items),
        source_sha256={rel(p): sha256(p) for p in sources if p.exists()},
        output_sha256={rel(p): sha256(p) for p in files},
        local_evidence_present=receipts.available(),
    )
    write_json(OUT / "manifest.json", out)
    return out


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def _regime():
    from review_v1 import regime_report
    return regime_report.build()


def _classifier():
    from review_v1 import classifier
    return classifier.build()


def _replay():
    from review_v1 import replay
    return replay.build()


def _sandbox():
    from review_v1 import sandbox
    return sandbox.build()


def _sentiment():
    from review_v1 import sentiment_link
    return sentiment_link.build()


def _walkthrough():
    from review_v1 import walkthrough
    return walkthrough.build()


def _report():
    from review_v1 import report
    return report.build()


STAGES = dict(
    regime=_regime, classifier=_classifier, replay=_replay, sandbox=_sandbox, sentiment=_sentiment, walkthrough=_walkthrough,
    inventory=inventory, narration=narration, report=_report, contract=nav.write_contract, manifest=manifest,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", nargs="*", default=["all"])
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(STAGES))
        return 0
    registration.require()
    names = list(STAGES) if args.stage == ["all"] else args.stage
    unknown = [n for n in names if n not in STAGES]
    if unknown:
        parser.error(f"unknown stage(s): {unknown}")
    for name in names:
        started = time.perf_counter()
        print(f"[{name}] …", flush=True)
        STAGES[name]()
        print(f"[{name}] done in {time.perf_counter() - started:.1f}s", flush=True)
    failed = [c for c in read_json(OUT / "manifest.json", {}).get("checks", []) if not c["passed"]] if "manifest" in names else []
    for c in failed:
        print(f"CHECK FAILED: {c['check']}: {c['detail']}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
