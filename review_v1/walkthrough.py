"""Trace recorded decisions from their inputs to their outcomes.

For each of the five decision receipts the scheduled collector wrote in October
2026, this builds one JSON file that the dashboard replays. Every fact carries
one provenance label:

* ``recorded``       written by the collector at the time (receipts, the bars it
                     downloaded, ledger orders, fills and marks, outcome rows);
* ``reconstructed``  recomputed now from those recorded inputs (features, the
                     two expert models behind the recorded probability, the risk
                     arithmetic, the HMM regime, later mark-to-market);
* ``simulated``      never used here. The simulated replay lives on its own page.

Reconstruction is checked, not assumed: recomputed probabilities, trend flags,
target weights and order quantities must equal the recorded values, and the
build stops if they do not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from review_v1 import fresh, nav, receipts, registration, regime_report
from review_v1.paths import CLASSES, OUT, RECONSTRUCTED, RECORDED, read_json, rel, write_json
from review_v1.regime import RegimeConfig, rolling_regime
from review_v1.registration import SYMBOLS

DIR = OUT / "walkthrough"
FEATURE_TEXT = {
    "return_1": "1-session return", "return_5": "5-session return", "return_21": "21-session return",
    "return_63": "63-session return", "return_126": "126-session return", "return_252": "252-session return",
    "vol_21": "21-session daily volatility", "vol_63": "63-session daily volatility",
    "relative_21": "21-session return minus SPY", "relative_63": "63-session return minus SPY",
    "average_20": "Price versus 20-session average", "average_50": "Price versus 50-session average",
    "average_200": "Price versus 200-session average", "gap": "Open versus previous close",
    "range": "High-low range over close", "volume_ratio": "Volume versus 20-session average",
    "momentum_12_1": "12-month return skipping the last month",
}
TREND_INPUTS = ("average_200", "momentum_12_1")


def day(value) -> str:
    return str(pd.Timestamp(value).date())


def risk_trace(raw: pd.Series, returns: pd.DataFrame) -> dict:
    """The four portfolio-level constraints, step by step, with what each one did."""
    from runtime_v2.risk import GROWTH, constrain_weights

    w = raw.clip(lower=0, upper=0.1).astype(float)
    steps = [dict(rule="Per-instrument cap", limit=0.10, value=float(raw.max()), binding=bool((raw > 0.1).any()), scale=1.0,
                  detail="largest single target weight")]
    growth = [s for s in w.index if s in GROWTH]
    total = float(w[growth].sum())
    scale = 0.35 / total if total > 0.35 else 1.0
    w.loc[growth] *= scale
    steps.append(dict(rule="Growth-group cap", limit=0.35, value=total, binding=bool(scale < 1), scale=float(scale),
                      detail="combined weight of AAPL, AMZN, GOOGL, META, MSFT, NVDA, QQQ, TSLA, XLK"))
    gross = float(w.sum())
    scale = 0.8 / gross if gross > 0.8 else 1.0
    w *= scale
    steps.append(dict(rule="Gross exposure cap", limit=0.80, value=gross, binding=bool(scale < 1), scale=float(scale), detail="sum of target weights"))
    cov = returns.tail(63).cov().reindex(index=w.index, columns=w.index).to_numpy() * 252
    cov = 0.5 * cov + 0.5 * np.diag(np.diag(cov))
    vol = float(np.sqrt(max(0, w.to_numpy() @ cov @ w.to_numpy())))
    scale = 0.12 / vol if vol > 0.12 else 1.0
    w *= scale
    steps.append(dict(rule="Volatility target", limit=0.12, value=vol, binding=bool(scale < 1), scale=float(scale),
                      detail="estimated annualised portfolio volatility from the last 63 sessions"))
    if not np.allclose(w, constrain_weights(raw, returns), atol=1e-15):
        raise ValueError("Risk trace disagrees with runtime_v2.risk.constrain_weights")
    return dict(steps=steps, weights=w)


def order_trace(weights: dict, marks: dict, mark: dict, limits) -> dict:
    """Reproduce the ledger's order sizing from the recorded account state."""
    equity = mark["equity"]
    cost = limits.commission + limits.slippage
    deltas = {s: weights[s] * equity / marks[s] - mark["positions"][s] for s in SYMBOLS}
    turnover = sum(abs(q) * marks[s] for s, q in deltas.items())
    scale = min(1.0, limits.max_turnover * equity / turnover) if turnover else 1.0
    rows = {}
    for s in SYMBOLS:
        value = deltas[s] * marks[s]
        rows[s] = dict(
            target_value=weights[s] * equity, held_value=mark["positions"][s] * marks[s], wanted_trade_value=value,
            scaled_trade_value=value * scale, quantity=abs(deltas[s]) * scale, limit_price=marks[s] * (1 + limits.reserve_buffer),
        )
    return dict(equity=equity, turnover=turnover, turnover_share=turnover / equity, turnover_limit=limits.max_turnover, scale=scale,
                binding=bool(scale < 1), cost_rate=cost, reserve_buffer=limits.reserve_buffer, rows=rows)


def regime_at(frames, session) -> dict:
    """HMM regime from the SPY bars in the recorded snapshot (so nothing later can leak in)."""
    posterior, fits = rolling_regime(regime_report.naive(frames["SPY"].Close), RegimeConfig())
    row = posterior.loc[pd.Timestamp(session)]
    state = next(f for f in reversed(fits) if pd.Timestamp(f["fit_date"]) <= pd.Timestamp(session))
    recent = posterior.loc[: pd.Timestamp(session)].tail(60)
    return dict(
        provenance=RECONSTRUCTED,
        p_turbulent=float(row.p_turbulent), p_calm=float(row.p_calm), state=row.state, fit_date=day(row.fit_date),
        states=[dict(name=s["name"], annualised_volatility=s["annualised_volatility"], annualised_return=s["annualised_return"],
                     expected_duration_sessions=s["expected_duration_sessions"]) for s in state["states"]],
        recent=dict(dates=[day(d) for d in recent.index], p_turbulent=[round(float(v), 6) for v in recent.p_turbulent]),
        role="Diagnostic. The HMM did not exist when this decision was made and was not an input to it.",
    )


def symbol_trace(symbol, item, frames, features, trace, sizing, events, outcome_rows, fresh_close, sentiment) -> dict:
    from runtime_v2.model import AdaptiveModel

    plan = item["plan"]
    session = plan["session"]
    signal = next(s for s in plan["signals"] if s["symbol"] == symbol)
    frame, x = frames[symbol], features[symbol]
    model = AdaptiveModel.load(receipts.local(plan["models"][symbol]))
    predicted = model.predict(x.iloc[[-1]])
    hybrid = predicted["hybrid"][0]
    difference = float(np.abs(hybrid - np.asarray(signal["probabilities"])).max())
    trend = bool(x.average_200.iloc[-1] > 0 and x.momentum_12_1.iloc[-1] > 0)
    if difference > 1e-12 or trend != signal["trend"]:
        raise ValueError(f"{symbol} {session}: reconstruction differs from the recorded signal")
    weight = plan["shadow_weights"][symbol]
    if abs(weight - float(trace["weights"][symbol])) > 1e-15:
        raise ValueError(f"{symbol} {session}: reconstructed weight differs from the recorded weight")

    decision_id = plan["information_cutoff"]
    order = next((e for e in events if e["kind"] == "order" and e["id"] == f"{decision_id}:{symbol}"), None)
    fill = next((e for e in events if e["kind"] == "fill" and order and e["order_id"] == order["id"]), None)
    mark = next(e for e in events if e["kind"] == "mark" and e["asof"] == decision_id)
    held = mark["positions"][symbol]
    size = sizing["rows"][symbol]
    if order and abs(order["qty"] - np.floor(size["quantity"] / 1e-6) * 1e-6) > 2e-6:
        raise ValueError(f"{symbol} {session}: reconstructed order quantity differs from the recorded order")
    if not plan["rebalance"]:
        action, reason = "HOLD", "No rebalance was scheduled on this session, so no order was created."
    elif order:
        action = order["side"].upper()
        reason = (
            f"Trend rule is {'on' if trend else 'off'}: target weight {weight:.2%} against {held * plan['marks'][symbol] / mark['equity']:.2%} held."
        )
    elif weight > 0:
        action, reason = "HOLD", "Already at the target weight; the difference was below the one-cent minimum."
    else:
        action, reason = "NO TRADE", "Trend rule is off and nothing is held, so the target is zero."

    tail = frame.tail(260)
    average = frame.Close.rolling(200).mean().tail(260)
    bars = frame.tail(5)
    manifest = item["manifest"]["data"][symbol]
    top = int(np.argmax(signal["probabilities"]))
    raw_mix = (predicted["linear"][0] + predicted["tree"][0]) / 2
    meta = model.metadata

    outcome = dict(available=False)
    recorded = outcome_rows.loc[(outcome_rows.decision_session == session) & (outcome_rows.symbol == symbol)] if len(outcome_rows) else outcome_rows
    if len(recorded):
        r = recorded.iloc[0]
        label, forward = int(r.label), float(r.realized_return)
        outcome = dict(available=True, provenance=RECORDED, entry_session=r.entry, exit_session=r.endpoint, entry_price=float(r.entry_price),
                       exit_price=float(r.endpoint_price), forward_return=forward, label=CLASSES[label], source=f"outcome read recorded on {r.source_session}")
    else:
        close = fresh_close[symbol]
        position = close.index.get_loc(pd.Timestamp(session))
        if position + 6 < len(close):
            entry, exit_ = close.iloc[position + 1], close.iloc[position + 6]
            forward = float(exit_ / entry - 1)
            label = 2 if forward > 0.01 else 0 if forward < -0.01 else 1
            outcome = dict(available=True, provenance=RECONSTRUCTED, entry_session=day(close.index[position + 1]), exit_session=day(close.index[position + 6]),
                           entry_price=float(entry), exit_price=float(exit_), forward_return=forward, label=CLASSES[label],
                           source="no outcome read was recorded for this date; recomputed from the fresh public download")
    if outcome["available"]:
        index = CLASSES.index(outcome["label"])
        outcome.update(
            predicted=CLASSES[top], correct=bool(top == index),
            hybrid_probability_of_outcome=float(signal["probabilities"][index]), prior_probability_of_outcome=float(model.prior[index]),
            hybrid_log_loss=float(-np.log(signal["probabilities"][index])), prior_log_loss=float(-np.log(model.prior[index])),
        )

    position = dict(opened=False)
    if fill and fill["side"] == "buy":
        close = fresh_close[symbol]
        start = pd.Timestamp(day(fill["asof"]))
        path = close.loc[start:]
        value = fill["qty"] * fill["price"] * path / path.iloc[0]
        basis = fill["qty"] * fill["price"] + fill["fee"]
        position = dict(
            opened=True, provenance=RECONSTRUCTED,
            note="The recorded position, marked with later public prices. The recorded ledger itself was last marked on the fill date.",
            quantity=fill["qty"], fill_price=fill["price"], fee=fill["fee"], cost_basis=basis,
            dates=[day(d) for d in path.index], value=[round(float(v), 4) for v in value], pnl=[round(float(v - basis), 4) for v in value],
            latest=dict(session=day(path.index[-1]), value=float(value.iloc[-1]), pnl=float(value.iloc[-1] - basis), pnl_percent=float(value.iloc[-1] / basis - 1)),
        )

    tone = dict(available=False, reason="No headline data was reconstructed for this decision.")
    if sentiment is not None:
        row = sentiment.loc[(sentiment.session == session) & (sentiment.symbol == symbol)]
        if len(row):
            r = row.iloc[0]
            tone = dict(available=True, provenance=RECONSTRUCTED, headlines=int(r.headlines), tone=float(r.tone), positive=float(r.positive),
                        negative=float(r.negative), negative_5d=float(r.negative_5d),
                        role="Diagnostic. Fetched for this review from headlines published before the cutoff. Not an input to the decision.")

    return dict(
        symbol=symbol,
        action=action, reason=reason, weight=weight, trend=trend,
        inputs=dict(
            provenance=RECORDED, rows=manifest["rows"], first=day(manifest["start"]), last=day(manifest["end"]), sha256=manifest["sha256"],
            corporate_actions=dict(dividends=len(manifest["corporate_actions"].get("dividends", {})), splits=len(manifest["corporate_actions"].get("splits", {}))),
            last_bars=[dict(date=day(d), open=float(r.Open), high=float(r.High), low=float(r.Low), close=float(r.Close), volume=float(r.Volume)) for d, r in bars.iterrows()],
            chart=dict(dates=[day(d) for d in tail.index], close=[round(float(v), 4) for v in tail.Close], average_200=[round(float(v), 4) for v in average]),
            sentiment=tone,
        ),
        features=dict(
            provenance=RECONSTRUCTED,
            rows=[dict(name=name, text=FEATURE_TEXT[name], value=float(x[name].iloc[-1]), used_by_trend_rule=name in TREND_INPUTS) for name in x.columns],
        ),
        models=dict(
            provenance=RECORDED,
            recorded=dict(probabilities=signal["probabilities"], expected_return=signal["expected_return"], gate=signal["gate"], top_class=CLASSES[top]),
            reconstructed=dict(
                provenance=RECONSTRUCTED, linear=predicted["linear"][0].tolist(), tree=predicted["tree"][0].tolist(), raw_mix=raw_mix.tolist(),
                hybrid=hybrid.tolist(), prior=model.prior.tolist(), temperature=float(model.temperature), max_abs_difference_from_recorded=difference,
            ),
            model=dict(
                fitted_at=day(meta["decision"]), fit_start=day(meta["fit_start"]), fit_end=day(meta["fit_end"]), n_fit=meta["n_fit"],
                calibration_start=day(meta["calibration_start"]), calibration_end=day(meta["calibration_end"]), n_calibration=meta["n_calibration"],
                latest_label_end=day(meta["latest_label_end"]), calibration_log_loss=meta["calibration_log_loss"], prior_log_loss=meta["prior_log_loss"],
                gate=meta["gate"], manifest_sha256=signal["model_manifest_sha256"],
            ),
            used_in_decision=False,
        ),
        decision=dict(
            provenance=RECORDED,
            trend_inputs=dict(average_200=float(x.average_200.iloc[-1]), momentum_12_1=float(x.momentum_12_1.iloc[-1])),
            raw_weight=float(trend) * 0.8 / len(SYMBOLS), final_weight=weight,
            sizing=dict(provenance=RECONSTRUCTED, **size),
        ),
        order=dict(provenance=RECORDED, created=bool(order), **({k: order[k] for k in ["id", "side", "qty", "limit_price", "decision_at"]} if order else {}),
                   reserved_cash=float(order["qty"] * order["limit_price"] * (1 + sizing["cost_rate"])) if order and order["side"] == "buy" else 0.0),
        fill=dict(provenance=RECORDED, filled=bool(fill), **({k: fill[k] for k in ["asof", "qty", "price", "fee", "side"]} if fill else {}),
                  slippage_vs_decision_close=float(fill["price"] / plan["marks"][symbol] - 1) if fill else None),
        outcome=outcome,
        position=position,
    )


def portfolio_path(final_mark: dict, marks: dict, fresh_close: pd.DataFrame) -> dict:
    """The last recorded positions, marked forward with later public prices."""
    start = pd.Timestamp(day(final_mark["asof"]))
    path = fresh_close.loc[start:]
    ratio = path / path.iloc[0]
    value = sum(final_mark["positions"][s] * marks[s] * ratio[s] for s in SYMBOLS)
    equity = final_mark["cash"] + value
    return dict(
        provenance=RECONSTRUCTED,
        note=(
            "What the last recorded positions would be worth if simply held, using later public prices. "
            "This is not a recorded account state: the collector halted and never marked these sessions."
        ),
        dates=[day(d) for d in path.index], equity=[round(float(v), 4) for v in equity],
        latest=dict(session=day(path.index[-1]), equity=float(equity.iloc[-1]), pnl=float(equity.iloc[-1] - 10000), net_return=float(equity.iloc[-1] / 10000 - 1)),
    )


def build_case(case: dict, fresh_close: pd.DataFrame, outcome_rows: pd.DataFrame, sentiment) -> dict:
    from runtime_v2.model import features as price_features
    from runtime_v2.risk import limits

    item = receipts.load_decision(case)
    plan, result, manifest = item["plan"], item["result"], item["manifest"]
    session = plan["session"]
    frames, _ = receipts.load_snapshot(item["snapshot"], SYMBOLS, session)
    features = {s: price_features(frames[s], frames["SPY"]) for s in SYMBOLS}
    prices = pd.DataFrame({s: f.Close for s, f in frames.items()})
    raw = pd.Series({s: float(next(x for x in plan["signals"] if x["symbol"] == s)["trend"]) * 0.8 / len(SYMBOLS) for s in sorted(SYMBOLS)})
    trace = risk_trace(raw, prices.pct_change(fill_method=None))
    events = receipts.ledger_events(case["segment"])
    decision_id = plan["information_cutoff"]
    mark = next(e for e in events if e["kind"] == "mark" and e["asof"] == decision_id)
    sizing = order_trace(plan["shadow_weights"], plan["marks"], mark, limits())
    orders = [e for e in events if e["kind"] == "order" and e["decision_at"] == decision_id]
    fills = [e for e in events if e["kind"] == "fill" and e["order_id"].startswith(decision_id)]
    marks = [e for e in events if e["kind"] == "mark"]
    final_mark = marks[-1]
    final_plan = read_json(receipts.SEGMENTS[case["segment"]]["workspace"] / "days" / day(final_mark["asof"]) / "plan.json")
    status = receipts.segment_status(case["segment"])
    schedule = [
        dict(id=e["id"], kind=e["kind"], session=e["session"], status=e["status"], checked_at=e["checked_at"])
        for e in receipts.collector_events(case["segment"])
    ]
    symbols = {s: symbol_trace(s, item, frames, features, trace, sizing, events, outcome_rows, fresh_close, sentiment) for s in SYMBOLS}
    next_mark = next((m for m in marks if pd.Timestamp(m["asof"]) > pd.Timestamp(decision_id)), None)
    traded = [s for s in SYMBOLS if symbols[s]["order"]["created"]]
    return dict(
        schema=1, case_id=case["case_id"], segment=case["segment"], session=session,
        title=f"Recorded decision at the close of {pd.Timestamp(session).strftime('%-d %B %Y')} (US session)",
        verification=dict(
            receipt=rel(case["receipt"]), receipt_sha256=case["receipt_sha256"], receipt_matches_registration=True,
            plan_bound_to_result=True, snapshot_manifest_matches=True,
            probabilities_reproduced=True, max_abs_probability_difference=max(v["models"]["reconstructed"]["max_abs_difference_from_recorded"] for v in symbols.values()),
            trend_flags_reproduced=True, weights_reproduced=True, order_quantities_reproduced=True,
            registered_in=rel(receipts.R7_RELEASE / "protocol.json"),
        ),
        collection=dict(
            provenance=RECORDED, provider=manifest["provider"], retrieved_at=manifest["retrieved_at"], finished_at=manifest["finished_at"],
            convention=manifest["convention"], requests=result.get("provider_request_count"), symbols=len(manifest["data"]),
            information_cutoff=plan["information_cutoff"], prepared_at=plan["prepared_at"], completed_at=result.get("completed_at"),
            trigger=plan.get("trigger_mode"), mode=plan.get("mode"), rebalance=plan["rebalance"], refit=plan["refit"],
            timings_seconds=plan.get("timings_seconds") or plan.get("timings"), peak_rss_bytes=result.get("peak_rss_bytes"),
            snapshot=rel(item["snapshot"]),
        ),
        regime=regime_at(frames, session),
        rule=dict(
            provenance=RECORDED,
            text="Hold an instrument when its price is above its 200-session average and its 12-month return (skipping the last month) is positive. Each held instrument gets 0.8 / 22 of equity.",
            trend_count=int(sum(v["trend"] for v in symbols.values())), raw_gross=float(raw.sum()), final_gross=float(sum(plan["shadow_weights"].values())),
            not_used=["classifier probabilities (recorded as diagnostics only)", "the model gate flag", "the HMM regime", "news sentiment"],
        ),
        risk=dict(provenance=RECONSTRUCTED, steps=trace["steps"],
                  account=dict(equity=sizing["equity"], wanted_turnover=sizing["turnover"], wanted_turnover_share=sizing["turnover_share"],
                               turnover_limit=sizing["turnover_limit"], scale=sizing["scale"], binding=sizing["binding"],
                               reserve_buffer=sizing["reserve_buffer"], cost_rate=sizing["cost_rate"], reserved_cash=mark["reserved_cash"],
                               drawdown=mark["drawdown"], drawdown_limit=limits().drawdown_limit,
                               drawdown_stop_triggered=any(e["kind"] == "drawdown_halt" and e.get("asof") == decision_id for e in events))),
        orders=dict(provenance=RECORDED, count=len(orders), buys=sum(o["side"] == "buy" for o in orders), sells=sum(o["side"] == "sell" for o in orders),
                    reserved_cash=mark["reserved_cash"], deployment_account="cash: zero target weights, no orders", external_orders=False),
        fills=dict(
            provenance=RECORDED, count=len(fills), session=day(fills[0]["asof"]) if fills else None,
            fees=float(sum(f["fee"] for f in fills)), notional=float(sum(f["qty"] * f["price"] for f in fills)),
            reason_if_none=None if fills else (
                (
                    f"No fill was recorded: the collector produced no receipt for the next session "
                    f"({min(s for s in status['status']['missing_sessions'] if s > session)}), so these orders stayed pending. "
                    if any(s > session for s in status["status"].get("missing_sessions", []))
                    else "No fill was recorded: this pilot segment ended on the decision date and no later close was marked, so these orders stayed pending. "
                )
                + "Nothing was backfilled."
            ),
        ),
        account=dict(
            provenance=RECORDED,
            at_decision={k: mark[k] for k in ["asof", "equity", "cash", "reserved_cash", "fees", "gross_exposure"]},
            after_fill={k: next_mark[k] for k in ["asof", "equity", "cash", "reserved_cash", "fees", "gross_exposure", "unrealized_pnl", "realized_pnl"]} if next_mark else None,
            last_recorded={k: final_mark[k] for k in ["asof", "equity", "cash", "fees", "gross_exposure", "unrealized_pnl", "realized_pnl"]},
            marks=[{k: m[k] for k in ["asof", "equity", "cash", "fees", "gross_exposure"]} for m in marks],
        ),
        after=dict(
            provenance=RECORDED, halt=status["halt"], missing_sessions=status["status"].get("missing_sessions", []),
            valid_observed_sessions=status["status"].get("valid_observed_sessions"), required_sessions=status["status"].get("required_sessions"),
            schedule=schedule,
        ),
        held_forward=portfolio_path(final_mark, final_plan["marks"], fresh_close),
        traded_symbols=traded,
        symbols=symbols,
    )


def build() -> dict:
    reg = registration.require()
    if not receipts.available():
        raise RuntimeError("Recorded receipts are not present on this machine; the committed walkthrough files remain valid")
    DIR.mkdir(parents=True, exist_ok=True)
    frames, meta = fresh.load(fresh.snapshot())
    fresh_close = regime_report.naive(pd.DataFrame({s: frames[s].Close for s in SYMBOLS}))
    outcome_rows = receipts.recorded_outcomes()
    sentiment_path = OUT / "sentiment/decision_dates.csv"
    sentiment = pd.read_csv(sentiment_path) if sentiment_path.exists() else None
    cases = []
    for case in receipts.registered_decisions():
        built = build_case(case, fresh_close, outcome_rows, sentiment)
        built["registration_sha256"] = reg["sha256"]
        built["later_prices"] = dict(provenance=RECONSTRUCTED, provider=meta["provider"], retrieved_at=meta["retrieved_at"], last_session=meta["expected_session"])
        write_json(DIR / f"{case['case_id']}.json", built)
        cases.append(dict(case_id=case["case_id"], segment=case["segment"], session=case["session"], symbols=SYMBOLS,
                          traded_symbols=built["traded_symbols"], filled=built["fills"]["count"] > 0, rebalance=built["collection"]["rebalance"],
                          title=built["title"], file=f"{case['case_id']}.json"))
    index = dict(schema=1, default_case=nav.DEFAULT_CASE, default_symbol=nav.DEFAULT_SYMBOL, cases=cases, steps=nav.STEPS)
    write_json(DIR / "index.json", index)
    return index


def load_index() -> dict:
    return read_json(DIR / "index.json")


def load_case(case_id: str) -> dict:
    return read_json(DIR / f"{case_id}.json")


if __name__ == "__main__":
    out = build()
    for c in out["cases"]:
        print(c["case_id"], len(c["traded_symbols"]), c["filled"], c["rebalance"])
    case = load_case(nav.DEFAULT_CASE)
    nvda = case["symbols"]["NVDA"]
    print(case["verification"])
    print(case["risk"])
    print({k: nvda[k] for k in ["action", "reason", "weight", "order", "fill", "outcome"]})
    print(nvda["position"]["latest"], case["held_forward"]["latest"], case["regime"]["p_turbulent"])
