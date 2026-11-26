"""The trade walkthrough: provenance labels, internal arithmetic of the saved cases, and the hash-checked replay.

The first group runs on the committed walkthrough files. The second group needs the
recorded pilot receipts, which are local evidence and are skipped when absent.
"""

import math

import pytest

from review_v1 import receipts, walkthrough
from review_v1.paths import CLASSES, PROVENANCE, RECONSTRUCTED, RECORDED, SIMULATED

COST = 0.0015       # commission 10 bp + slippage 5 bp
BUFFER = 0.03       # reserve price is the decision close plus 3%
THRESHOLD = 0.01    # label threshold on the forward return


def cases():
    if not (walkthrough.DIR / "index.json").exists():
        return []
    return [walkthrough.load_case(c["case_id"]) for c in walkthrough.load_index()["cases"]]


CASES = cases()
needs_cases = pytest.mark.skipif(not CASES, reason="walkthrough files have not been built")
needs_receipts = pytest.mark.skipif(not receipts.available(), reason="recorded pilot receipts are local evidence and are not present")


def labels(node, path=""):
    """Every provenance label in a saved case, with where it sits."""
    if isinstance(node, dict):
        if "provenance" in node:
            yield path, node["provenance"]
        for key, value in node.items():
            yield from labels(value, f"{path}/{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from labels(value, f"{path}/{i}")


@needs_cases
def test_index_lists_the_five_registered_decisions_and_a_filled_default():
    index = walkthrough.load_index()
    assert len(index["cases"]) == 5
    assert [s["id"] for s in index["steps"]] == ["inputs", "features", "models", "decision", "order", "fill", "outcome"]
    assert [s["n"] for s in index["steps"]] == list(range(1, 8))
    assert {s["provenance"] for s in index["steps"]} <= set(PROVENANCE)
    default = next(c for c in index["cases"] if c["case_id"] == index["default_case"])
    assert default["filled"] and index["default_symbol"] in default["traded_symbols"]


@needs_cases
@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_every_section_is_labelled_and_nothing_recomputed_is_called_recorded(case):
    found = dict(labels(case))
    assert set(found.values()) <= set(PROVENANCE)
    # A recorded pilot decision contains no simulated event at all.
    assert SIMULATED not in found.values()
    for section in ["collection", "rule", "orders", "fills", "account", "after"]:
        assert case[section]["provenance"] == RECORDED, section
    # Anything computed for this review, or from prices downloaded later, must say so.
    for section in ["regime", "risk", "held_forward", "later_prices"]:
        assert case[section]["provenance"] == RECONSTRUCTED, section
    for symbol, item in case["symbols"].items():
        for section in ["inputs", "models", "decision", "order", "fill"]:
            assert item[section]["provenance"] == RECORDED, (symbol, section)
        assert item["features"]["provenance"] == RECONSTRUCTED
        assert item["models"]["reconstructed"]["provenance"] == RECONSTRUCTED
        assert item["decision"]["sizing"]["provenance"] == RECONSTRUCTED
        if item["inputs"]["sentiment"]["available"]:
            assert item["inputs"]["sentiment"]["provenance"] == RECONSTRUCTED
        if item["position"]["opened"]:
            assert item["position"]["provenance"] == RECONSTRUCTED
        if item["outcome"]["available"]:
            recorded = item["outcome"]["source"].startswith("outcome read recorded")
            assert item["outcome"]["provenance"] == (RECORDED if recorded else RECONSTRUCTED)


@needs_cases
@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_nothing_but_the_trend_rule_and_the_risk_limits_sets_a_weight(case):
    symbols = case["symbols"]
    assert len(symbols) == 22
    assert case["rule"]["trend_count"] == sum(item["trend"] for item in symbols.values())
    assert case["rule"]["raw_gross"] == pytest.approx(case["rule"]["trend_count"] * 0.8 / 22)
    assert case["rule"]["final_gross"] <= case["rule"]["raw_gross"] + 1e-12
    assert case["regime"]["role"].startswith("Diagnostic")
    for symbol, item in symbols.items():
        inputs = item["decision"]["trend_inputs"]
        assert item["trend"] == (inputs["average_200"] > 0 and inputs["momentum_12_1"] > 0), symbol
        assert item["decision"]["raw_weight"] == pytest.approx(0.8 / 22 if item["trend"] else 0.0)
        assert 0 <= item["decision"]["final_weight"] <= item["decision"]["raw_weight"] + 1e-12
        # The classifier is recorded beside the decision and never feeds it.
        assert item["models"]["used_in_decision"] is False
        assert sum(item["models"]["recorded"]["probabilities"]) == pytest.approx(1.0)
        assert item["models"]["reconstructed"]["max_abs_difference_from_recorded"] <= 1e-12
        assert len(item["features"]["rows"]) == 17
        assert {r["name"] for r in item["features"]["rows"] if r["used_by_trend_rule"]} == {"average_200", "momentum_12_1"}
        if item["inputs"]["sentiment"]["available"]:
            assert item["inputs"]["sentiment"]["role"].startswith("Diagnostic")


@needs_cases
@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_orders_follow_from_the_recorded_account_by_arithmetic(case):
    account = case["risk"]["account"]
    assert account["wanted_turnover_share"] == pytest.approx(account["wanted_turnover"] / account["equity"])
    assert account["scale"] == pytest.approx(min(1.0, account["turnover_limit"] * account["equity"] / account["wanted_turnover"]) if account["wanted_turnover"] else 1.0)
    assert account["binding"] == (account["scale"] < 1)
    assert account["cost_rate"] == pytest.approx(COST) and account["reserve_buffer"] == pytest.approx(BUFFER)
    assert account["drawdown_stop_triggered"] == (account["drawdown"] <= -account["drawdown_limit"])
    created = [item for item in case["symbols"].values() if item["order"]["created"]]
    assert len(created) == case["orders"]["count"] == case["orders"]["buys"] + case["orders"]["sells"]
    assert sorted(case["traded_symbols"]) == sorted(item["symbol"] for item in created)
    assert case["orders"]["external_orders"] is False
    if not case["collection"]["rebalance"]:
        assert not created
    for item in created:
        order, sizing = item["order"], item["decision"]["sizing"]
        assert item["action"] == order["side"].upper()
        assert order["decision_at"] == case["collection"]["information_cutoff"]
        assert sizing["scaled_trade_value"] == pytest.approx(sizing["wanted_trade_value"] * account["scale"])
        assert sizing["wanted_trade_value"] == pytest.approx(sizing["target_value"] - sizing["held_value"])
        assert order["qty"] == pytest.approx(math.floor(sizing["quantity"] / 1e-6) * 1e-6, abs=2e-6)
        assert order["limit_price"] == pytest.approx(sizing["limit_price"])
        if order["side"] == "buy":
            assert order["reserved_cash"] == pytest.approx(order["qty"] * order["limit_price"] * (1 + COST))
    assert sum(item["order"]["reserved_cash"] for item in created) == pytest.approx(case["orders"]["reserved_cash"])
    for item in case["symbols"].values():
        if not item["order"]["created"]:
            assert item["action"] in ("HOLD", "NO TRADE")


@needs_cases
@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_fills_happen_only_at_a_later_close_and_costs_add_up(case):
    filled = [item for item in case["symbols"].values() if item["fill"]["filled"]]
    assert len(filled) == case["fills"]["count"]
    if not filled:
        # No fill is ever invented: the reason is stated and no position is shown.
        assert case["fills"]["reason_if_none"] and "Nothing was backfilled" in case["fills"]["reason_if_none"]
        assert case["account"]["after_fill"] is None or case["account"]["after_fill"]["fees"] == case["account"]["at_decision"]["fees"]
        assert not any(item["position"]["opened"] for item in case["symbols"].values())
        return
    for item in filled:
        order, fill = item["order"], item["fill"]
        assert order["created"] and fill["side"] == order["side"]
        assert fill["asof"] > order["decision_at"]           # never on the decision close
        assert fill["asof"][:10] == case["fills"]["session"]
        assert fill["qty"] == pytest.approx(order["qty"])
        assert fill["fee"] == pytest.approx(fill["qty"] * fill["price"] * COST)
        if fill["side"] == "buy":
            assert fill["price"] <= order["limit_price"]
            position = item["position"]
            assert position["opened"] and position["quantity"] == fill["qty"] and position["fill_price"] == fill["price"]
            assert position["cost_basis"] == pytest.approx(fill["qty"] * fill["price"] + fill["fee"])
            assert position["value"][0] == pytest.approx(fill["qty"] * fill["price"], abs=1e-3)
            assert position["pnl"][0] == pytest.approx(-fill["fee"], abs=1e-3)   # on the fill date the only loss is the cost
            assert position["dates"][0] == case["fills"]["session"]
    assert case["fills"]["fees"] == pytest.approx(sum(item["fill"]["fee"] for item in filled))
    assert case["fills"]["notional"] == pytest.approx(sum(item["fill"]["qty"] * item["fill"]["price"] for item in filled))
    before, after = case["account"]["at_decision"], case["account"]["after_fill"]
    assert after["asof"] > before["asof"]
    assert after["fees"] - before["fees"] == pytest.approx(case["fills"]["fees"])
    assert after["cash"] >= -1e-9 and after["gross_exposure"] <= 0.8 + 1e-9


@needs_cases
@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_outcomes_use_the_registered_label_and_score(case):
    for symbol, item in case["symbols"].items():
        outcome = item["outcome"]
        if not outcome["available"]:
            continue
        forward = outcome["forward_return"]
        assert forward == pytest.approx(outcome["exit_price"] / outcome["entry_price"] - 1), symbol
        assert outcome["entry_session"] > case["session"]            # entry is the next close, never the decision close
        assert outcome["label"] == ("UP" if forward > THRESHOLD else "DOWN" if forward < -THRESHOLD else "FLAT"), symbol
        probabilities = item["models"]["recorded"]["probabilities"]
        index = CLASSES.index(outcome["label"])
        assert outcome["predicted"] == item["models"]["recorded"]["top_class"]
        assert outcome["correct"] == (outcome["predicted"] == outcome["label"])
        assert outcome["hybrid_log_loss"] == pytest.approx(-math.log(probabilities[index]))
        assert outcome["prior_log_loss"] == pytest.approx(-math.log(item["models"]["reconstructed"]["prior"][index]))


@needs_cases
def test_held_forward_path_is_marked_as_a_reconstruction_not_a_recorded_account():
    for case in CASES:
        path = case["held_forward"]
        assert "not a recorded account state" in path["note"]
        assert len(path["dates"]) == len(path["equity"]) and path["dates"] == sorted(path["dates"])
        assert path["dates"][0] >= case["account"]["last_recorded"]["asof"][:10]
        assert path["latest"]["pnl"] == pytest.approx(path["latest"]["equity"] - 10000)


# --------------------------------------------------------------------------
# With the recorded receipts present
# --------------------------------------------------------------------------

@needs_receipts
def test_registered_receipts_are_unchanged_and_bind_to_their_plans():
    registered = receipts.registered_decisions()
    assert [c["case_id"] for c in registered] == [c["case_id"] for c in walkthrough.load_index()["cases"]]
    for case in registered:
        item = receipts.load_decision(case)   # raises on any hash mismatch
        assert item["plan"]["session"] == case["session"]
        saved = walkthrough.load_case(case["case_id"])
        assert saved["verification"]["receipt_sha256"] == case["receipt_sha256"]


@needs_receipts
@pytest.mark.parametrize("field", ["receipt_sha256", "content_sha256"])
def test_a_receipt_that_does_not_match_its_registration_is_refused(field):
    case = dict(receipts.registered_decisions()[0])
    case[field] = "0" * 64
    with pytest.raises(ValueError):
        receipts.load_decision(case)


@needs_receipts
def test_an_edited_receipt_is_refused(tmp_path):
    import json
    import shutil

    case = dict(receipts.registered_decisions()[0])
    day = tmp_path / case["receipt"].parent.name
    shutil.copytree(case["receipt"].parent, day)
    copy = day / case["receipt"].name
    body = json.loads(copy.read_text())
    body["completed_at"] = "2026-10-01T23:59:59+00:00"
    copy.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="changed since registration"):
        receipts.load_decision(case | dict(receipt=copy))


@needs_receipts
def test_saved_orders_and_fills_equal_the_recorded_ledger():
    """The dashboard's order and fill numbers are the ledger's own events, read again from the read-only database."""
    for case in CASES:
        events = receipts.ledger_events(case["segment"])
        cutoff = case["collection"]["information_cutoff"]
        orders = {e["symbol"]: e for e in events if e["kind"] == "order" and e["decision_at"] == cutoff}
        fills = {e["order_id"]: e for e in events if e["kind"] == "fill" and e["order_id"].startswith(cutoff)}
        assert len(orders) == case["orders"]["count"] and len(fills) == case["fills"]["count"]
        for symbol, item in case["symbols"].items():
            assert item["order"]["created"] == (symbol in orders)
            if symbol in orders:
                assert item["order"]["qty"] == orders[symbol]["qty"] and item["order"]["limit_price"] == orders[symbol]["limit_price"]
                fill = fills.get(orders[symbol]["id"])
                assert item["fill"]["filled"] == (fill is not None)
                if fill:
                    assert (item["fill"]["qty"], item["fill"]["price"], item["fill"]["fee"]) == (fill["qty"], fill["price"], fill["fee"])
        mark = next(e for e in events if e["kind"] == "mark" and e["asof"] == cutoff)
        assert case["account"]["at_decision"]["equity"] == mark["equity"]
        assert case["risk"]["account"]["drawdown"] == mark["drawdown"]
