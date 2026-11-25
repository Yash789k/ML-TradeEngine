"""The sandbox harness: ledger arithmetic by hand, causality, the registered decision rule, sealed-account parity."""

import numpy as np
import pandas as pd
import pytest

from review_v1 import sandbox
from review_v1.paths import OUT

SYMBOLS = list("ABCDEFGHIJ")


def panel(sessions=12, price=100.0, weight=0.08):
    index = pd.bdate_range("2024-01-02", periods=sessions, tz="UTC")
    prices = pd.DataFrame(price, index=index, columns=SYMBOLS)
    targets = pd.DataFrame(weight, index=index, columns=SYMBOLS)
    return prices, targets


def test_first_purchase_matches_hand_arithmetic():
    """80% wanted, 40% turnover cap, fill at the next close, 15 bp cost: the same shape as the recorded pilot fills."""
    prices, targets = panel()
    rebalance = np.zeros(len(prices), bool)
    rebalance[0] = True
    history, events = sandbox.run_account(prices, targets, rebalance)

    orders = [e for e in events if e["kind"] == "order"]
    fills = [e for e in events if e["kind"] == "fill"]
    assert len(orders) == len(fills) == 10
    # Wanted 0.08 x 10,000 = 800 per instrument; the cap halves it: 400 / 100 = 4 units, reserve price 103.
    assert all(o["qty"] == pytest.approx(4.0) and o["limit_price"] == pytest.approx(103.0) for o in orders)

    first, second = history.iloc[0], history.iloc[1]
    assert first.equity == pytest.approx(10000.0) and first.gross_exposure == 0  # nothing fills on the decision close
    assert first.reserved_cash == pytest.approx(10 * 4 * 103.0 * 1.0015)
    assert second.fees == pytest.approx(4000 * 0.0015)
    assert second.equity == pytest.approx(10000 - 6.0)
    assert second.cash == pytest.approx(10000 - 4000 - 6.0)
    assert second.gross_exposure == pytest.approx(4000 / 9994.0)
    assert all(pd.Timestamp(f["asof"]) == prices.index[1] and f["price"] == 100.0 for f in fills)
    # No further rebalance was scheduled, so nothing else may change.
    assert (history.equity.iloc[1:] == second.equity).all()

    report = sandbox.summarise(history, events)
    assert report["fills"] == 10 and report["orders"] == 10 and report["rejections"] == 0
    assert report["pnl_usd"] == pytest.approx(-6.0)
    assert report["combined_fees_usd"] == pytest.approx(6.0)
    assert report["sessions"] == 12 and abs(report["max_reconciliation_error"]) < 1e-6
    assert "far too few to interpret a Sharpe ratio" in report["sharpe_interpretation"]


def test_a_fill_that_would_breach_the_exposure_cap_is_rejected_not_forced():
    prices, targets = panel()
    rebalance = np.zeros(len(prices), bool)
    rebalance[[0, 5]] = True
    history, events = sandbox.run_account(prices, targets, rebalance)
    report = sandbox.summarise(history, events)
    # The second rebalance asks for exactly 80% gross; costs have reduced equity, so the last buy cannot fill.
    assert report["rejections"] == 1 and report["fills"] == 19
    assert history.gross_exposure.max() <= 0.8 + 1e-9
    assert (history.cash >= -1e-9).all()


def test_cash_portfolio_never_trades():
    prices, targets = panel(weight=0.0)
    history, events = sandbox.run_account(prices, targets, np.ones(len(prices), bool))
    assert (history.equity == 10000.0).all()
    assert not [e for e in events if e["kind"] in ("order", "fill")]


def test_account_path_does_not_depend_on_later_prices():
    rng = np.random.default_rng(5)
    prices, targets = panel(sessions=40)
    prices = prices * np.exp(np.cumsum(rng.normal(0, 0.01, prices.shape), axis=0))
    rebalance = np.arange(40) % 5 == 0
    base, _ = sandbox.run_account(prices, targets, rebalance)
    changed = prices.copy()
    changed.iloc[25:] *= 1.5
    other, _ = sandbox.run_account(changed, targets, rebalance)
    pd.testing.assert_frame_equal(base.iloc[:25], other.iloc[:25])
    assert not np.allclose(base.equity.iloc[25:], other.equity.iloc[25:])


@pytest.mark.parametrize("fault", ["nan_price", "zero_price", "negative_target", "nan_target", "other_sessions", "other_columns"])
def test_invalid_panels_are_refused(fault):
    prices, targets = panel()
    if fault == "nan_price":
        prices.iloc[3, 2] = np.nan
    elif fault == "zero_price":
        prices.iloc[3, 2] = 0.0
    elif fault == "negative_target":
        targets.iloc[3, 2] = -0.01
    elif fault == "nan_target":
        targets.iloc[3, 2] = np.nan
    elif fault == "other_sessions":
        targets = targets.iloc[1:]
    else:
        targets = targets[SYMBOLS[::-1]]
    with pytest.raises(ValueError):
        sandbox.run_account(prices, targets, np.ones(len(prices), bool))


def test_daily_returns_start_from_the_initial_capital():
    history = pd.DataFrame(dict(equity=[9990.0, 10089.9, 10089.9]))
    returns = sandbox.daily_returns(history)
    assert returns.tolist() == pytest.approx([-0.001, 0.01, 0.0])


def test_sharpe_difference_is_paired_antisymmetric_and_reproducible():
    rng = np.random.default_rng(2)
    a = rng.normal(0.0008, 0.01, 600)
    b = 0.6 * a + rng.normal(0.0, 0.004, 600)
    forward, backward = sandbox.sharpe_difference(a, b), sandbox.sharpe_difference(b, a)
    assert forward == sandbox.sharpe_difference(a, b)
    assert forward["effect"] == pytest.approx(-backward["effect"])
    assert forward["ci_low"] == pytest.approx(-backward["ci_high"])
    assert forward["ci_low"] < forward["effect"] < forward["ci_high"]
    same = sandbox.sharpe_difference(a, a)
    assert same["effect"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0 and same["p_two_sided"] == 1
    with pytest.raises(ValueError):
        sandbox.sharpe_difference(a, b[:-1])


def test_registered_decision_rule_only_supports_a_positive_significant_overlay():
    """An overlay that is reliably worse must read 'not established', never 'supported'."""
    rng = np.random.default_rng(9)
    parent = rng.normal(0.0005, 0.01, 700)
    noise = rng.normal(0.0, 0.0005, 700)
    better = dict(trend=parent, trend_regime=parent + 0.0006 + noise, equal_weight=parent, equal_weight_regime=parent - 0.0006 + noise)
    frame = sandbox.contrasts(better).set_index("arm")
    assert frame.loc["trend_regime", "sharpe_difference"] > 0 and frame.loc["trend_regime", "sharpe_holm_p"] < 0.05
    assert frame.loc["trend_regime", "verdict"] == "supported on this history"
    assert frame.loc["equal_weight_regime", "sharpe_difference"] < 0 and frame.loc["equal_weight_regime", "sharpe_holm_p"] < 0.05
    assert frame.loc["equal_weight_regime", "verdict"] == "not established"

    nothing = dict(trend=parent, trend_regime=parent + noise, equal_weight=parent, equal_weight_regime=parent - noise)
    assert set(sandbox.contrasts(nothing).verdict) == {"not established"}
    assert (frame.sharpe_holm_p >= frame.sharpe_p - 1e-15).all()


def test_saved_results_keep_the_registered_arms_and_passed_their_parity_checks():
    if not (OUT / "sandbox/summary.json").exists():
        pytest.skip("sandbox outputs have not been built")
    import json

    summary = json.loads((OUT / "sandbox/summary.json").read_text())
    metrics = pd.read_csv(OUT / "sandbox/metrics.csv")
    contrasts = pd.read_csv(OUT / "sandbox/contrasts.csv")
    assert summary["history"]["parity"]["passed"] and summary["post_release"]["parity"]["passed"]
    assert set(metrics.loc[metrics.period == "history", "arm"]) == {"cash", "equal_weight", "trend", "trend_regime", "equal_weight_regime"}
    assert sorted(zip(contrasts.arm, contrasts.parent, strict=True)) == [("equal_weight_regime", "equal_weight"), ("trend_regime", "trend")]
    # The decision rule, re-applied to the saved numbers, gives the saved verdicts.
    expected = np.where((contrasts.sharpe_difference > 0) & (contrasts.sharpe_holm_p < 0.05), "supported on this history", "not established")
    assert list(contrasts.verdict) == list(expected)
    # A two-month window must not carry an interpretable Sharpe ratio.
    post = metrics.loc[metrics.period == "post_release"].set_index("arm")
    assert post.drop(index="cash").sharpe_interpretation.str.contains("far too few").all()
    assert post.loc["cash", "sharpe_interpretation"].startswith("Undefined")


def test_harness_reproduces_the_sealed_trend_account():
    """Recomputes the sealed September trend account from its targets and prices (needs the local release)."""
    if not (sandbox.EXP / "trend_account.parquet").exists() or not (OUT / "regime/posterior_history.csv").exists():
        pytest.skip("sealed adaptive release is local evidence and is not present")
    prices, targets, rebalance, _ = sandbox.history_panels()
    history, _ = sandbox.run_account(prices, targets["trend"], rebalance)
    sealed = pd.read_parquet(sandbox.EXP / "trend_account.parquet")
    assert float((sealed.equity - history.equity).abs().max()) < 1e-6
    assert float((sealed.fees - history.fees).abs().max()) < 1e-6
