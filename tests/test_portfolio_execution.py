import numpy as np
import pandas as pd
import pytest
from src.execution.account import Account, Limits, ExecutionSession, combine_targets


def account(path=None, **kwargs):
    return Account(["A", "B"], 10000, Limits(max_position=0.5, **kwargs), path)


def test_shared_cash_pending_reservations_and_repeated_decision():
    a = account()
    prices = {"A": 100.0, "B": 100.0}
    weights = {"A": 0.4, "B": 0.4}
    ids = a.rebalance("day1", "2024-01-02", weights, prices)
    assert len(ids) == 2
    assert a.reserved_cash() < a.cash
    assert a.rebalance("day1", "2024-01-02", weights, prices) == []
    with pytest.raises(ValueError, match="reused"):
        a.rebalance("day1", "2024-01-02", {"A": 0.3, "B": 0.4}, prices)
    for oid in ids:
        a.fill(oid, a.orders[oid]["qty"], 100, "2024-01-03", prices, oid + "fill")
    assert a.snapshot(prices)["gross_exposure"] <= 0.8 + 1e-6
    assert a.cash >= 0


def test_long_entry_partial_fill_restart_hold_reduce_exit(tmp_path):
    path = tmp_path / "account.db"
    a = account(path)
    prices = {"A": 100.0, "B": 100.0}
    ids = a.rebalance("one", "2024-01-02", {"A": 0.3, "B": 0.0}, prices)
    a.fill(ids[0], 10.0, 100.0, "2024-01-03", prices, "partial")
    a = account(path)
    assert a.positions["A"] == 10.0
    a.fill(ids[0], 10.0, 100.0, "2024-01-03", prices, "partial")
    assert a.positions["A"] == 10.0
    a.fill(ids[0], 20.0, 100.0, "2024-01-03", prices, "remaining")
    session = ExecutionSession(a)
    session.on_close("2024-01-04", prices, {"A": 0.1, "B": 0.0})
    session.on_close("2024-01-05", prices, {"A": 0.0, "B": 0.0})
    state = session.on_close("2024-01-08", prices, {"A": 0.0, "B": 0.0})
    assert state["positions"] == {"A": 0.0, "B": 0.0}
    assert state["equity"] == pytest.approx(10000 - a.fees)
    assert a.realized == pytest.approx(-a.fees)
    assert state["reconciliation_error"] == pytest.approx(0.0)


def test_fill_at_decision_time_and_gaps_fail_closed():
    a = account()
    prices = {"A": 100.0, "B": 100.0}
    oid = a.rebalance("one", "2024-01-02", {"A": 0.3, "B": 0.0}, prices)[0]
    with pytest.raises(ValueError, match="cutoff"):
        a.fill(oid, 30.0, 100.0, "2024-01-02", prices, "bad")
    a.fill(oid, 30.0, 110.0, "2024-01-03", {"A": 110.0, "B": 100.0}, "gap")
    assert a.orders[oid]["status"] == "rejected"
    assert a.positions["A"] == 0 and a.cash == 10000


def test_short_leverage_concentration_and_correlated_group_limits():
    a = account(groups={"A": "tech", "B": "tech"}, group_limits={"tech": 0.5})
    prices = {"A": 100.0, "B": 100.0}
    for weights in [{"A": -0.1, "B": 0.0}, {"A": 0.6, "B": 0.0}, {"A": 0.4, "B": 0.4}]:
        with pytest.raises(ValueError):
            a.rebalance("one", "2024-01-02", weights, prices)
    assert a.orders == {}


def test_pending_cash_cannot_be_spent_twice():
    a = account()
    p = {"A": 100.0, "B": 100.0}
    a.rebalance("one", "2024-01-02", {"A": 0.4, "B": 0.4}, p)
    assert a.reserved_cash() <= a.cash
    # A fresh decision replaces unfilled intent, so it cannot stack exposure.
    a.rebalance("two", "2024-01-03", {"A": 0.4, "B": 0.4}, p)
    assert len(a.pending()) == 2 and a.reserved_cash() <= a.cash


def test_replay_and_paper_restart_are_identical(tmp_path):
    research = ExecutionSession(account())
    paper = ExecutionSession(account(tmp_path / "state.db"))
    rng = np.random.default_rng(8)
    for i in range(60):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
        prices = {"A": float(100 + rng.normal()), "B": float(80 + rng.normal())}
        weights = {"A": 0.3 if i % 7 < 4 else 0.0, "B": 0.2 if i % 5 < 2 else 0.0}
        expected = research.on_close(date, prices, weights)
        actual = paper.on_close(date, prices, weights)
        assert actual == expected
        # Retry followed by process restart leaves exposure unchanged.
        assert paper.on_close(date, prices, weights) == actual
        paper = ExecutionSession(account(tmp_path / "state.db"))
    assert paper.account.events == research.account.events


def test_configuration_and_concurrent_writer_fail_closed(tmp_path):
    path = tmp_path / "state.db"
    a = account(path)
    stale = account(path)
    p = {"A": 100.0, "B": 100.0}
    w = {"A": 0.1, "B": 0.0}
    a.rebalance("a", "2024-01-02", w, p)
    with pytest.raises(RuntimeError, match="Concurrent"):
        stale.rebalance("b", "2024-01-02", w, p)
    with pytest.raises(ValueError, match="configuration"):
        Account(["A"], 10000, Limits(), path)


def test_conflicts_net_against_strategy_budgets():
    targets = combine_targets(
        {"trend": {"A": 1}, "defensive": {"A": 0}},
        {"trend": 0.3, "defensive": 0.7},
        ["A"],
    )
    assert targets == {"A": 0.3}


def test_actual_live_engine_matches_research_handler(monkeypatch, tmp_path):
    from src.live.signal_generator import LiveSignal
    from src.live.engine import LiveEngine
    import src.live.engine as live
    from unittest.mock import MagicMock

    gen = MagicMock()
    monkeypatch.setattr(live, "SignalGenerator", lambda **kw: gen)
    monkeypatch.setattr(
        live,
        "AlpacaBroker",
        lambda *a, **kw: pytest.fail("Broker must remain isolated"),
    )
    engine = LiveEngine(
        tickers=["A", "B"],
        max_position_frac=0.5,
        dry_run=False,
        state_path=tmp_path / "paper.db",
    )
    research = ExecutionSession(
        Account(["A", "B"], 10000, Limits(max_position=0.5, drawdown_limit=0.15))
    )
    for i in range(10):
        date = pd.Timestamp("2024-01-02", tz="UTC") + pd.Timedelta(days=i)
        price = {"A": 100.0 + i, "B": 90.0 + i}
        directions = {"A": 2 if i < 5 else 1, "B": 2 if i < 7 else 0}
        gen.generate_batch.return_value = {
            s: LiveSignal(
                ticker=s,
                date=date,
                direction=directions[s],
                confidence=0.7,
                p_up=0.7,
                p_flat=0.2,
                p_down=0.1,
                kelly_frac=0.3,
                atr=1.0,
                stop_loss=80.0,
                close=price[s],
                size_cap=0.5 if i == 4 else 1.0,
            )
            for s in price
        }
        expected = research.on_close(
            date, price, engine.target_weights(gen.generate_batch.return_value)
        )
        result = engine.run()
        assert not result.errors
        assert result.snapshot == expected


def test_invalid_close_rolls_back_pending_fills(tmp_path):
    path = tmp_path / "atomic.db"
    a = account(path)
    session = ExecutionSession(a)
    p = {"A": 100.0, "B": 100.0}
    session.on_close("2024-01-02", p, {"A": 0.3, "B": 0.0})
    before = a.snapshot(p)
    events = list(a.events)
    with pytest.raises(ValueError):
        session.on_close("2024-01-03", p, {"A": 0.9, "B": 0.9})
    assert a.snapshot(p) == before and a.events == events
    assert account(path).snapshot(p) == before
