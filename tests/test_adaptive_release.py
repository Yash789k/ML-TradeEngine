import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from src.models.adaptive import (
    AdaptiveConfig,
    AdaptiveModel,
    features,
    execution_labels,
)
from src.execution.account import Account, ExecutionSession, Limits
from src.research.adaptive_portfolio import constrain_weights


def market(n=1150):
    rng = np.random.default_rng(13)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.013, n)))
    return pd.DataFrame(
        dict(
            Open=c,
            High=c * 1.01,
            Low=c * 0.99,
            Close=c,
            Volume=rng.uniform(1000, 2000, n),
        ),
        index=pd.bdate_range("2020-01-01", periods=n, tz="UTC"),
    )


def test_labels_start_at_execution_close_and_mask_unavailable_tail():
    c = pd.Series(
        [100, 1000, 10, 20, 30, 40, 50, 60],
        index=pd.bdate_range("2024-01-01", periods=8),
    )
    y, r = execution_labels(c)
    assert r.iloc[0] == pytest.approx(50 / 1000 - 1)
    assert y.iloc[0] == 0
    assert (y.iloc[-6:] == -1).all()


def test_future_perturbations_leave_features_fit_and_prediction_unchanged(tmp_path):
    raw = market()
    changed = raw.copy()
    decision = raw.index[1000]
    changed.loc[changed.index > decision, ["Open", "High", "Low", "Close"]] *= 9
    x, z = features(raw, raw), features(changed, changed)
    pd.testing.assert_frame_equal(x.loc[:decision], z.loc[:decision])
    with threadpool_limits(limits=1):
        m = AdaptiveModel().fit_at(x, raw.Close, decision)
        n = AdaptiveModel().fit_at(z, changed.Close, decision)
        assert m.metadata == n.metadata
        assert pd.Timestamp(m.metadata["latest_label_end"]) < decision
        assert pd.Timestamp(m.metadata["fit_label_end"]) < pd.Timestamp(
            m.metadata["calibration_start"]
        )
        np.testing.assert_allclose(
            m.predict(x.loc[[decision]])["hybrid"],
            n.predict(z.loc[[decision]])["hybrid"],
        )
        m.save(tmp_path / "bundle")
        saved = AdaptiveModel.load(tmp_path / "bundle")
        np.testing.assert_array_equal(
            m.predict(x.loc[[decision]])["hybrid"],
            saved.predict(x.loc[[decision]])["hybrid"],
        )
        with pytest.raises(ValueError, match="before"):
            m.predict(x.loc[[raw.index[999]]])
        with pytest.raises(ValueError, match="schema"):
            m.predict(x.loc[[decision]].iloc[:, :-1])
        with (tmp_path / "bundle/model.joblib").open("ab") as f:
            f.write(b"tampered")
        with pytest.raises(ValueError, match="checksum"):
            AdaptiveModel.load(tmp_path / "bundle")


def test_insufficient_history_does_not_silently_predict():
    raw = market(500)
    with pytest.raises(ValueError, match="Insufficient"):
        AdaptiveModel().fit_at(features(raw, raw), raw.Close, raw.index[-1])


def test_hold_sessions_fill_mark_and_retry_without_rebalancing(tmp_path):
    config = Limits(max_position=0.5, max_turnover=0.8)
    a = Account(["A"], 10000, config, tmp_path / "account.db")
    s = ExecutionSession(a)
    s.on_close("2024-01-05", {"A": 100.0}, {"A": 0.4})
    s.on_close("2024-01-08", {"A": 101.0}, None)
    assert len([e for e in a.events if e["kind"] == "fill"]) == 1
    assert not a.pending()
    n = len(a.events)
    actual = s.on_close("2024-01-09", {"A": 120.0}, None)
    assert not a.pending() and a.positions["A"] == 40
    restarted = ExecutionSession(Account(["A"], 10000, config, tmp_path / "account.db"))
    assert restarted.on_close("2024-01-09", {"A": 120.0}, None) == actual
    assert len(a.events) == n + 1
    with pytest.raises(ValueError, match="reused"):
        restarted.on_close("2024-01-09", {"A": 121.0}, None)
    with pytest.raises(ValueError, match="Out-of-order"):
        restarted.on_close("2024-01-07", {"A": 100.0}, None)


def test_hold_session_still_enforces_drawdown_stop():
    a = Account(["A"], 10000, Limits(max_position=0.5, drawdown_limit=0.1))
    s = ExecutionSession(a)
    s.on_close("2024-01-05", {"A": 100.0}, {"A": 0.5})
    s.on_close("2024-01-08", {"A": 100.0}, None)
    s.on_close("2024-01-09", {"A": 70.0}, None)
    assert any(e["kind"] == "drawdown_halt" for e in a.events)
    assert a.pending()[0]["side"] == "sell"


def test_covariance_caps_concentration_and_volatility():
    rng = np.random.default_rng(12)
    cols = ["AAPL", "MSFT", "NVDA", "QQQ", "XLK", "SPY", "GLD"]
    history = pd.DataFrame(rng.normal(0, 0.08, (63, 7)), columns=cols)
    w = constrain_weights(pd.Series(0.5, index=cols), history)
    assert w.max() <= 0.1 and w.sum() <= 0.8
    assert w.iloc[:5].sum() <= 0.35
    cov = history.cov().to_numpy() * 252
    cov = 0.5 * cov + 0.5 * np.diag(np.diag(cov))
    assert np.sqrt(w.to_numpy() @ cov @ w.to_numpy()) <= 0.12 + 1e-12


def test_daily_snapshot_retries_restarts_and_missed_sessions(monkeypatch, tmp_path):
    import json
    import src.live.adaptive as live

    release = tmp_path / "release"
    release.mkdir()
    (release / "manifest.json").write_text("{}")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "manifest.json").write_text("{}")
    ready = dict(
        symbols=["A", "SPY"],
        capital=10000,
        candidate="trend",
        prospective_start="2026-09-29",
    )
    monkeypatch.setattr(live, "verify_release", lambda p: ready)
    frame = market(800)
    frame.index = pd.bdate_range(end="2026-09-29", periods=800, tz="UTC")
    frame.loc[:, ["Open", "High", "Low", "Close"]] = [100.0, 101.0, 99.0, 100.0]
    frames = {s: frame.copy() for s in ready["symbols"]}
    monkeypatch.setattr(live, "load_snapshot", lambda p: (frames, {}))
    monkeypatch.setattr(
        live,
        "daily_targets",
        lambda *a: (
            {s: 0.05 for s in ready["symbols"]},
            [],
            {s: "test" for s in ready["symbols"]},
        ),
    )
    workspace = tmp_path / "paper"
    first = live.run_daily(release, workspace, snapshot, now="2026-09-29T21:00Z")
    assert first["mode"] == "prospective_shadow"
    assert first["shadow"]["reserved_cash"] > 0
    assert first["deployment"]["cash"] == 10000
    for f in frames.values():
        f.loc[pd.Timestamp("2026-09-30", tz="UTC")] = [101, 102, 100, 101, 1000]
    second = live.run_daily(release, workspace, snapshot, now="2026-09-30T21:00Z")
    retry = live.run_daily(release, workspace, now="2026-09-30T21:30Z")
    assert second["shadow"] == retry["shadow"]
    assert second["shadow"]["positions"]["A"] > 0
    assert not second["rebalance"]
    assert second["shadow"]["reconciliation_error"] == pytest.approx(0)
    # Process dies after committing ledgers but before saving the receipt.
    (workspace / "days/2026-09-30/result.json").unlink()
    resumed = live.run_daily(release, workspace, snapshot, now="2026-09-30T21:40Z")
    assert resumed["shadow"] == second["shadow"]
    with pytest.raises(ValueError, match="stale"):
        live.run_daily(release, workspace, snapshot, now="2026-10-01T21:00Z")
    for f in frames.values():
        f.loc[pd.Timestamp("2026-10-02", tz="UTC")] = [101, 102, 100, 101, 1000]
    with pytest.raises(ValueError, match="Missed"):
        live.run_daily(release, workspace, snapshot, now="2026-10-02T21:00Z")


def test_proxy_marks_are_invariant_to_vendor_scale_revision():
    from src.live.adaptive import normalized_marks

    ix = pd.date_range("2026-09-24", periods=2, tz="UTC")
    previous = dict(session=str(ix[0]), marks={"A": 50.0})
    data = {"A": pd.DataFrame({"Close": [25.0, 26.0]}, index=ix)}
    assert normalized_marks(data, previous)["A"] == 52


def test_release_verifier_rejects_artifact_tampering(tmp_path):
    import json
    from src.data.snapshot import digest
    from src.live.adaptive import verify_release

    ready = tmp_path / "readiness.json"
    ready.write_text(json.dumps(dict(approved_strategy="cash", symbols=["SPY"])))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                status="complete",
                source={},
                symbols=["SPY"],
                artifacts={"readiness.json": digest(ready)},
            )
        )
    )
    assert verify_release(tmp_path)["approved_strategy"] == "cash"
    ready.write_text(json.dumps(dict(approved_strategy="trend", symbols=["SPY"])))
    with pytest.raises(ValueError, match="artifact changed"):
        verify_release(tmp_path)
