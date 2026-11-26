import numpy as np
import pandas as pd
import pytest
from src.features.regime import add_hmm_regime, decode_hmm
from src.models.gating import rolling_expert_accuracy, build_market_context
from src.risk.policy_engine import PolicyEngine


def hmm_frame(n=340):
    rng = np.random.default_rng(713)
    return pd.DataFrame(
        {
            "log_return": rng.normal(0, 0.02, n),
            "realized_vol_21": rng.uniform(0.01, 0.03, n),
        },
        index=pd.date_range("2020-01-01", periods=n),
    )


def test_hmm_prefix_invariant_and_warmup_alignment():
    frame = hmm_frame()
    frame.iloc[:8] = np.nan
    short, _ = add_hmm_regime(frame.iloc[:300].copy(), covariance_type="diag")
    long, model = add_hmm_regime(frame.copy(), covariance_type="diag")
    pd.testing.assert_series_equal(short.hmm_regime, long.hmm_regime.iloc[:300])
    assert (long.hmm_regime.iloc[:260] == -1).all()
    assert (decode_hmm(model, frame)[:8] == -1).all()
    frame.iloc[300:] = 999
    changed, _ = add_hmm_regime(frame, covariance_type="diag")
    pd.testing.assert_series_equal(short.hmm_regime, changed.hmm_regime.iloc[:300])


def test_gate_cannot_observe_unmatured_labels():
    p = np.tile([0, 0, 1.0], (70, 1))
    y = np.full(70, 2)
    old = rolling_expert_accuracy([p], y, horizon=5)
    y[19] = 0
    new = rolling_expert_accuracy([p], y, horizon=5)
    np.testing.assert_array_equal(old[:25], new[:25])
    assert old[25, 0] != new[25, 0]


def test_warmup_volatility_cannot_use_future_median():
    frame = hmm_frame(100)
    frame["Close"] = 100
    frame["hmm_regime"] = 2
    frame.loc[frame.index[:12], "realized_vol_21"] = np.nan
    for fn in [build_market_context, PolicyEngine().compute_policy]:
        pd.testing.assert_frame_equal(fn(frame.iloc[:60]), fn(frame).iloc[:60])


def test_macro_requires_release_and_vintage_evidence():
    from src.features.cross_asset import add_macro_features

    df = pd.DataFrame(index=pd.date_range("2024-01-01", periods=6, tz="UTC"))
    macro = pd.DataFrame(
        {"CPI": [3.0], "available_at": [pd.Timestamp("2024-01-04", tz="UTC")]},
        index=[df.index[0]],
    )
    assert "macro_CPI" not in add_macro_features(df.copy(), macro)
    macro.attrs["point_in_time"] = True
    out = add_macro_features(df.copy(), macro)
    assert out.macro_CPI.iloc[:3].isna().all()
    assert (out.macro_CPI.iloc[3:] == 3).all()


def test_purge_must_cover_label_horizon(tmp_path):
    from src.models.trainer import ModelTrainer
    from src.models.splitter import WalkForwardSplit

    frame = hmm_frame()
    frame["Close"] = 100 * np.exp(frame.log_return.cumsum())
    trainer = ModelTrainer(
        models_dir=tmp_path / "models",
        mlflow_uri="sqlite:///" + str(tmp_path / "tracking.db"),
    )
    with pytest.raises(ValueError, match="horizon"):
        trainer.run(frame, embargo_days=1, horizon=5)
    for tr, te in WalkForwardSplit(embargo_days=5).split(frame):
        assert tr[-1] + 5 < te[0]


def test_supervised_selection_not_run_during_prepare(tmp_path, monkeypatch):
    from src.models.trainer import ModelTrainer
    from src.features.pipeline import FeatureEngineer

    monkeypatch.setattr(
        FeatureEngineer,
        "select_features",
        lambda *a, **k: pytest.fail("Global selection leaks"),
    )
    frame = hmm_frame()
    frame["Close"] = 100 * np.exp(frame.log_return.cumsum())
    trainer = ModelTrainer(
        models_dir=tmp_path / "models",
        mlflow_uri="sqlite:///" + str(tmp_path / "tracking.db"),
    )
    X, y, cols = trainer.prepare_data(frame, top_n_features=1)
    assert len(cols) == 2


def test_all_zoo_signals_and_risk_decisions_preserve_prefix():
    from tests.test_feature_pipeline import _make_ohlcv
    from src.research.strategies import ALL_STRATEGIES
    from src.risk.engine import RiskEngine
    from src.features.technical import add_all_technical
    from src.features.statistical import add_all_statistical

    raw = _make_ohlcv(n=430)
    altered = raw.copy()
    altered.iloc[360:] *= 1.7
    for strategy in ALL_STRATEGIES:
        a = strategy.generate_signals(raw.iloc[:360], pair_df=raw.iloc[:360])
        b = strategy.generate_signals(altered, pair_df=altered)
        pd.testing.assert_series_equal(a, b.iloc[:360], obj=strategy.name)
    feat = add_all_statistical(add_all_technical(raw.copy()))
    feat["hmm_regime"] = 2
    sig = pd.DataFrame(
        {"close": raw.Close, "p_up": 0.8, "filtered_signal": 2}, index=raw.index
    )
    for policy in [None, PolicyEngine()]:
        engine = RiskEngine()
        a = engine.run("X", sig.iloc[:360], feat.iloc[:360], policy_engine=policy)
        b = engine.run("X", sig, feat, policy_engine=policy)
        pd.testing.assert_series_equal(a.equity_curve, b.equity_curve.iloc[:360])
        np.testing.assert_allclose(
            b.ledger.equity, b.ledger.cash + b.ledger.shares * b.ledger.price
        )
