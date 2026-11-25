import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from src.data.validation import observed_equity_bars


def bars(index):
    return pd.DataFrame(
        dict(Open=100.0, High=101.0, Low=99.0, Close=100.0, Volume=10000.0),
        index=pd.to_datetime(index, utc=True),
    )


def test_holidays_unfinished_sessions_missing_sessions_and_freshness():
    df = bars(["2024-07-03", "2024-07-04", "2024-07-05"])
    out = observed_equity_bars(df, now="2024-07-05T19:59Z")
    assert list(out.index.day) == [3]
    with pytest.raises(ValueError, match="Stale"):
        observed_equity_bars(df.iloc[:1], now="2024-07-05T20:01Z", require_fresh=True)
    with pytest.raises(ValueError, match="Missing exchange sessions"):
        observed_equity_bars(
            bars(["2024-07-01", "2024-07-03"]), now="2024-07-05T21:00Z"
        )


def test_blank_smtp_port_uses_default(monkeypatch):
    from src.live.alerts import Alerts

    monkeypatch.setenv("SMTP_PORT", "")
    assert Alerts().smtp_port == 587


def test_news_early_close_and_dst_alignment():
    from src.features.sentiment import SentimentPipeline

    dates = pd.to_datetime(["2024-07-03", "2024-07-05"], utc=True)
    created = pd.Series(
        pd.to_datetime(["2024-07-03T16:59Z", "2024-07-03T17:01Z", "2024-07-04T12:00Z"])
    )
    actual = SentimentPipeline.attribute_to_session(created, dates)
    assert list(actual) == [dates[0], dates[1], dates[1]]


def test_inference_requires_schema_full_sequence_and_latest_row(monkeypatch, tmp_path):
    import src.live.signal_generator as sg
    import src.models.bundle as bundle
    from src.models.xgb_model import XGBModel
    from src.models.lgbm_model import LGBMModel
    from src.models.lstm_model import LSTMModel

    tree = MagicMock()
    tree.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
    lstm = MagicMock()
    lstm.seq_len = 3
    lstm.predict_proba.side_effect = lambda x: np.tile([0.1, 0.2, 0.7], (len(x), 1))
    monkeypatch.setattr(XGBModel, "load", lambda *a: tree)
    monkeypatch.setattr(LGBMModel, "load", lambda *a: tree)
    monkeypatch.setattr(LSTMModel, "load", lambda *a: lstm)
    monkeypatch.setattr(
        bundle,
        "validate_bundle",
        lambda *a: dict(trained_through="2023-01-01T00:00Z", skip_lstm=False),
    )
    df = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0]}, index=pd.date_range("2024-01-01", periods=3, tz="UTC")
    )
    assert sg._infer_latest_bar("X", df, ["a"])["p_up"] == pytest.approx(0.7)
    assert lstm.predict_proba.call_args.args[0].shape == (3, 1)
    with pytest.raises(ValueError, match="missing"):
        sg._infer_latest_bar("X", df, ["missing"])
    with pytest.raises(ValueError, match="consecutive"):
        sg._infer_latest_bar("X", df.iloc[-1:], ["a"])
    df.iloc[-1] = np.nan
    with pytest.raises(ValueError, match="consecutive"):
        sg._infer_latest_bar("X", df, ["a"])


def test_batch_failure_prevents_any_execution(monkeypatch, tmp_path):
    import src.live.engine as engine

    gen = MagicMock()
    gen.generate_batch.return_value = {}
    gen.errors = {"X": "stale data"}
    monkeypatch.setattr(engine, "SignalGenerator", lambda **kw: gen)
    monkeypatch.setattr(
        engine,
        "AlpacaBroker",
        lambda *a, **kw: pytest.fail("External order path entered"),
    )
    result = engine.LiveEngine(
        tickers=["X"], dry_run=False, state_path=tmp_path / "account.db"
    ).run()
    assert result.errors == [("X", "stale data")]
    assert not (tmp_path / "account.db").exists()


def test_bundle_hashes_detect_missing_or_changed_artifacts(tmp_path):
    from src.models.bundle import seal_bundle, validate_bundle

    (tmp_path / "training_results.json").write_text(
        json.dumps(dict(causality_version=2, skip_lstm=True))
    )
    for name in [
        "feature_cols.json",
        "xgb_final.json",
        "xgb_final_scaler.pkl",
        "lgbm_final.txt",
        "lgbm_final_scaler.pkl",
    ]:
        (tmp_path / name).write_text("example")
    seal_bundle(tmp_path)
    validate_bundle(tmp_path)
    (tmp_path / "xgb_final_scaler.pkl").write_text("changed")
    with pytest.raises(ValueError, match="altered"):
        validate_bundle(tmp_path)


def test_feature_cache_invalidated_by_inputs_and_forced_load(tmp_path, monkeypatch):
    import src.features.pipeline as pipeline

    monkeypatch.setattr(pipeline, "add_all_technical", lambda df: df)
    monkeypatch.setattr(pipeline, "add_all_statistical", lambda df: df)
    monkeypatch.setattr(pipeline, "add_hmm_regime", lambda df, **kw: (df, None))
    fe = pipeline.FeatureEngineer(cache_dir=tmp_path)
    df = bars(["2024-07-01", "2024-07-02"])
    first = fe.compute_features(df, "X")
    df.iloc[-1, df.columns.get_loc("Close")] = 110
    second = fe.compute_features(df, "X")
    assert first.Close.iloc[-1] == 100 and second.Close.iloc[-1] == 110
    df.iloc[-1, df.columns.get_loc("Close")] = 120
    assert fe.load_features("X", df=df, force_refresh=True).Close.iloc[-1] == 120
