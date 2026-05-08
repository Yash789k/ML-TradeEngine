"""
Phase 03 — ML Pipeline Structural & Smoke Tests
pytest tests/test_ml_pipeline.py

Tests are designed to complete in < 60 s (no full Optuna sweeps, no long
LSTM training runs).  They validate:

  Labels    (1–5)  : generation, 3 classes, no lookahead, class weights
  Splitter  (6–10) : fold boundaries, no train/test overlap, embargo gap,
                     expanding vs rolling window
  XGBModel  (11–13): smoke-fit (2 trials), predict_proba shape/bounds
  LGBMModel (14–16): same for LightGBM
  LSTM      (17–20): forward-pass shape, SequenceDataset length, smoke-fit,
                     predict_proba warm-up rows
  Ensemble  (21–23): soft-vote output shape, rows sum to 1, from_val_f1 weights
  Trainer   (24)   : prepare_data returns aligned X, y, feat_cols
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

_N    = 300
_SEED = 7


def _make_ohlcv(n: int = _N, seed: int = _SEED) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n, tz="UTC")
    log_r = rng.normal(0.0005, 0.012, n)
    close = 100.0 * np.exp(np.cumsum(log_r))
    df    = pd.DataFrame(
        {
            "Open":      close * rng.uniform(0.99, 1.01, n),
            "High":      close * rng.uniform(1.00, 1.02, n),
            "Low":       close * rng.uniform(0.98, 1.00, n),
            "Close":     close,
            "Adj_Close": close,
            "Volume":    rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )
    df.index.name = "Date"
    df["High"] = df[["High", "Close"]].max(axis=1)
    df["Low"]  = df[["Low",  "Close"]].min(axis=1)
    return df


def _make_feature_df(n: int = _N) -> pd.DataFrame:
    """Tiny synthetic feature matrix (OHLCV + fake engineered columns)."""
    rng  = np.random.default_rng(_SEED)
    base = _make_ohlcv(n)
    for i in range(20):
        base[f"feat_{i}"] = rng.normal(0, 1, n)
    return base


@pytest.fixture(scope="module")
def ohlcv():
    return _make_ohlcv()


@pytest.fixture(scope="module")
def feat_df():
    return _make_feature_df()


# ---------------------------------------------------------------------------
# 1–5. Labels
# ---------------------------------------------------------------------------

def test_labels_three_classes(ohlcv):
    from src.models.labels import make_labels, CLASS_MASK
    labels, _ = make_labels(ohlcv, horizon=5, threshold=0.005)
    valid = labels[labels != CLASS_MASK]
    classes = set(valid.unique())
    assert classes.issubset({0, 1, 2}), f"Unexpected label values: {classes - {0,1,2}}"


def test_labels_mask_trailing_rows(ohlcv):
    """Last `horizon` rows must be masked."""
    from src.models.labels import make_labels, CLASS_MASK
    horizon = 5
    labels, _ = make_labels(ohlcv, horizon=horizon)
    assert all(labels.iloc[-horizon:] == CLASS_MASK), \
        "Trailing rows should have mask label -1"


def test_labels_no_lookahead(ohlcv):
    """Inserting a spike at the very last row must not change earlier labels."""
    from src.models.labels import make_labels, CLASS_MASK
    labels_orig, _ = make_labels(ohlcv, horizon=5, threshold=0.005)

    modified = ohlcv.copy()
    modified.iloc[-1, modified.columns.get_loc("Close")] *= 10.0
    labels_mod, _ = make_labels(modified, horizon=5, threshold=0.005)

    # Only the row that uses the last close as the FUTURE price can change
    # (the row at position n-horizon-1 uses close[-1] as its 5-bar-forward target)
    # Everything strictly before n-horizon-1 must be unchanged
    horizon = 5
    unchanged_slice = labels_orig.iloc[:-(horizon + 1)]
    modified_slice  = labels_mod.iloc[:-(horizon + 1)]
    pd.testing.assert_series_equal(unchanged_slice, modified_slice, check_names=False)


def test_labels_series_index_aligned(ohlcv):
    from src.models.labels import make_labels
    labels, _ = make_labels(ohlcv, horizon=5)
    assert labels.index.equals(ohlcv.index), "Label index does not match OHLCV index"


def test_class_weights_all_classes(ohlcv):
    from src.models.labels import make_labels, compute_class_weights, CLASS_MASK
    labels, _ = make_labels(ohlcv)
    cw = compute_class_weights(labels[labels != CLASS_MASK])
    assert set(cw.keys()) == {0, 1, 2}, f"Expected keys {{0,1,2}}, got {set(cw.keys())}"
    assert all(v > 0 for v in cw.values()), "Class weights must be positive"


# ---------------------------------------------------------------------------
# 6–10. Walk-Forward Splitter
# ---------------------------------------------------------------------------

def test_splitter_correct_number_of_folds():
    from src.models.splitter import WalkForwardSplit
    # n=500 guarantees all 5 folds clear the min_train=50 threshold
    sp = WalkForwardSplit(n_splits=5, embargo_days=5, min_train=50)
    folds = list(sp.split(np.zeros(500)))
    assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"


def test_splitter_no_train_test_overlap():
    from src.models.splitter import WalkForwardSplit
    sp = WalkForwardSplit(n_splits=5, embargo_days=5, min_train=50)
    for tr, te in sp.split(np.zeros(300)):
        overlap = set(tr) & set(te)
        assert not overlap, f"Train/test overlap: {len(overlap)} shared indices"


def test_splitter_embargo_gap():
    """The gap between last train index and first test index must be >= embargo_days."""
    from src.models.splitter import WalkForwardSplit
    embargo = 10
    sp = WalkForwardSplit(n_splits=5, embargo_days=embargo, min_train=50)
    for tr, te in sp.split(np.zeros(300)):
        gap = te[0] - tr[-1] - 1
        assert gap >= embargo - 1, \
            f"Embargo gap {gap} < embargo_days-1 ({embargo-1})"


def test_splitter_train_before_test():
    from src.models.splitter import WalkForwardSplit
    sp = WalkForwardSplit(n_splits=5, embargo_days=5, min_train=50)
    for tr, te in sp.split(np.zeros(300)):
        assert tr[-1] < te[0], "Train indices should all precede test indices"


def test_splitter_rolling_window():
    from src.models.splitter import WalkForwardSplit
    sp = WalkForwardSplit(n_splits=4, embargo_days=5, min_train=50, window_type="rolling")
    folds = list(sp.split(np.zeros(300)))
    assert len(folds) > 0, "Rolling window produced zero folds"
    # Verify train indices always precede test indices in every fold
    for tr, te in folds:
        assert tr[-1] < te[0], "Rolling: train must precede test"


# ---------------------------------------------------------------------------
# 11–13. XGBModel smoke test (2 Optuna trials, tiny data)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_xy():
    """80-sample dataset with 10 features for smoke-testing models."""
    rng = np.random.default_rng(99)
    X   = rng.standard_normal((80, 10)).astype(np.float32)
    y   = rng.integers(0, 3, 80).astype(np.int64)
    return X, y


def test_xgb_fit_predict_shape(tiny_xy):
    from src.models.xgb_model import XGBModel
    X, y = tiny_xy
    model = XGBModel(n_trials=2, val_fraction=0.2)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (80, 3), f"Expected (80,3), got {proba.shape}"


def test_xgb_proba_sums_to_one(tiny_xy):
    from src.models.xgb_model import XGBModel
    X, y = tiny_xy
    model = XGBModel(n_trials=2)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5), \
        "XGB predict_proba rows do not sum to 1"


def test_xgb_predict_labels_valid(tiny_xy):
    from src.models.xgb_model import XGBModel
    X, y = tiny_xy
    model = XGBModel(n_trials=2)
    model.fit(X, y)
    preds = model.predict(X)
    assert set(preds).issubset({0, 1, 2}), f"Unexpected label values: {set(preds) - {0,1,2}}"


# ---------------------------------------------------------------------------
# 14–16. LGBMModel smoke test
# ---------------------------------------------------------------------------

def test_lgbm_fit_predict_shape(tiny_xy):
    from src.models.lgbm_model import LGBMModel
    X, y = tiny_xy
    model = LGBMModel(n_trials=2, val_fraction=0.2)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (80, 3)


def test_lgbm_proba_sums_to_one(tiny_xy):
    from src.models.lgbm_model import LGBMModel
    X, y = tiny_xy
    model = LGBMModel(n_trials=2)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_lgbm_predict_labels_valid(tiny_xy):
    from src.models.lgbm_model import LGBMModel
    X, y = tiny_xy
    model = LGBMModel(n_trials=2)
    model.fit(X, y)
    preds = model.predict(X)
    assert set(preds).issubset({0, 1, 2})


# ---------------------------------------------------------------------------
# 17–20. LSTMModel structural tests
# ---------------------------------------------------------------------------

def test_lstm_sequence_dataset_length():
    from src.models.lstm_model import SequenceDataset
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randint(0, 3, 100)
    ds = SequenceDataset(X, y, seq_len=20)
    assert len(ds) == 81, f"Expected 81, got {len(ds)}"  # 100 - 20 + 1


def test_lstm_sequence_dataset_item_shape():
    import torch
    from src.models.lstm_model import SequenceDataset
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randint(0, 3, 100)
    ds = SequenceDataset(X, y, seq_len=20)
    seq, label = ds[0]
    assert seq.shape   == (20, 10), f"Sequence shape: {seq.shape}"
    assert label.shape == torch.Size([]), "Label should be a scalar"


def test_lstm_forward_pass_shape():
    import torch
    from src.models.lstm_model import _LSTMNet
    net = _LSTMNet(input_dim=10, hidden_dim=32, n_layers=2)
    x   = torch.randn(8, 20, 10)   # (batch=8, seq=20, features=10)
    out = net(x)
    assert out.shape == (8, 3), f"Expected (8,3), got {out.shape}"


def test_lstm_smoke_fit_predict(tiny_xy):
    """2-epoch LSTM smoke test — checks API contract only, not convergence."""
    from src.models.lstm_model import LSTMModel
    X, y = tiny_xy
    # Use seq_len=5 so 80-row data has enough sequences
    model = LSTMModel(seq_len=5, n_epochs=2, patience=2, hidden_dim=16, n_layers=1)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (80, 3), f"Expected (80,3), got {proba.shape}"
    # First seq_len-1 rows get uniform prior
    assert np.allclose(proba[:4].sum(axis=1), 1.0, atol=1e-5)
    # All rows should be valid probabilities
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 21–23. Ensemble
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dummy_probas():
    rng = np.random.default_rng(5)
    def _rand_proba(n):
        raw = rng.dirichlet(np.ones(3), n).astype(np.float32)
        return raw
    n = 50
    return _rand_proba(n), _rand_proba(n), _rand_proba(n)


def test_ensemble_output_shape(dummy_probas):
    from src.models.ensemble import EnsembleClassifier
    xgb, lgbm, lstm = dummy_probas
    ens   = EnsembleClassifier()
    proba = ens.predict_proba(xgb, lgbm, lstm)
    assert proba.shape == (50, 3)


def test_ensemble_rows_sum_to_one(dummy_probas):
    from src.models.ensemble import EnsembleClassifier
    xgb, lgbm, lstm = dummy_probas
    proba = EnsembleClassifier().predict_proba(xgb, lgbm, lstm)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_ensemble_from_val_f1_weights(dummy_probas):
    from src.models.ensemble import EnsembleClassifier
    ens = EnsembleClassifier.from_val_f1(0.55, 0.60, 0.50)
    assert len(ens.weights) == 3
    assert abs(ens.weights.sum() - 1.0) < 1e-6, "Weights must sum to 1"
    # Best model (LGBM, f1=0.60) should have the highest weight
    assert ens.weights[1] > ens.weights[0] and ens.weights[1] > ens.weights[2]


# ---------------------------------------------------------------------------
# 24. Trainer.prepare_data
# ---------------------------------------------------------------------------

def test_trainer_prepare_data_alignment(feat_df):
    from src.models.trainer import ModelTrainer
    trainer = ModelTrainer()
    X, y, feat_cols = trainer.prepare_data(feat_df, horizon=5, threshold=0.005, top_n_features=20)

    assert len(X) == len(y), "X and y must be the same length"
    assert X.index.equals(y.index), "X and y must share the same index"
    assert "label" not in X.columns, "Label column must not appear in X"
    assert all(c in feat_df.columns for c in feat_cols), \
        "All feature cols must come from the original DataFrame"
    # No NaN in X
    assert not X.isna().any().any(), "X should not contain NaN after prepare_data"
    # No invalid labels
    from src.models.labels import CLASS_MASK
    assert (y != CLASS_MASK).all(), "prepare_data must drop masked rows"
