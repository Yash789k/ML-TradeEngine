"""
Tests for the MoE gating model (src/models/gating.py) and the
ensemble_mode plumbing in EnsembleClassifier.

Coverage
--------
  build_market_context   — causal columns, regime one-hot, NaN handling
  rolling_expert_accuracy — warm-up default, 1-bar shift (no leakage)
  GatingModel            — min-evidence rule, weight floor/normalisation,
                           learns context-dependent expert preference,
                           save/load round-trip
  EnsembleClassifier     — backward compatibility, moe mode validation,
                           moe weights actually change the blend
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.ensemble import EnsembleClassifier
from src.models.gating import (
    GatingModel,
    build_gate_context,
    build_market_context,
    rolling_expert_accuracy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _feat_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    return pd.DataFrame(
        {
            "Close": close,
            "realized_vol_21": np.abs(rng.normal(0.15, 0.05, n)),
            "hmm_regime": rng.choice([0, 1, 2], size=n),
        },
        index=idx,
    )


def _proba(preferred: np.ndarray, strength: float, n_classes: int = 3) -> np.ndarray:
    """Probability matrix that puts `strength` on the preferred class."""
    n = len(preferred)
    p = np.full((n, n_classes), (1 - strength) / (n_classes - 1))
    p[np.arange(n), preferred] = strength
    return p


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------

class TestContext:
    def test_market_context_columns(self):
        ctx = build_market_context(_feat_df())
        assert list(ctx.columns) == [
            "regime_bear", "regime_ranging", "regime_bull",
            "vol_21", "vol_pctile", "drawdown_state",
        ]
        assert not ctx.isna().any().any()

    def test_regime_one_hot_sums_to_one(self):
        ctx = build_market_context(_feat_df())
        onehot = ctx[["regime_bear", "regime_ranging", "regime_bull"]].sum(axis=1)
        assert (onehot == 1.0).all()

    def test_drawdown_state_non_positive(self):
        ctx = build_market_context(_feat_df())
        assert (ctx["drawdown_state"] <= 1e-9).all()

    def test_missing_regime_column_defaults(self):
        df = _feat_df().drop(columns=["hmm_regime"])
        ctx = build_market_context(df)
        assert (ctx[["regime_bear", "regime_ranging", "regime_bull"]].sum(axis=1) == 0).all()

    def test_rolling_accuracy_shifted_no_leakage(self):
        """Accuracy at row t must not include row t's own outcome."""
        n = 100
        y = np.zeros(n, dtype=int)
        # expert right ONLY on row 50
        preds = np.ones(n, dtype=int)
        preds[50] = 0
        probs = [_proba(preds, 0.9)]
        acc = rolling_expert_accuracy(probs, y, window=10)
        # at row 50 the window covers rows <=49 where expert was always wrong
        assert acc[50, 0] == 0.0
        # Outcome matures at 55 and is used strictly after that close.
        assert acc[51, 0] == 0.0
        assert acc[56, 0] > 0.0

    def test_gate_context_has_accuracy_columns(self):
        feat = _feat_df(200)
        y = np.random.default_rng(1).choice([0, 1, 2], 200)
        probs = [_proba(y, 0.8)] * 3
        ctx = build_gate_context(feat, probs, y, index=feat.index)
        assert {"acc_xgb", "acc_lgbm", "acc_lstm"}.issubset(ctx.columns)
        assert len(ctx) == 200


# ---------------------------------------------------------------------------
# GatingModel
# ---------------------------------------------------------------------------

class TestGatingModel:
    def test_min_evidence_uniform(self):
        gate = GatingModel(min_train=100)
        ctx = pd.DataFrame(np.random.rand(50, 4))
        gate.fit(ctx, np.random.randint(0, 2, (50, 3)))
        W = gate.predict_weights(ctx)
        assert np.allclose(W, 1.0 / 3)

    def test_weights_sum_to_one_and_floored(self):
        rng = np.random.default_rng(2)
        ctx = pd.DataFrame(rng.normal(size=(300, 5)))
        correct = rng.integers(0, 2, (300, 3))
        gate = GatingModel(min_train=100).fit(ctx, correct)
        W = gate.predict_weights(ctx)
        assert np.allclose(W.sum(axis=1), 1.0)
        assert (W >= gate.min_weight / 3).all()   # floor propagates through normalisation

    def test_learns_regime_dependent_expert(self):
        """Expert 0 is right in regime A, expert 1 in regime B — the gate
        must upweight the right expert in each regime."""
        rng = np.random.default_rng(3)
        n = 600
        regime = rng.integers(0, 2, n)
        ctx = pd.DataFrame({"regime": regime.astype(float),
                            "noise": rng.normal(size=n)})
        correct = np.zeros((n, 3))
        correct[:, 0] = (regime == 0).astype(float)   # expert 0 right in regime 0
        correct[:, 1] = (regime == 1).astype(float)   # expert 1 right in regime 1
        correct[:, 2] = 0.33 * rng.random(n) < 0.2    # expert 2 random-ish

        gate = GatingModel(min_train=100).fit(ctx, correct)
        W0 = gate.predict_weights(pd.DataFrame({"regime": [0.0], "noise": [0.0]}))
        W1 = gate.predict_weights(pd.DataFrame({"regime": [1.0], "noise": [0.0]}))
        assert W0[0, 0] > W0[0, 1], "regime 0 should prefer expert 0"
        assert W1[0, 1] > W1[0, 0], "regime 1 should prefer expert 1"

    def test_degenerate_expert_target(self):
        """An expert that is ALWAYS right must not crash the fit."""
        rng = np.random.default_rng(4)
        ctx = pd.DataFrame(rng.normal(size=(200, 3)))
        correct = np.column_stack([
            np.ones(200),                    # always right
            rng.integers(0, 2, 200),
            np.zeros(200),                   # never right
        ])
        gate = GatingModel(min_train=100).fit(ctx, correct)
        W = gate.predict_weights(ctx)
        assert np.allclose(W.sum(axis=1), 1.0)
        assert W[:, 0].mean() > W[:, 2].mean()

    def test_save_load_roundtrip(self, tmp_path):
        rng = np.random.default_rng(5)
        ctx = pd.DataFrame(rng.normal(size=(200, 4)))
        correct = rng.integers(0, 2, (200, 3))
        gate = GatingModel(min_train=50).fit(ctx, correct)
        path = tmp_path / "gate.joblib"
        gate.save(str(path))
        loaded = GatingModel.load(str(path))
        np.testing.assert_allclose(
            gate.predict_weights(ctx), loaded.predict_weights(ctx)
        )


# ---------------------------------------------------------------------------
# EnsembleClassifier plumbing
# ---------------------------------------------------------------------------

class TestEnsembleModes:
    def _three_probas(self, n=50):
        rng = np.random.default_rng(6)
        out = []
        for _ in range(3):
            p = rng.random((n, 3))
            out.append(p / p.sum(axis=1, keepdims=True))
        return out

    def test_static_default_backward_compatible(self):
        a, b, c = self._three_probas()
        ens = EnsembleClassifier()
        proba = ens.predict_proba(a, b, c)
        np.testing.assert_allclose(proba, (a + b + c) / 3, atol=1e-9)

    def test_moe_requires_gate(self):
        with pytest.raises(ValueError, match="requires a fitted GatingModel"):
            EnsembleClassifier(ensemble_mode="moe")

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="ensemble_mode"):
            EnsembleClassifier(ensemble_mode="dynamic")

    def test_moe_without_context_falls_back_to_static(self):
        a, b, c = self._three_probas()
        gate = GatingModel(min_train=10**9)   # never fits → uniform
        ens = EnsembleClassifier(ensemble_mode="moe", gate=gate)
        np.testing.assert_allclose(
            ens.predict_proba(a, b, c), (a + b + c) / 3, atol=1e-9
        )

    def test_moe_weights_change_blend(self):
        n = 30
        a, b, c = self._three_probas(n)

        class FixedGate(GatingModel):
            def predict_weights(self, context):
                return np.tile([0.8, 0.1, 0.1], (len(context), 1))

        ens = EnsembleClassifier(ensemble_mode="moe", gate=FixedGate())
        ctx = pd.DataFrame(np.zeros((n, 2)))
        proba = ens.predict_proba(a, b, c, context=ctx)
        expected = 0.8 * a + 0.1 * b + 0.1 * c
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(proba, expected, atol=1e-9)

    def test_moe_context_length_mismatch(self):
        a, b, c = self._three_probas(30)
        gate = GatingModel()
        ens = EnsembleClassifier(ensemble_mode="moe", gate=gate)
        with pytest.raises(ValueError, match="align row-for-row"):
            ens.predict_proba(a, b, c, context=pd.DataFrame(np.zeros((10, 2))))
