"""
Tests for the dynamic policy layer (src/risk/policy_engine.py) and its
integration points in signals.build_signal_df and RiskEngine.

Coverage
--------
  compute_policy   — column set, causality-safe values, rule behaviour:
                     unstable regime tightens confidence, high vol widens
                     ATR stop, stress halves size cap, extremes block entry
  decide_latest    — returns a PolicyDecision for the final bar
  stops            — atr_multiplier parameter honoured
  build_signal_df  — dynamic thresholds flip low-confidence bars to FLAT
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.policy_engine import (
    BASE_ATR_MULT,
    BASE_CONFIDENCE,
    MAX_ATR_MULT,
    PolicyDecision,
    PolicyEngine,
)
from src.risk.stops import StopState, initial_stop, update_trailing_stop


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _feat_df(
    n: int = 300,
    vol: float = 0.15,
    regime_seq=None,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    regime = regime_seq if regime_seq is not None else np.full(n, 2)
    return pd.DataFrame(
        {
            "Close": close,
            "realized_vol_21": np.full(n, vol),
            "hmm_regime": regime,
            "atr_14": np.full(n, 2.0),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# PolicyEngine.compute_policy
# ---------------------------------------------------------------------------

class TestComputePolicy:
    def test_columns_and_length(self):
        pol = PolicyEngine().compute_policy(_feat_df())
        assert list(pol.columns) == [
            "conf_threshold", "atr_mult", "size_cap",
            "eligible", "regime_stable", "vol_pctile",
        ]
        assert len(pol) == 300

    def test_stable_regime_base_confidence(self):
        """Constant regime + flat vol → base thresholds everywhere after warm-up."""
        pol = PolicyEngine().compute_policy(_feat_df())
        tail = pol.iloc[50:]
        assert (tail["conf_threshold"] == BASE_CONFIDENCE).all()
        assert (tail["regime_stable"]).all()

    def test_unstable_regime_tightens_confidence(self):
        n = 300
        regime = np.full(n, 2)
        regime[150::2] = 0          # regime flips every other bar from 150
        pol = PolicyEngine().compute_policy(_feat_df(regime_seq=regime))
        assert (pol["conf_threshold"].iloc[160:] > BASE_CONFIDENCE).all()
        assert not pol["regime_stable"].iloc[160:].any()

    def test_high_vol_widens_atr_stop(self):
        n = 300
        vol = np.full(n, 0.10)
        vol[250:] = 0.90            # extreme vol at the end → top percentile
        df = _feat_df(n)
        df["realized_vol_21"] = vol
        pol = PolicyEngine().compute_policy(df)
        assert pol["atr_mult"].iloc[100] == pytest.approx(BASE_ATR_MULT)
        assert pol["atr_mult"].iloc[-1] > BASE_ATR_MULT
        assert pol["atr_mult"].iloc[-1] <= MAX_ATR_MULT + 1e-9

    def test_extreme_vol_blocks_eligibility(self):
        n = 300
        vol = np.full(n, 0.10)
        vol[280:] = 2.0             # far beyond anything in history
        df = _feat_df(n)
        df["realized_vol_21"] = vol
        pol = PolicyEngine().compute_policy(df)
        assert not pol["eligible"].iloc[-1]
        assert pol["eligible"].iloc[100]

    def test_drawdown_stress_halves_size_cap(self):
        n = 300
        df = _feat_df(n)
        close = df["Close"].values.copy()
        close[250:] = close[249] * 0.80     # 20% crash below trailing peak
        df["Close"] = close
        pol = PolicyEngine().compute_policy(df)
        assert pol["size_cap"].iloc[-1] == 0.5
        assert pol["size_cap"].iloc[100] == 1.0

    def test_unknown_regime_ineligible(self):
        df = _feat_df(regime_seq=np.full(300, -1))
        pol = PolicyEngine().compute_policy(df)
        assert not pol["eligible"].any()

    def test_causality_future_does_not_change_past(self):
        """Corrupting the tail of the series must not alter earlier policy rows."""
        df = _feat_df(300)
        pe  = PolicyEngine()
        base = pe.compute_policy(df)
        df2 = df.copy()
        df2.loc[df2.index[-50]:, "realized_vol_21"] = 5.0
        df2.loc[df2.index[-50]:, "hmm_regime"] = 0
        mod = pe.compute_policy(df2)
        pd.testing.assert_frame_equal(base.iloc[:250], mod.iloc[:250])


# ---------------------------------------------------------------------------
# decide_latest
# ---------------------------------------------------------------------------

class TestDecideLatest:
    def test_returns_decision(self):
        d = PolicyEngine().decide_latest(_feat_df())
        assert isinstance(d, PolicyDecision)
        assert d.confidence_threshold == BASE_CONFIDENCE
        assert d.eligible

    def test_as_dict_keys(self):
        d = PolicyEngine().decide_latest(_feat_df())
        assert set(d.as_dict()) == {
            "confidence_threshold", "atr_multiplier", "size_cap",
            "eligible", "regime_stable", "vol_pctile",
        }


# ---------------------------------------------------------------------------
# stops.py multiplier parameter
# ---------------------------------------------------------------------------

class TestStopsMultiplier:
    def test_initial_stop_multiplier(self):
        assert initial_stop(100.0, 2.0, direction=1) == 96.0                 # default 2×
        assert initial_stop(100.0, 2.0, direction=1, atr_multiplier=3.0) == 94.0

    def test_trailing_stop_multiplier(self):
        s = StopState(active=True, stop_level=90.0, direction=1,
                      entry_price=100.0, entry_atr=2.0)
        update_trailing_stop(s, 110.0, 2.0, atr_multiplier=3.0)
        assert s.stop_level == 104.0     # 110 - 3*2


# ---------------------------------------------------------------------------
# build_signal_df integration
# ---------------------------------------------------------------------------

class TestSignalIntegration:
    def test_dynamic_threshold_flips_marginal_bars_to_flat(self, monkeypatch, tmp_path):
        """A bar with confidence between base (0.38) and tightened (0.48)
        threshold must be FLAT when the regime is unstable."""
        import src.backtest.signals as sig_mod

        n = 300
        regime = np.full(n, 2)
        regime[n - 40:] = np.where(np.arange(40) % 2 == 0, 0, 2)  # unstable tail
        feat = _feat_df(regime_seq=regime)
        feat.attrs["causality_version"] = 2
        monkeypatch.setattr("src.models.bundle.validate_bundle", lambda path: {})

        # OOS predictions with confidence 0.42 everywhere (UP-leaning)
        proba = pd.DataFrame(
            {"p_down": 0.29, "p_flat": 0.29, "p_up": 0.42, "true_label": 2},
            index=feat.index,
        )
        monkeypatch.setattr(sig_mod, "load_oos_predictions", lambda t: proba)
        monkeypatch.setattr(
            sig_mod, "_MODELS_ROOT", tmp_path
        )
        (tmp_path / "TEST").mkdir()
        (tmp_path / "TEST" / "feature_cols.json").write_text('["rsi_14"]')

        df_static = sig_mod.build_signal_df("TEST", feat)
        df_moe    = sig_mod.build_signal_df("TEST", feat, policy_engine=PolicyEngine())

        # static: threshold 0.38 < 0.42 → UP everywhere
        assert (df_static["filtered_signal"].iloc[-20:] == 2).all()
        # dynamic: unstable tail threshold 0.48 > 0.42 → forced FLAT
        assert (df_moe["filtered_signal"].iloc[-20:] == 1).all()
        # stable early region unchanged
        assert (df_moe["filtered_signal"].iloc[50:200] == 2).all()
