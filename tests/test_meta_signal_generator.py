"""
Tests for the live meta-strategy wiring (src/live/meta_signal_generator.py).

DataLoader/FeatureEngineer network calls are mocked with synthetic OHLCV so
these tests run offline. Coverage:

  compute_strategy_returns — one column per zoo strategy, aligned index
  _today_direction         — long_only collapse of -1 → flat; correct
                              {2,1,0} encoding
  _strategy_by_name        — lookup by name / None for unknown
  MetaSignalGenerator.generate — end-to-end with mocked data fetch:
                              selected_strategy populated, strategy_mode
                              == "meta", direction matches the selected
                              strategy's own live signal, EW burn-in → FLAT
  LiveEngine integration    — strategy_mode="meta" uses equal-weight sizing,
                              not Kelly
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.live.meta_signal_generator import (
    MetaSignalGenerator,
    _strategy_by_name,
    _today_direction,
    compute_strategy_returns,
)
from src.live.signal_generator import LiveSignal
from src.research.strategies import ALL_STRATEGIES
from src.research.strategies.momentum import MomentumStrategy


# ---------------------------------------------------------------------------
# Synthetic OHLCV fixture — long enough for every zoo strategy's warm-up
# (Momentum needs 252+21 bars) plus meta-strategy burn-in (60 in-regime obs)
# ---------------------------------------------------------------------------

def _synthetic_ohlcv(n: int = 500, seed: int = 0, trend: float = 0.0007) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n, freq="B", tz="UTC")
    close = 100 * np.cumprod(1 + rng.normal(trend, 0.012, n))
    high  = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low   = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    openp = close * (1 + rng.normal(0, 0.002, n))
    vol   = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": openp, "High": high, "Low": low, "Close": close,
         "Adj_Close": close, "Volume": vol},
        index=idx,
    )


def _synthetic_feat_df(ohlcv: pd.DataFrame, regime_seq=None) -> pd.DataFrame:
    n = len(ohlcv)
    regime = regime_seq if regime_seq is not None else np.full(n, 2)
    feat = ohlcv.copy()
    feat["atr_14"]      = (ohlcv["High"] - ohlcv["Low"]).rolling(14, min_periods=1).mean()
    feat["hmm_regime"]  = regime
    return feat


# ---------------------------------------------------------------------------
# compute_strategy_returns
# ---------------------------------------------------------------------------

class TestComputeStrategyReturns:
    def test_one_column_per_strategy(self):
        ohlcv = _synthetic_ohlcv()
        spy   = _synthetic_ohlcv(seed=1)
        out   = compute_strategy_returns(ohlcv, spy, macro=None)
        assert set(out.columns) == {s.name for s in ALL_STRATEGIES}
        assert len(out) == len(ohlcv)
        assert not out.isna().any().any()

    def test_works_without_spy_or_macro(self):
        """Pairs/Carry degrade to all-flat (0 returns) rather than crashing."""
        ohlcv = _synthetic_ohlcv()
        out = compute_strategy_returns(ohlcv, spy_df=None, macro=None)
        assert "Pairs_StatArb" in out.columns
        assert (out["Pairs_StatArb"] == 0).all()


# ---------------------------------------------------------------------------
# _today_direction / _strategy_by_name
# ---------------------------------------------------------------------------

class TestDirectionMapping:
    def test_long_only_collapses_short_to_flat(self):
        ohlcv = _synthetic_ohlcv()
        strat = MomentumStrategy()
        assert strat.long_only

        with mock.patch.object(
            strat, "generate_signals",
            return_value=pd.Series(-1, index=ohlcv.index),
        ):
            direction = _today_direction(strat, ohlcv, None, None)
        assert direction == 1   # -1 collapsed to FLAT, not treated as short

    def test_long_signal_maps_to_up(self):
        ohlcv = _synthetic_ohlcv()
        strat = MomentumStrategy()
        with mock.patch.object(
            strat, "generate_signals",
            return_value=pd.Series(1, index=ohlcv.index),
        ):
            assert _today_direction(strat, ohlcv, None, None) == 2

    def test_flat_signal_maps_to_flat(self):
        ohlcv = _synthetic_ohlcv()
        strat = MomentumStrategy()
        with mock.patch.object(
            strat, "generate_signals",
            return_value=pd.Series(0, index=ohlcv.index),
        ):
            assert _today_direction(strat, ohlcv, None, None) == 1

    def test_strategy_lookup(self):
        assert _strategy_by_name("Momentum_12_1") is not None
        assert _strategy_by_name("Momentum_12_1").name == "Momentum_12_1"
        assert _strategy_by_name("does_not_exist") is None


# ---------------------------------------------------------------------------
# MetaSignalGenerator.generate — end to end with mocked fetch
# ---------------------------------------------------------------------------

@pytest.fixture
def mocked_fetch(monkeypatch):
    """Patch the data-fetch and feature-compute steps with synthetic data."""
    ohlcv = _synthetic_ohlcv(n=400, trend=0.0015)   # trending → momentum wins
    spy   = _synthetic_ohlcv(n=400, seed=2)
    feat  = _synthetic_feat_df(ohlcv)

    monkeypatch.setattr(
        "src.live.meta_signal_generator._fetch_market_data",
        lambda ticker, force_refresh: (ohlcv, spy, None),
    )

    class FakeFE:
        def __init__(self, *a, **k): ...
        def compute_features(self, **kw):
            return feat

    monkeypatch.setattr(
        "src.features.pipeline.FeatureEngineer", FakeFE
    )
    return ohlcv, spy, feat


class TestMetaSignalGeneratorGenerate:
    def test_returns_live_signal_with_meta_fields(self, mocked_fetch):
        gen = MetaSignalGenerator()
        signal = gen.generate("TEST")
        assert isinstance(signal, LiveSignal)
        assert signal.strategy_mode == "meta"
        assert signal.selected_strategy is not None
        assert signal.direction in (0, 1, 2)

    def test_kelly_frac_is_binary_weight_not_probability(self, mocked_fetch):
        gen = MetaSignalGenerator()
        signal = gen.generate("TEST")
        assert signal.kelly_frac in (0.0, 1.0)

    def test_ew_burnin_forces_flat(self, monkeypatch):
        """With almost no history, min_regime_obs (default 60) is never
        satisfied → EW burn-in → the generator must emit FLAT rather than
        guessing."""
        ohlcv = _synthetic_ohlcv(n=50)
        spy   = _synthetic_ohlcv(n=50, seed=3)
        feat  = _synthetic_feat_df(ohlcv)

        monkeypatch.setattr(
            "src.live.meta_signal_generator._fetch_market_data",
            lambda ticker, force_refresh: (ohlcv, spy, None),
        )

        class FakeFE:
            def __init__(self, *a, **k): ...
            def compute_features(self, **kw):
                return feat

        monkeypatch.setattr("src.features.pipeline.FeatureEngineer", FakeFE)

        gen = MetaSignalGenerator()
        signal = gen.generate("TEST")
        assert signal.selected_strategy == "EW"
        assert signal.direction == 1

    def test_generate_batch_skips_failures(self, mocked_fetch):
        gen = MetaSignalGenerator()
        real_generate = MetaSignalGenerator.generate

        def flaky(self, ticker, force_refresh=True):
            if ticker == "BAD":
                raise RuntimeError("boom")
            return real_generate(self, ticker, force_refresh)

        with mock.patch.object(MetaSignalGenerator, "generate", flaky):
            out = gen.generate_batch(["GOOD", "BAD"])
        assert "GOOD" in out
        assert "BAD" not in out


# ---------------------------------------------------------------------------
# LiveEngine integration — equal-weight sizing under strategy_mode="meta"
# ---------------------------------------------------------------------------

class TestLiveEngineMetaSizing:
    def test_meta_mode_uses_equal_weight_not_kelly(self):
        from src.live.engine import LiveEngine

        engine = LiveEngine(strategy_mode="meta", max_position_frac=0.05)
        signal = LiveSignal(
            ticker="X", date=mock.Mock(), direction=2, confidence=1.0,
            p_up=1.0, p_flat=0.0, p_down=0.0,
            kelly_frac=1.0,   # meta mode always sets this to 1.0 for UP
            atr=1.0, stop_loss=95.0, close=100.0,
            strategy_mode="meta", selected_strategy="Momentum_12_1",
        )
        # 5 tickers in the run → equal weight 1/5 = 0.20, capped at 0.05
        qty = engine._compute_qty(signal, equity=10_000, broker=None, n_tickers=5)
        assert qty == pytest.approx(0.05 * 10_000 / 100.0, rel=1e-6)

    def test_ensemble_mode_still_uses_kelly(self):
        from src.live.engine import LiveEngine

        engine = LiveEngine(strategy_mode="ensemble", max_position_frac=0.05)
        signal = LiveSignal(
            ticker="X", date=mock.Mock(), direction=2, confidence=0.5,
            p_up=0.5, p_flat=0.3, p_down=0.2,
            kelly_frac=0.02, atr=1.0, stop_loss=95.0, close=100.0,
        )
        qty = engine._compute_qty(signal, equity=10_000, broker=None, n_tickers=5)
        assert qty == pytest.approx(0.02 * 10_000 / 100.0, rel=1e-6)

    def test_invalid_strategy_mode_rejected(self):
        from src.live.engine import LiveEngine
        with pytest.raises(ValueError, match="strategy_mode"):
            LiveEngine(strategy_mode="bogus")
