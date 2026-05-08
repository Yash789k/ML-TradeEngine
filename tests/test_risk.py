"""
Phase 05 — Risk Management Unit Tests

All tests are synthetic / structural — no trained models or disk I/O required.
Runs in < 2 seconds.

Coverage
--------
  TestKellySizing      : Kelly fraction correctness, edge cases, vectorised
  TestATRStops         : trailing stop direction, ratchet, stop-hit detection
  TestVaR              : VaR / CVaR numerical correctness, rolling series shape
  TestPortfolio        : HeatTracker mechanics, circuit breaker logic
  TestRiskEngine       : smoke-test with synthetic signals and features
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_returns(n: int = 252, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")
    return pd.Series(rng.normal(0.0003, 0.012, n), index=idx, name="ret")


def _make_signal_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")
    proba = rng.dirichlet([1, 1, 1], size=n)
    prices = 150.0 * np.cumprod(1 + rng.normal(0.0002, 0.01, n))
    df = pd.DataFrame(
        {"p_down": proba[:, 0], "p_flat": proba[:, 1], "p_up": proba[:, 2],
         "close": prices},
        index=dates,
    )
    df["signal"]          = np.argmax(df[["p_down", "p_flat", "p_up"]].values, axis=1)
    df["confidence"]      = df[["p_down", "p_flat", "p_up"]].max(axis=1)
    df["filtered_signal"] = np.where(df["confidence"] >= 0.38, df["signal"], 1)
    return df


def _make_feat_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")
    prices = 150.0 * np.cumprod(1 + rng.normal(0.0002, 0.01, n))
    atr    = rng.uniform(1.0, 5.0, n)
    return pd.DataFrame({"Close": prices, "atr_14": atr}, index=dates)


# ── Kelly Sizing ─────────────────────────────────────────────────────────

class TestKellySizing:

    def test_kelly_positive_edge(self):
        from src.risk.sizing import kelly_fraction
        f = kelly_fraction(p_win=0.6, win_loss_ratio=2.0)
        assert f > 0.0

    def test_kelly_zero_edge_returns_zero(self):
        from src.risk.sizing import kelly_fraction
        # p=0.33, b=2: f* = (0.33*2 - 0.67)/2 = -0.005 → 0
        f = kelly_fraction(p_win=0.33, win_loss_ratio=2.0)
        assert f == 0.0

    def test_kelly_half_multiplier(self):
        from src.risk.sizing import kelly_fraction
        full = kelly_fraction(p_win=0.6, win_loss_ratio=2.0, kelly_multiplier=1.0)
        half = kelly_fraction(p_win=0.6, win_loss_ratio=2.0, kelly_multiplier=0.5)
        assert half == pytest.approx(full / 2, rel=1e-6)

    def test_kelly_invalid_inputs_return_zero(self):
        from src.risk.sizing import kelly_fraction
        assert kelly_fraction(p_win=0.0, win_loss_ratio=2.0) == 0.0
        assert kelly_fraction(p_win=0.6, win_loss_ratio=0.0) == 0.0
        assert kelly_fraction(p_win=1.0, win_loss_ratio=2.0) == 0.0

    def test_kelly_series_shape(self):
        from src.risk.sizing import kelly_series
        p = np.array([0.4, 0.5, 0.6, 0.7])
        k = kelly_series(p, win_loss_ratio=2.0)
        assert k.shape == (4,)
        assert (k >= 0).all()

    def test_atr_scale_high_vol_reduces_size(self):
        from src.risk.sizing import atr_scale_series
        n   = 100
        atr = np.full(n, 1.0)
        atr[-1] = 4.0               # last bar is 4× normal volatility
        scale = atr_scale_series(atr, window=50)
        # Last bar should have smaller scale (more volatile)
        assert scale[-1] < scale[-2]

    def test_position_fractions_bounded(self):
        from src.risk.sizing import position_fractions
        rng = np.random.default_rng(0)
        p   = rng.uniform(0.3, 0.7, 200)
        atr = rng.uniform(1.0, 5.0, 200)
        f   = position_fractions(p, atr, win_loss_ratio=2.0, max_position=0.20)
        assert (f >= 0).all()
        assert (f <= 0.201).all()    # small float tolerance


# ── ATR Trailing Stops ───────────────────────────────────────────────────

class TestATRStops:

    def test_initial_stop_below_entry_long(self):
        from src.risk.stops import initial_stop
        s = initial_stop(entry_price=100.0, atr=2.0, direction=1)
        assert s == pytest.approx(96.0)

    def test_initial_stop_above_entry_short(self):
        from src.risk.stops import initial_stop
        s = initial_stop(entry_price=100.0, atr=2.0, direction=-1)
        assert s == pytest.approx(104.0)

    def test_trailing_stop_only_rises_long(self):
        from src.risk.stops import StopState, update_trailing_stop
        state = StopState(active=True, stop_level=90.0, direction=1)
        update_trailing_stop(state, current_price=102.0, current_atr=2.0)
        assert state.stop_level >= 90.0          # never goes down for longs

    def test_stop_hit_below_level(self):
        from src.risk.stops import StopState, is_stop_hit
        state = StopState(active=True, stop_level=95.0, direction=1)
        assert is_stop_hit(state, current_price=94.9)

    def test_stop_not_hit_above_level(self):
        from src.risk.stops import StopState, is_stop_hit
        state = StopState(active=True, stop_level=95.0, direction=1)
        assert not is_stop_hit(state, current_price=95.1)

    def test_inactive_stop_never_triggers(self):
        from src.risk.stops import StopState, is_stop_hit
        state = StopState(active=False, stop_level=999.0, direction=1)
        assert not is_stop_hit(state, current_price=1.0)

    def test_compute_trailing_stops_ratchets_up(self):
        from src.risk.stops import compute_trailing_stops
        closes = np.array([100.0, 105.0, 110.0, 108.0, 115.0])
        atrs   = np.array([2.0,   2.0,   2.0,   2.0,   2.0])
        stops  = compute_trailing_stops(closes, atrs, entry_idx=0)
        # Stops at idx 0: 96.0; at idx 2 (close=110): 106.0; must not fall after
        assert stops[2] >= stops[1]
        assert stops[3] >= stops[2]     # ratchet: doesn't drop when price dips


# ── VaR / CVaR ───────────────────────────────────────────────────────────

class TestVaR:

    def test_var_positive_number(self):
        from src.risk.var import var_historical
        returns = np.array([-0.05, -0.02, 0.01, 0.03, -0.01, 0.02])
        v = var_historical(returns, confidence=0.95)
        assert v >= 0.0

    def test_var_larger_at_higher_confidence(self):
        from src.risk.var import var_historical
        r = _make_returns(n=500).values
        v95 = var_historical(r, 0.95)
        v99 = var_historical(r, 0.99)
        assert v99 >= v95

    def test_cvar_ge_var(self):
        from src.risk.var import cvar_historical, var_historical
        r = _make_returns(n=500).values
        v = var_historical(r, 0.99)
        c = cvar_historical(r, 0.99)
        assert c >= v

    def test_rolling_var_shape(self):
        from src.risk.var import rolling_var_cvar
        returns = _make_returns(n=300)
        df = rolling_var_cvar(returns, window=63)
        assert len(df) == 300
        assert "var_95" in df.columns
        assert "cvar_99" in df.columns

    def test_rolling_var_non_negative(self):
        from src.risk.var import rolling_var_cvar
        returns = _make_returns(n=300)
        df = rolling_var_cvar(returns, window=63)
        assert (df["var_95"]  >= 0).all()
        assert (df["cvar_99"] >= 0).all()

    def test_risk_summary_keys(self):
        from src.risk.var import risk_summary
        returns = _make_returns(n=252)
        s = risk_summary(returns)
        expected = {"var_95_pct", "cvar_99_pct", "var_95_dollar",
                    "cvar_99_dollar", "worst_day_pct", "negative_days_pct"}
        assert expected.issubset(s.keys())


# ── Portfolio Controls ───────────────────────────────────────────────────

class TestPortfolio:

    def test_heat_tracker_allows_within_limit(self):
        from src.risk.portfolio import HeatTracker
        h = HeatTracker(max_heat=0.20)
        assert h.can_open("AAPL", 0.15)

    def test_heat_tracker_blocks_over_limit(self):
        from src.risk.portfolio import HeatTracker
        h = HeatTracker(max_heat=0.20)
        h.open("AAPL", 0.18)
        assert not h.can_open("MSFT", 0.10)

    def test_heat_tracker_allows_after_close(self):
        from src.risk.portfolio import HeatTracker
        h = HeatTracker(max_heat=0.20)
        h.open("AAPL", 0.18)
        h.close("AAPL")
        assert h.can_open("MSFT", 0.15)

    def test_circuit_breaker_trips_on_15pct_dd(self):
        from src.risk.portfolio import is_circuit_broken
        assert is_circuit_broken(equity=8499, peak_equity=10_000, threshold=0.85)

    def test_circuit_breaker_not_tripped_on_small_dd(self):
        from src.risk.portfolio import is_circuit_broken
        assert not is_circuit_broken(equity=9500, peak_equity=10_000, threshold=0.85)

    def test_circuit_breaker_resets_on_recovery(self):
        from src.risk.portfolio import circuit_breaker_reset
        assert circuit_breaker_reset(equity=9050, peak_equity=10_000, resume_threshold=0.90)

    def test_circuit_breaker_stays_on_partial_recovery(self):
        from src.risk.portfolio import circuit_breaker_reset
        assert not circuit_breaker_reset(equity=8800, peak_equity=10_000, resume_threshold=0.90)


# ── RiskEngine Smoke Test ─────────────────────────────────────────────────

class TestRiskEngine:

    def test_run_returns_risk_result(self, tmp_path, monkeypatch):
        from src.risk.engine import RiskEngine

        # Patch the backtest dir so _estimate_win_loss_ratio falls back gracefully
        monkeypatch.setattr(
            "src.risk.engine._BACKTEST_DIR", tmp_path
        )

        engine   = RiskEngine(initial_capital=10_000.0)
        sig_df   = _make_signal_df(n=200)
        feat_df  = _make_feat_df(n=200)

        result = engine.run("TEST", sig_df, feat_df)

        assert len(result.equity_curve) == 200
        assert len(result.daily_returns) == 200
        assert len(result.var_series) == 200

    def test_equity_always_positive(self, tmp_path, monkeypatch):
        from src.risk.engine import RiskEngine
        monkeypatch.setattr("src.risk.engine._BACKTEST_DIR", tmp_path)

        result = RiskEngine().run("TEST", _make_signal_df(250), _make_feat_df(250))
        assert (result.equity_curve > 0).all()

    def test_equity_starts_at_capital(self, tmp_path, monkeypatch):
        from src.risk.engine import RiskEngine
        monkeypatch.setattr("src.risk.engine._BACKTEST_DIR", tmp_path)

        result = RiskEngine(initial_capital=10_000.0).run(
            "TEST", _make_signal_df(200), _make_feat_df(200)
        )
        assert result.equity_curve.iloc[0] == pytest.approx(10_000.0)

    def test_risk_report_has_var_keys(self, tmp_path, monkeypatch):
        from src.risk.engine import RiskEngine
        monkeypatch.setattr("src.risk.engine._BACKTEST_DIR", tmp_path)

        result = RiskEngine().run("TEST", _make_signal_df(200), _make_feat_df(200))
        assert "var_95_pct"  in result.risk_report
        assert "cvar_99_pct" in result.risk_report
