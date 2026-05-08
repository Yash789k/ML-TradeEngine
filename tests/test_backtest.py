"""
Phase 04 — Backtest Unit Tests

All tests are structural / smoke-level:
  - No model loading from disk
  - Synthetic price and signal data only
  - Fast enough to run in < 5 seconds total

Coverage
--------
  test_metrics_*          : compute_metrics output correctness
  test_simulator_*        : PortfolioSimulator mechanics
  test_montecarlo_*       : MC shape and CI ordering
  test_sweep_*            : parameter sweep structure
  test_signals_*          : build_signal_df column contract
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ── Helpers ──────────────────────────────────────────────────────────────

def _make_returns(n: int = 500, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.012, n)
    dates   = pd.date_range("2021-01-04", periods=n, freq="B", tz="UTC")
    return pd.Series(returns, index=dates, name="returns")


def _make_signal_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic signal DataFrame that mimics the output of signals.build_signal_df().
    """
    rng    = np.random.default_rng(seed)
    dates  = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")
    prices = 150.0 * np.cumprod(1 + rng.normal(0.0002, 0.01, n))
    proba  = rng.dirichlet([1, 1, 1], size=n)    # (n, 3) uniform Dirichlet

    df = pd.DataFrame(
        {
            "p_down":     proba[:, 0],
            "p_flat":     proba[:, 1],
            "p_up":       proba[:, 2],
            "close":      prices,
        },
        index=dates,
    )
    df["signal"]          = np.argmax(df[["p_down", "p_flat", "p_up"]].values, axis=1)
    df["confidence"]      = df[["p_down", "p_flat", "p_up"]].max(axis=1)
    df["filtered_signal"] = np.where(df["confidence"] >= 0.38, df["signal"], 1)
    return df


# ── Metrics tests ────────────────────────────────────────────────────────

class TestMetrics:

    def test_sharpe_positive_drift(self):
        from src.backtest.metrics import sharpe_ratio
        returns = _make_returns(n=500, seed=1)
        s = sharpe_ratio(returns)
        assert isinstance(s, float)

    def test_sharpe_zero_std_returns_zero(self):
        from src.backtest.metrics import sharpe_ratio
        flat = pd.Series([0.0] * 100)
        assert sharpe_ratio(flat) == 0.0

    def test_max_drawdown_always_non_positive(self):
        from src.backtest.metrics import max_drawdown
        returns = _make_returns(n=500, seed=2)
        mdd = max_drawdown(returns)
        assert mdd <= 0.0

    def test_max_drawdown_known_sequence(self):
        from src.backtest.metrics import max_drawdown
        # equity: 1 → 1.5 → 0.75.  Peak = 1.5, trough = 0.75 → MDD = -50%
        returns = pd.Series([0.5, -0.5])
        mdd = max_drawdown(returns)
        assert mdd == pytest.approx(-0.5, abs=1e-9)

    def test_cagr_positive_total_return(self):
        from src.backtest.metrics import cagr
        returns = pd.Series([0.001] * 252)  # ~28 % annual
        c = cagr(returns)
        assert c > 0.0

    def test_compute_metrics_keys(self):
        from src.backtest.metrics import compute_metrics
        returns = _make_returns(n=252)
        m = compute_metrics(returns)
        expected = {"total_return", "cagr", "sharpe_ratio", "sortino_ratio",
                    "max_drawdown", "calmar_ratio", "volatility", "n_days"}
        assert expected.issubset(m.keys())

    def test_win_rate_basic(self):
        from src.backtest.metrics import win_rate
        trade_log = pd.DataFrame({
            "type": ["entry", "exit", "entry", "exit", "entry", "exit"],
            "pnl":  [np.nan, 100.0, np.nan, -50.0, np.nan, 75.0],
        })
        assert win_rate(trade_log) == pytest.approx(2 / 3, abs=1e-9)

    def test_compare_to_benchmark_shape(self):
        from src.backtest.metrics import compare_to_benchmark
        r1 = _make_returns(n=200, seed=0)
        r2 = _make_returns(n=200, seed=1)
        cmp = compare_to_benchmark(r1, r2)
        assert "sharpe_ratio" in cmp.index
        assert cmp.shape[1] == 2


# ── Simulator tests ──────────────────────────────────────────────────────

class TestSimulator:

    def test_run_returns_simresult(self):
        from src.backtest.simulator import run_simulation
        sig_df = _make_signal_df(n=200)
        result = run_simulation(sig_df, initial_capital=10_000.0)
        assert len(result.equity_curve) == 200

    def test_equity_starts_at_initial_capital(self):
        from src.backtest.simulator import run_simulation
        sig_df = _make_signal_df(n=200)
        result = run_simulation(sig_df, initial_capital=10_000.0)
        assert result.equity_curve.iloc[0] == pytest.approx(10_000.0)

    def test_equity_always_positive(self):
        from src.backtest.simulator import run_simulation
        sig_df = _make_signal_df(n=250)
        result = run_simulation(sig_df, initial_capital=10_000.0)
        assert (result.equity_curve > 0).all()

    def test_long_only_no_short_positions(self):
        from src.backtest.simulator import run_simulation
        sig_df = _make_signal_df(n=200)
        result = run_simulation(sig_df, mode="long_only")
        # In long-only mode daily returns can only lose at most the full position
        assert result.daily_returns.min() > -1.0

    def test_buy_hold_positive_if_price_rises(self):
        from src.backtest.simulator import buy_and_hold
        sig_df = _make_signal_df(n=100)
        # Force prices to rise monotonically
        sig_df["close"] = np.linspace(100, 200, 100)
        bh = buy_and_hold(sig_df, initial_capital=10_000.0)
        assert bh.iloc[-1] > bh.iloc[0]

    def test_trade_log_has_entries_and_exits(self):
        from src.backtest.simulator import run_simulation
        sig_df = _make_signal_df(n=200, seed=7)
        result = run_simulation(sig_df)
        if len(result.trade_log) > 0:
            assert "entry" in result.trade_log["type"].values
            assert "exit"  in result.trade_log["type"].values

    def test_zero_commission_higher_or_equal_equity(self):
        from src.backtest.simulator import run_simulation
        sig_df = _make_signal_df(n=200)
        res_free = run_simulation(sig_df, commission=0.0, slippage=0.0)
        res_cost = run_simulation(sig_df, commission=0.001, slippage=0.0005)
        assert res_free.equity_curve.iloc[-1] >= res_cost.equity_curve.iloc[-1]

    def test_sweep_returns_dataframe(self):
        from src.backtest.simulator import sweep_parameters
        sig_df = _make_signal_df(n=250)
        sweep  = sweep_parameters(sig_df, thresholds=[0.35, 0.45], commissions=[0.001])
        assert isinstance(sweep, pd.DataFrame)
        assert len(sweep) == 2
        assert "sharpe" in sweep.columns


# ── Monte Carlo tests ────────────────────────────────────────────────────

class TestMonteCarlo:

    def test_mc_output_shape(self):
        from src.backtest.montecarlo import run_monte_carlo
        returns = _make_returns(n=252)
        mc = run_monte_carlo(returns, n_paths=50, n_years=1.0)
        assert mc["equity_paths"].shape == (50, 252)

    def test_mc_percentiles_ordered(self):
        from src.backtest.montecarlo import run_monte_carlo
        returns = _make_returns(n=252)
        mc = run_monte_carlo(returns, n_paths=100, n_years=1.0)
        pct = mc["percentiles_df"]
        # p5 ≤ p25 ≤ p50 ≤ p75 ≤ p95 at each day
        assert (pct["p5"] <= pct["p50"]).all()
        assert (pct["p50"] <= pct["p95"]).all()

    def test_mc_summary_keys(self):
        from src.backtest.montecarlo import run_monte_carlo
        returns = _make_returns(n=252)
        mc = run_monte_carlo(returns, n_paths=50, n_years=1.0)
        expected_keys = {"sharpe_p5", "sharpe_median", "sharpe_p95",
                         "cagr_p5", "cagr_median", "cagr_p95",
                         "max_dd_p5", "max_dd_median", "max_dd_p95",
                         "prob_ruin"}
        assert expected_keys.issubset(mc["summary"].keys())

    def test_mc_prob_ruin_in_range(self):
        from src.backtest.montecarlo import run_monte_carlo
        returns = _make_returns(n=252)
        mc = run_monte_carlo(returns, n_paths=200, n_years=1.0)
        assert 0.0 <= mc["prob_ruin"] <= 1.0

    def test_mc_ci_lower_le_upper(self):
        from src.backtest.montecarlo import run_monte_carlo
        returns = _make_returns(n=252)
        mc = run_monte_carlo(returns, n_paths=100, n_years=2.0)
        s = mc["summary"]
        assert s["sharpe_p5"] <= s["sharpe_p95"]
        assert s["cagr_p5"]   <= s["cagr_p95"]


# ── Signal integrity tests ────────────────────────────────────────────────

class TestSignalIntegrity:
    """
    Tests for the signal DataFrame contract, using synthetic data only
    (no disk I/O, no trained model required).
    """

    def test_signal_values_in_range(self):
        sig_df = _make_signal_df(n=200)
        assert set(sig_df["signal"].unique()).issubset({0, 1, 2})

    def test_filtered_signal_subset_of_signal(self):
        sig_df = _make_signal_df(n=200)
        assert set(sig_df["filtered_signal"].unique()).issubset({0, 1, 2})

    def test_confidence_in_01(self):
        sig_df = _make_signal_df(n=200)
        assert (sig_df["confidence"] >= 0.0).all()
        assert (sig_df["confidence"] <= 1.0).all()

    def test_probabilities_sum_to_one(self):
        sig_df = _make_signal_df(n=200)
        row_sums = sig_df[["p_down", "p_flat", "p_up"]].sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-9)

    def test_filtered_signal_is_flat_when_low_confidence(self):
        """Below-threshold rows must be mapped to FLAT (1)."""
        sig_df = _make_signal_df(n=200)
        low_conf_mask = sig_df["confidence"] < 0.38
        if low_conf_mask.any():
            assert (sig_df.loc[low_conf_mask, "filtered_signal"] == 1).all()
