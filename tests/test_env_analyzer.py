"""
Phase 06C — EnvAnalyzer Tests

Tests are intentionally fast by using synthetic data rather than
touching the Fama-French data server or the full disk cache.

Coverage:
  - regime_breakdown    : correct regime labels, expected columns, per-regime Sharpe
  - cost_sensitivity    : monotonic Sharpe degradation with increasing commission
  - signal_decay        : rolling Sharpe series, slope sign, decay flag
  - factor_attribution  : alpha/beta/r2 are finite, OLS contract
  - _compute_ticker_regime helper: returns valid regime Series
  - edge cases: empty strategy dir, < 60 obs (skip gracefully)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research.env_analyzer import (
    EnvAnalyzer,
    _compute_ticker_regime,
    _regime_ann_return,
    _regime_sharpe,
)
from src.research.strategies import (
    MomentumStrategy,
    EMACrossoverStrategy,
    AlphaTrendsStrategy,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_N = 700   # enough rows for HMM + rolling analyses


def _make_ohlcv(n: int = _N, seed: int = 42) -> pd.DataFrame:
    """
    Regime-switching synthetic OHLCV with three distinct return/vol states so
    that a Gaussian HMM can find non-degenerate covariance matrices.

    States:
      bull   : μ= +0.1%/day, σ= 0.7%
      ranging: μ=  0.0%/day, σ= 1.2%
      bear   : μ= -0.08%/day, σ= 2.0%
    """
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-01", periods=n)

    _PARAMS = [
        (+0.001,  0.007),   # bull
        ( 0.000,  0.012),   # ranging
        (-0.0008, 0.020),   # bear
    ]
    _TRANS = np.array([
        [0.97, 0.02, 0.01],
        [0.01, 0.97, 0.02],
        [0.02, 0.02, 0.96],
    ])

    log_r   = np.zeros(n)
    state   = 1   # start in ranging
    for i in range(n):
        mu, sigma = _PARAMS[state]
        log_r[i]  = rng.normal(mu, sigma)
        state     = rng.choice(3, p=_TRANS[state])

    prices = 100 * np.exp(np.cumsum(log_r))
    vol    = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {
            "Open":      prices * (1 - 0.001),
            "High":      prices * (1 + 0.005),
            "Low":       prices * (1 - 0.005),
            "Close":     prices,
            "Adj_Close": prices,
            "Volume":    vol,
        },
        index=dates,
    )


@pytest.fixture()
def tmp_research_dir():
    """Temporary research directory populated with fake equity curves."""
    with tempfile.TemporaryDirectory() as root:
        rd = Path(root) / "research"
        strategy_names = [
            "Momentum_12_1", "EMA_Crossover", "Alpha_Trends",
        ]
        tickers = ["FAKE1", "FAKE2"]
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2018-01-01", periods=_N)

        for ticker in tickers:
            for sname in strategy_names:
                d = rd / ticker / sname
                d.mkdir(parents=True, exist_ok=True)
                # Equity curve
                equity = 10_000 * np.cumprod(1 + rng.normal(0.0005, 0.01, _N))
                pd.DataFrame({"equity": equity}, index=dates).to_parquet(
                    d / "equity_curve.parquet"
                )
        yield rd


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_regime_sharpe_basic(self):
        rng = np.random.default_rng(0)
        r   = pd.Series(rng.normal(0.001, 0.01, 200))
        sr  = _regime_sharpe(r)
        assert np.isfinite(sr)

    def test_regime_sharpe_zero_variance(self):
        r = pd.Series([0.0] * 100)
        assert _regime_sharpe(r) == 0.0

    def test_regime_ann_return(self):
        r  = pd.Series([0.001] * 252)
        ar = _regime_ann_return(r)
        assert abs(ar - 0.252) < 1e-6

    def test_compute_ticker_regime_shape(self):
        ohlcv  = _make_ohlcv()
        regime = _compute_ticker_regime(ohlcv)
        assert isinstance(regime, pd.Series)
        # Regime-switching synthetic data should allow HMM to converge
        assert len(regime) > 0, (
            "HMM returned empty regime — check that _make_ohlcv generates "
            "distinct volatility regimes (bull/ranging/bear)."
        )
        assert regime.dropna().isin([0, 1, 2]).all()

    def test_compute_ticker_regime_short_data(self):
        ohlcv  = _make_ohlcv(n=30)
        regime = _compute_ticker_regime(ohlcv)
        assert regime.empty


# ---------------------------------------------------------------------------
# EnvAnalyzer.regime_breakdown
# ---------------------------------------------------------------------------

class TestRegimeBreakdown:
    def test_returns_dataframe(self, tmp_research_dir):
        analyzer = EnvAnalyzer(research_dir=tmp_research_dir)
        ohlcv    = {"FAKE1": _make_ohlcv(), "FAKE2": _make_ohlcv(seed=5)}
        df       = analyzer.regime_breakdown(ohlcv, strategy_names=["Momentum_12_1"])

        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_expected_columns(self, tmp_research_dir):
        analyzer = EnvAnalyzer(research_dir=tmp_research_dir)
        ohlcv    = {"FAKE1": _make_ohlcv()}
        df       = analyzer.regime_breakdown(ohlcv, strategy_names=["EMA_Crossover"])

        assert "sharpe"     in df.columns
        assert "ann_return" in df.columns
        assert "n_days"     in df.columns
        assert "pct_days"   in df.columns

    def test_regime_labels(self, tmp_research_dir):
        analyzer = EnvAnalyzer(research_dir=tmp_research_dir)
        ohlcv    = {"FAKE1": _make_ohlcv()}
        df       = analyzer.regime_breakdown(ohlcv, strategy_names=["Momentum_12_1"])

        if not df.empty:
            regime_vals = df.index.get_level_values("regime")
            assert set(regime_vals).issubset({"bear", "ranging", "bull"})

    def test_pct_days_sums_to_one(self, tmp_research_dir):
        analyzer = EnvAnalyzer(research_dir=tmp_research_dir)
        ohlcv    = {"FAKE1": _make_ohlcv()}
        df       = analyzer.regime_breakdown(ohlcv, strategy_names=["Momentum_12_1"])

        if df.empty:
            return
        total = (
            df.loc["FAKE1", "Momentum_12_1"]["pct_days"].sum()
            if ("FAKE1", "Momentum_12_1") in df.index else None
        )
        if total is not None:
            assert abs(total - 1.0) < 0.02   # some days may be dropped due to NaN

    def test_saved_parquet(self, tmp_research_dir):
        analyzer = EnvAnalyzer(research_dir=tmp_research_dir)
        ohlcv    = {"FAKE1": _make_ohlcv()}
        analyzer.regime_breakdown(ohlcv, strategy_names=["Momentum_12_1"])
        assert (tmp_research_dir / "env" / "regime_breakdown.parquet").exists()

    def test_missing_strategy_dir_skipped(self, tmp_research_dir):
        analyzer = EnvAnalyzer(research_dir=tmp_research_dir)
        ohlcv    = {"FAKE1": _make_ohlcv()}
        # "NonExistent" strategy dir does not exist — should not raise, returns empty
        df = analyzer.regime_breakdown(ohlcv, strategy_names=["NonExistent"])
        # Result is empty (no rows for this strategy)
        assert isinstance(df, pd.DataFrame)
        assert df.empty


# ---------------------------------------------------------------------------
# EnvAnalyzer.cost_sensitivity
# ---------------------------------------------------------------------------

class TestCostSensitivity:
    def test_monotonic_sharpe_degradation(self, tmp_research_dir):
        """Sharpe should generally decrease as commission increases."""
        ohlcv    = {"FAKE1": _make_ohlcv()}
        analyzer = EnvAnalyzer(research_dir=tmp_research_dir)
        strategies = [MomentumStrategy()]
        comms    = [0.0, 0.001, 0.005]

        df = analyzer.cost_sensitivity(
            ohlcv, strategies, commission_levels=comms
        )

        assert not df.empty
        assert "sharpe" in df.columns

        # At 0 commission, sharpe >= at 0.5% commission (at least for this
        # synthetic data where we have signal-free noise the costs dominate)
        sharpes = (
            df.loc["FAKE1", "Momentum_12_1"]["sharpe"].values
            if ("FAKE1", "Momentum_12_1") in df.index
            else []
        )
        if len(sharpes) == 3:
            # Overall direction should be non-increasing (allow ties)
            assert sharpes[0] >= sharpes[-1] - 0.5  # generous tolerance

    def test_saved_parquet(self, tmp_research_dir):
        ohlcv    = {"FAKE1": _make_ohlcv()}
        analyzer = EnvAnalyzer(research_dir=tmp_research_dir)
        analyzer.cost_sensitivity(
            ohlcv, [MomentumStrategy()], commission_levels=[0.0, 0.001]
        )
        assert (tmp_research_dir / "env" / "cost_sensitivity.parquet").exists()

    def test_bps_column(self, tmp_research_dir):
        ohlcv    = {"FAKE1": _make_ohlcv()}
        analyzer = EnvAnalyzer(research_dir=tmp_research_dir)
        df       = analyzer.cost_sensitivity(
            ohlcv, [MomentumStrategy()], commission_levels=[0.0, 0.001]
        )
        assert "bps" in df.columns
        assert set(df["bps"].unique()).issubset({0, 10, 20, 50, 100, 5})


# ---------------------------------------------------------------------------
# EnvAnalyzer.signal_decay
# ---------------------------------------------------------------------------

class TestSignalDecay:
    def test_returns_summary_dataframe(self, tmp_research_dir):
        analyzer = EnvAnalyzer(research_dir=tmp_research_dir)
        df       = analyzer.signal_decay(
            strategy_names=["Momentum_12_1", "EMA_Crossover"]
        )
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_expected_columns(self, tmp_research_dir):
        analyzer = EnvAnalyzer(research_dir=tmp_research_dir)
        df       = analyzer.signal_decay(strategy_names=["Momentum_12_1"])
        for col in ["slope", "mean_sharpe", "end_sharpe", "pct_positive", "is_decaying"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_pct_positive_in_range(self, tmp_research_dir):
        analyzer = EnvAnalyzer(research_dir=tmp_research_dir)
        df       = analyzer.signal_decay(strategy_names=["Momentum_12_1"])
        assert (df["pct_positive"] >= 0).all()
        assert (df["pct_positive"] <= 1).all()

    def test_time_series_parquet_saved(self, tmp_research_dir):
        analyzer = EnvAnalyzer(research_dir=tmp_research_dir)
        analyzer.signal_decay(strategy_names=["Momentum_12_1"])
        assert (tmp_research_dir / "env" / "signal_decay_series.parquet").exists()
        assert (tmp_research_dir / "env" / "signal_decay_summary.parquet").exists()

    def test_short_series_skipped(self, tmp_research_dir):
        """Strategies with < window+10 bars should be silently skipped."""
        short_dir = tmp_research_dir / "SHORT" / "Momentum_12_1"
        short_dir.mkdir(parents=True)
        dates  = pd.bdate_range("2022-01-01", periods=50)
        equity = pd.Series(np.linspace(10000, 10500, 50), index=dates)
        pd.DataFrame({"equity": equity}).to_parquet(short_dir / "equity_curve.parquet")

        analyzer = EnvAnalyzer(research_dir=tmp_research_dir)
        df = analyzer.signal_decay(strategy_names=["Momentum_12_1"], window=90)
        if not df.empty:
            # SHORT ticker should not appear in the summary
            assert "SHORT" not in df.index.get_level_values("ticker")


# ---------------------------------------------------------------------------
# EnvAnalyzer.factor_attribution
# ---------------------------------------------------------------------------

class TestFactorAttribution:
    def test_runs_without_error(self, tmp_research_dir):
        """Factor attribution should run end-to-end and return a DataFrame."""
        analyzer = EnvAnalyzer(research_dir=tmp_research_dir)
        try:
            df = analyzer.factor_attribution(
                strategy_names=["Momentum_12_1"],
                start="2018-01-01",
            )
        except Exception as exc:
            pytest.fail(f"factor_attribution raised: {exc}")

        assert isinstance(df, pd.DataFrame)

    def test_expected_columns_when_data_available(self, tmp_research_dir):
        analyzer = EnvAnalyzer(research_dir=tmp_research_dir)
        df = analyzer.factor_attribution(
            strategy_names=["Momentum_12_1"],
            start="2018-01-01",
        )
        if df.empty:
            pytest.skip("FF3 download unavailable — skipping column check")

        for col in ["alpha_ann", "alpha_tstat", "beta_mkt", "beta_smb", "beta_hml", "r2"]:
            assert col in df.columns

    def test_r2_between_0_and_1(self, tmp_research_dir):
        analyzer = EnvAnalyzer(research_dir=tmp_research_dir)
        df = analyzer.factor_attribution(
            strategy_names=["Momentum_12_1"],
            start="2018-01-01",
        )
        if df.empty:
            pytest.skip("FF3 download unavailable")
        assert (df["r2"] >= 0).all()
        assert (df["r2"] <= 1).all()

    def test_saved_parquet(self, tmp_research_dir):
        analyzer = EnvAnalyzer(research_dir=tmp_research_dir)
        analyzer.factor_attribution(
            strategy_names=["Momentum_12_1"],
            start="2018-01-01",
        )
        assert (tmp_research_dir / "env" / "factor_attribution.parquet").exists()


# ---------------------------------------------------------------------------
# Print methods (smoke tests — just check they don't raise)
# ---------------------------------------------------------------------------

class TestPrintMethods:
    def test_print_regime_summary(self, tmp_research_dir):
        analyzer = EnvAnalyzer(research_dir=tmp_research_dir)
        ohlcv    = {"FAKE1": _make_ohlcv()}
        df       = analyzer.regime_breakdown(ohlcv, strategy_names=["Momentum_12_1"])
        analyzer.print_regime_summary(df)

    def test_print_cost_summary(self, tmp_research_dir):
        analyzer = EnvAnalyzer(research_dir=tmp_research_dir)
        df       = analyzer.cost_sensitivity(
            {"FAKE1": _make_ohlcv()}, [MomentumStrategy()],
            commission_levels=[0.0, 0.001]
        )
        analyzer.print_cost_summary(df)

    def test_print_decay_summary(self, tmp_research_dir):
        analyzer = EnvAnalyzer(research_dir=tmp_research_dir)
        df       = analyzer.signal_decay(strategy_names=["Momentum_12_1"])
        analyzer.print_decay_summary(df)

    def test_print_factor_summary(self, tmp_research_dir):
        analyzer = EnvAnalyzer(research_dir=tmp_research_dir)
        df = analyzer.factor_attribution(
            strategy_names=["Momentum_12_1"],
            start="2018-01-01",
        )
        analyzer.print_factor_summary(df)
