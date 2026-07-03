"""
Phase 06D — Tests for the Regime-Gated Meta-Strategy

Coverage
--------
  _sharpe            — degenerate inputs
  MetaStrategy.run_ticker
    - output shapes and alignment
    - burn-in produces equal-weight ("EW")
    - allocator is strictly walk-forward (future data cannot change past picks)
    - regime gating actually selects the in-regime winner
    - switching cost applied on transitions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.meta_strategy import MetaStrategy, _sharpe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")


def _synthetic(n: int = 500, seed: int = 42):
    """
    Two strategies with regime-dependent edge:
      strat_bull earns +0.2%/day in regime 2, -0.1% otherwise
      strat_bear earns +0.2%/day in regime 0, -0.1% otherwise
    Regime alternates in blocks of 50 bars: 2,0,2,0,...
    """
    rng = np.random.default_rng(seed)
    idx = _dates(n)
    regime = pd.Series(
        [2 if (i // 50) % 2 == 0 else 0 for i in range(n)],
        index=idx, dtype=int,
    )
    noise = lambda: rng.normal(0, 0.001, n)
    bull = np.where(regime.values == 2,  0.002, -0.001) + noise()
    bear = np.where(regime.values == 0,  0.002, -0.001) + noise()
    R = pd.DataFrame({"strat_bull": bull, "strat_bear": bear}, index=idx)
    return R, regime


# ---------------------------------------------------------------------------
# _sharpe helper
# ---------------------------------------------------------------------------

class TestSharpeHelper:
    def test_empty_returns_neg_inf(self):
        assert _sharpe(np.array([]), 0.0002) == -np.inf

    def test_single_obs_neg_inf(self):
        assert _sharpe(np.array([0.01]), 0.0002) == -np.inf

    def test_zero_variance_neg_inf(self):
        assert _sharpe(np.full(100, 0.001), 0.0002) == -np.inf

    def test_positive_sharpe(self):
        rng = np.random.default_rng(0)
        rets = rng.normal(0.001, 0.01, 1000)
        assert _sharpe(rets, 0.0) > 0


# ---------------------------------------------------------------------------
# run_ticker
# ---------------------------------------------------------------------------

class TestRunTicker:
    def test_output_shapes(self):
        R, g = _synthetic()
        meta = MetaStrategy(min_regime_obs=30, rebalance_every=21)
        ret, alloc = meta.run_ticker(R, g)
        assert len(ret) == len(R)
        assert len(alloc) == len(R)
        assert (ret.index == R.index).all()

    def test_burn_in_equal_weight(self):
        R, g = _synthetic()
        meta = MetaStrategy(min_regime_obs=30)
        ret, alloc = meta.run_ticker(R, g)
        # First bars must be EW (not enough in-regime history)
        assert alloc.iloc[0] == "EW"
        assert (alloc.iloc[:10] == "EW").all()

    def test_eventually_selects_strategies(self):
        R, g = _synthetic()
        meta = MetaStrategy(min_regime_obs=30)
        _, alloc = meta.run_ticker(R, g)
        selected = set(alloc.unique()) - {"EW"}
        assert len(selected) >= 1, "allocator never left burn-in"

    def test_regime_gating_picks_in_regime_winner(self):
        """After burn-in, in regime 2 the allocator should hold strat_bull
        and in regime 0 it should hold strat_bear (majority of the time)."""
        R, g = _synthetic(n=800)
        meta = MetaStrategy(min_regime_obs=30, rebalance_every=10)
        _, alloc = meta.run_ticker(R, g)

        # Look at the second half (well past burn-in)
        half   = len(R) // 2
        a      = alloc.iloc[half:]
        # regime known at t is g[t-1]; approximate with same-day state — block
        # structure (50 bars) makes off-by-one negligible
        st     = g.iloc[half:]

        bull_bars = a[(st == 2) & (a != "EW")]
        bear_bars = a[(st == 0) & (a != "EW")]
        assert (bull_bars == "strat_bull").mean() > 0.7
        assert (bear_bars == "strat_bear").mean() > 0.7

    def test_walk_forward_no_lookahead(self):
        """Changing FUTURE strategy returns must not change PAST allocations."""
        R, g = _synthetic()
        meta = MetaStrategy(min_regime_obs=30)

        _, alloc_base = meta.run_ticker(R, g)

        R2 = R.copy()
        R2.iloc[-100:] = R2.iloc[-100:] * -5.0     # corrupt the future
        _, alloc_mod = meta.run_ticker(R2, g)

        cutoff = len(R) - 100
        pd.testing.assert_series_equal(
            alloc_base.iloc[:cutoff], alloc_mod.iloc[:cutoff]
        )

    def test_switch_cost_reduces_returns(self):
        R, g = _synthetic()
        free   = MetaStrategy(min_regime_obs=30, switch_cost=0.0)
        costly = MetaStrategy(min_regime_obs=30, switch_cost=0.01)
        ret_free, _   = free.run_ticker(R, g)
        ret_costly, _ = costly.run_ticker(R, g)
        assert ret_costly.sum() < ret_free.sum()

    def test_meta_beats_static_average_on_synthetic(self):
        """On regime-dependent synthetic data the meta-strategy must clearly
        beat the naive average of both strategies (which nets ≈ +0.05%/day)."""
        R, g = _synthetic(n=1000)
        meta = MetaStrategy(min_regime_obs=30, rebalance_every=10, switch_cost=0.0)
        ret, _ = meta.run_ticker(R, g)
        naive = R.mean(axis=1)
        # Compare over post-burn-in window
        assert ret.iloc[200:].mean() > naive.iloc[200:].mean()

    def test_handles_unknown_regime_state(self):
        """Regime value -1 (unmapped) must not crash the allocator."""
        R, g = _synthetic()
        g.iloc[:20] = -1
        meta = MetaStrategy(min_regime_obs=30)
        ret, alloc = meta.run_ticker(R, g)
        assert len(ret) == len(R)
        assert not ret.isna().any()
