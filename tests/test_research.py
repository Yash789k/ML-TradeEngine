"""
Phase 06A — Strategy Zoo Tests

Tests are structural/smoke tests using synthetic OHLCV data.
They verify:
  - Signal shape and valid values
  - BaseStrategy.run() returns correct keys
  - ZooRunner aggregates results correctly
  - Ranker produces a ranked DataFrame with required columns
  - No lookahead: signals are not computed from future data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------

def make_ohlcv(
    n: int = 500,
    seed: int = 0,
    freq: str = "B",
    start: str = "2019-01-02",
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a mild upward trend + noise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n, tz="UTC")

    log_ret = rng.normal(0.0003, 0.015, n)  # slight drift
    close   = 100.0 * np.exp(np.cumsum(log_ret))
    spread  = rng.uniform(0.001, 0.003, n)
    volume  = rng.integers(1_000_000, 10_000_000, n).astype(float)

    return pd.DataFrame(
        {
            "Open":      close * (1 - spread),
            "High":      close * (1 + spread * 2),
            "Low":       close * (1 - spread * 2),
            "Close":     close,
            "Adj_Close": close,
            "Volume":    volume,
        },
        index=idx,
    )


def make_macro(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic macro DataFrame aligned with the OHLCV index."""
    n = len(ohlcv)
    spread = np.sin(np.linspace(0, 4 * np.pi, n)) * 1.5  # oscillates ±1.5
    return pd.DataFrame(
        {
            "VIX":               15 + np.abs(spread),
            "yield_spread_10_2": spread,
            "CPI":               250.0 + np.arange(n) * 0.01,
            "rate_10y":          3.0 + spread * 0.2,
            "rate_2y":           2.5 + spread * 0.1,
        },
        index=ohlcv.index,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ohlcv():
    return make_ohlcv(n=600)


@pytest.fixture(scope="module")
def spy_df():
    return make_ohlcv(n=600, seed=1)


@pytest.fixture(scope="module")
def macro(ohlcv):
    return make_macro(ohlcv)


# ---------------------------------------------------------------------------
# BaseStrategy contract
# ---------------------------------------------------------------------------

class TestBaseStrategyContract:

    def _assert_signal_valid(self, signals: pd.Series, ohlcv: pd.DataFrame) -> None:
        assert isinstance(signals, pd.Series), "generate_signals must return pd.Series"
        assert len(signals) == len(ohlcv), "Signal length must match OHLCV"
        assert set(signals.unique()).issubset({-1, 0, 1}), (
            f"Signal values must be in {{-1, 0, 1}}, got {set(signals.unique())}"
        )
        assert not signals.isna().any(), "Signals must not contain NaN"

    def _assert_run_result(self, result: dict) -> None:
        assert "metrics"   in result
        assert "result"    in result
        assert "strategy"  in result
        assert "sharpe_ratio" in result["metrics"]
        assert "cagr"         in result["metrics"]
        assert "max_drawdown" in result["metrics"]

    def test_momentum_signals(self, ohlcv):
        from src.research.strategies.momentum import MomentumStrategy
        s = MomentumStrategy()
        sig = s.generate_signals(ohlcv)
        self._assert_signal_valid(sig, ohlcv)

    def test_mean_reversion_signals(self, ohlcv):
        from src.research.strategies.mean_reversion import MeanReversionStrategy
        s = MeanReversionStrategy()
        sig = s.generate_signals(ohlcv)
        self._assert_signal_valid(sig, ohlcv)

    def test_ema_crossover_signals(self, ohlcv):
        from src.research.strategies.ema_crossover import EMACrossoverStrategy
        s = EMACrossoverStrategy()
        sig = s.generate_signals(ohlcv)
        self._assert_signal_valid(sig, ohlcv)

    def test_turtle_signals(self, ohlcv):
        from src.research.strategies.turtle import TurtleStrategy
        s = TurtleStrategy()
        sig = s.generate_signals(ohlcv)
        self._assert_signal_valid(sig, ohlcv)

    def test_pairs_arb_no_pair(self, ohlcv):
        """Without pair_df the strategy must return all-flat, not crash."""
        from src.research.strategies.pairs_arb import PairsArbStrategy
        s = PairsArbStrategy()
        sig = s.generate_signals(ohlcv)
        assert (sig == 0).all(), "PairsArb without pair_df must be all-flat"

    def test_pairs_arb_with_pair(self, ohlcv, spy_df):
        from src.research.strategies.pairs_arb import PairsArbStrategy
        s = PairsArbStrategy()
        sig = s.generate_signals(ohlcv, pair_df=spy_df)
        self._assert_signal_valid(sig, ohlcv)

    def test_carry_proxy_no_macro(self, ohlcv):
        """Without macro, strategy falls back to pure MA trend — must not crash."""
        from src.research.strategies.carry_proxy import CarryProxyStrategy
        s = CarryProxyStrategy()
        sig = s.generate_signals(ohlcv, macro=None)
        self._assert_signal_valid(sig, ohlcv)

    def test_carry_proxy_with_macro(self, ohlcv, macro):
        from src.research.strategies.carry_proxy import CarryProxyStrategy
        s = CarryProxyStrategy()
        sig = s.generate_signals(ohlcv, macro=macro)
        self._assert_signal_valid(sig, ohlcv)

    def test_vol_breakout_signals(self, ohlcv):
        from src.research.strategies.vol_breakout import VolBreakoutStrategy
        s = VolBreakoutStrategy()
        sig = s.generate_signals(ohlcv)
        self._assert_signal_valid(sig, ohlcv)

    def test_alpha_trends_signals(self, ohlcv):
        from src.research.strategies.alpha_trends import AlphaTrendsStrategy
        s = AlphaTrendsStrategy()
        sig = s.generate_signals(ohlcv)
        self._assert_signal_valid(sig, ohlcv)

    def test_alpha_trends_insufficient_data(self):
        """AlphaTrends should return all-flat when data is too short for HMM."""
        from src.research.strategies.alpha_trends import AlphaTrendsStrategy
        tiny_ohlcv = make_ohlcv(n=20)
        s = AlphaTrendsStrategy()
        sig = s.generate_signals(tiny_ohlcv)
        assert (sig == 0).all()

    def test_run_returns_required_keys(self, ohlcv):
        from src.research.strategies.momentum import MomentumStrategy
        s = MomentumStrategy()
        result = s.run(ohlcv)
        self._assert_run_result(result)

    def test_equity_curve_length(self, ohlcv):
        from src.research.strategies.ema_crossover import EMACrossoverStrategy
        s = EMACrossoverStrategy()
        result = s.run(ohlcv)
        ec = result["result"].equity_curve
        assert len(ec) == len(ohlcv)

    def test_equity_curve_positive(self, ohlcv):
        """Equity must remain non-negative (no short-selling loss explosions)."""
        from src.research.strategies.turtle import TurtleStrategy
        s = TurtleStrategy()
        result = s.run(ohlcv)
        assert (result["result"].equity_curve >= 0).all()

    def test_long_only_no_short_positions(self, ohlcv):
        """Long-only strategy must never have target=-1."""
        from src.research.strategies.momentum import MomentumStrategy
        s = MomentumStrategy()
        sdf = s.build_signal_df(ohlcv)
        # encoded value 0 = short in simulator; must not appear in long_only
        assert (sdf["filtered_signal"] != 0).all()


# ---------------------------------------------------------------------------
# No-lookahead check
# ---------------------------------------------------------------------------

class TestNoLookahead:

    def test_momentum_signal_uses_only_past(self, ohlcv):
        """
        Flip the last 50 rows of close to a very different value.
        The first-half signals must be unchanged (no lookahead contamination).
        """
        from src.research.strategies.momentum import MomentumStrategy
        s  = MomentumStrategy()
        original_sigs = s.generate_signals(ohlcv)

        modified        = ohlcv.copy()
        modified.loc[modified.index[-50:], "Close"]     = 1.0
        modified.loc[modified.index[-50:], "Adj_Close"] = 1.0
        modified_sigs   = s.generate_signals(modified)

        n = len(ohlcv) - 50 - 252  # safe buffer before the modified region
        if n > 0:
            pd.testing.assert_series_equal(
                original_sigs.iloc[:n].reset_index(drop=True),
                modified_sigs.iloc[:n].reset_index(drop=True),
            )


# ---------------------------------------------------------------------------
# ZooRunner smoke test
# ---------------------------------------------------------------------------

class TestZooRunner:

    def test_zoo_runner_returns_scorecard(self, ohlcv, spy_df, macro, tmp_path):
        from src.research.strategies.momentum import MomentumStrategy
        from src.research.strategies.alpha_trends import AlphaTrendsStrategy
        from src.research.zoo_runner import ZooRunner

        runner = ZooRunner(
            strategies=[MomentumStrategy(), AlphaTrendsStrategy()],
            output_dir=tmp_path,
        )
        scorecard = runner.run(
            ohlcv_dict={"SYNTHETIC": ohlcv},
            spy_df=spy_df,
            macro=macro,
            save=True,
        )
        assert isinstance(scorecard, pd.DataFrame)
        assert len(scorecard) == 2   # 1 ticker × 2 strategies
        assert "sharpe_ratio" in scorecard.columns

    def test_zoo_runner_persists_files(self, ohlcv, spy_df, macro, tmp_path):
        from src.research.strategies.ema_crossover import EMACrossoverStrategy
        from src.research.zoo_runner import ZooRunner

        runner = ZooRunner(
            strategies=[EMACrossoverStrategy()],
            output_dir=tmp_path,
        )
        runner.run(
            ohlcv_dict={"SYNTHETIC": ohlcv},
            spy_df=spy_df,
            macro=macro,
            save=True,
        )
        ec_path = tmp_path / "SYNTHETIC" / "EMA_Crossover" / "equity_curve.parquet"
        assert ec_path.exists(), f"Expected equity_curve.parquet at {ec_path}"

    def test_zoo_runner_all_strategies(self, ohlcv, spy_df, macro, tmp_path):
        """Smoke test: all 8 strategies must complete without exception."""
        from src.research.strategies import ALL_STRATEGIES
        from src.research.zoo_runner import ZooRunner

        runner = ZooRunner(strategies=ALL_STRATEGIES, output_dir=tmp_path)
        scorecard = runner.run(
            ohlcv_dict={"SYNTHETIC": ohlcv},
            spy_df=spy_df,
            macro=macro,
            save=True,
        )
        assert len(scorecard) == len(ALL_STRATEGIES)
        assert scorecard.index.get_level_values("ticker").unique().tolist() == ["SYNTHETIC"]


# ---------------------------------------------------------------------------
# Ranker smoke test
# ---------------------------------------------------------------------------

class TestRanker:

    def test_ranker_produces_score_column(self, ohlcv, spy_df, macro, tmp_path):
        from src.research.strategies import ALL_STRATEGIES
        from src.research.zoo_runner import ZooRunner
        from src.research.ranker import Ranker

        runner = ZooRunner(strategies=ALL_STRATEGIES, output_dir=tmp_path)
        runner.run(
            ohlcv_dict={"SYNTHETIC": ohlcv},
            spy_df=spy_df,
            macro=macro,
            save=True,
        )

        # Inject SPY parquet for ranker benchmark
        spy_parquet = tmp_path / "SPY_daily.parquet"
        spy_df.to_parquet(spy_parquet)

        ranker  = Ranker(research_dir=tmp_path, parquet_dir=tmp_path)
        ranked  = ranker.rank()

        assert "score" in ranked.columns
        assert "alpha" in ranked.columns
        assert "beta"  in ranked.columns
        assert "t_stat" in ranked.columns
        assert ranked["score"].notna().all()
        assert ranked.index.names == ["ticker", "strategy"]

    def test_ranker_sorted_descending(self, ohlcv, spy_df, macro, tmp_path):
        from src.research.strategies import ALL_STRATEGIES
        from src.research.zoo_runner import ZooRunner
        from src.research.ranker import Ranker

        runner = ZooRunner(strategies=ALL_STRATEGIES, output_dir=tmp_path)
        runner.run({"SYNTHETIC": ohlcv}, spy_df=spy_df, macro=macro, save=True)

        spy_parquet = tmp_path / "SPY_daily.parquet"
        spy_df.to_parquet(spy_parquet)

        ranker = Ranker(research_dir=tmp_path, parquet_dir=tmp_path)
        ranked = ranker.rank()
        scores = ranked["score"].tolist()
        assert scores == sorted(scores, reverse=True), "Ranked scores must be descending"
