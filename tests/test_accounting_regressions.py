import numpy as np
import pandas as pd
import pytest
from src.backtest.simulator import run_simulation
from src.backtest.metrics import (
    max_drawdown,
    sortino_ratio,
    compute_metrics,
    compare_to_benchmark,
)


def simulate(prices, signals, **kwargs):
    df = pd.DataFrame(
        {"close": prices, "filtered_signal": signals},
        index=pd.date_range("2024-01-02", periods=len(prices), freq="B"),
    )
    return run_simulation(df, **kwargs)


@pytest.mark.parametrize(
    "prices,signals,mode,expected",
    [
        ([100, 100, 110, 110], [2, 1, 1, 1], "long_only", 11000),
        ([100, 100, 90, 90], [0, 0, 1, 1], "long_short", 11000),
        ([100, 100, 110, 110], [0, 1, 1, 1], "long_short", 9000),
        ([100, 100, 100, 100], [2, 1, 1, 1], "long_only", 10000),
    ],
)
def test_hand_calculated_nav(prices, signals, mode, expected):
    result = simulate(prices, signals, mode=mode, commission=0, slippage=0)
    assert result.equity_curve.iloc[-1] == pytest.approx(expected)
    assert result.trade_log.query('type == "exit"').pnl.sum() == pytest.approx(
        expected - 10000
    )


@pytest.mark.parametrize("direction", [0, 2])
def test_fees_and_cash_reconcile(direction):
    r = simulate([100, 100, 100.2, 100.2], [direction, 1, 1, 1], mode="long_short")
    q = (1 if direction == 2 else -1) * 10000 / (100 * 1.0015)
    expected = q * 0.2 - abs(q) * (100 + 100.2) * 0.0015
    assert r.trade_log.query('type == "exit"').pnl.iloc[0] == pytest.approx(expected)
    assert r.equity_curve.iloc[-1] == pytest.approx(10000 + expected)
    assert compute_metrics(r.daily_returns, r.trade_log)["win_rate"] == 0
    np.testing.assert_allclose(
        r.ledger.equity, r.ledger.cash + r.ledger.shares * r.ledger.price
    )


def test_random_reversals_reconcile_open_and_closed_pnl():
    rng = np.random.default_rng(35)
    r = simulate(
        100 * np.exp(rng.normal(0, 0.02, 150).cumsum()),
        rng.integers(0, 3, 150),
        mode="long_short",
    )
    np.testing.assert_allclose(
        r.ledger.equity, 10000 + r.ledger.realized_pnl + r.ledger.unrealized_pnl
    )
    assert max_drawdown(pd.Series([-0.1])) == pytest.approx(-0.1)


def test_downside_deviation_and_benchmark_coverage():
    r = pd.Series([0.01, -0.02, 0.03])
    assert sortino_ratio(r, 0) == pytest.approx(
        r.mean() / np.sqrt((0.02**2) / 3) * np.sqrt(252)
    )
    with pytest.raises(ValueError, match="Benchmark"):
        compare_to_benchmark(r, r.iloc[1:])
    with pytest.raises(ValueError, match="non-finite"):
        compute_metrics(pd.Series([np.nan]))


def test_zero_trade_rerun_clears_previous_log(tmp_path):
    from src.research.zoo_runner import ZooRunner

    runner = ZooRunner(strategies=[], output_dir=tmp_path)
    r = simulate([100, 100, 110], [2, 1, 1])
    runner._persist("X", "example", r, {})
    flat = simulate([100, 100, 110], [1, 1, 1])
    runner._persist("X", "example", flat, {})
    assert pd.read_parquet(tmp_path / "X/example/trade_log.parquet").empty
