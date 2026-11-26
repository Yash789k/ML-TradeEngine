"""Oracles for the review statistics: hand-computed classification metrics and published Sharpe examples."""

import math

import numpy as np
import pytest

from review_v1 import stats


def test_classification_report_matches_hand_computed_values():
    y = [0, 0, 1, 2, 2, 2]
    p = np.array([
        [0.6, 0.3, 0.1],  # DOWN, predicted DOWN
        [0.2, 0.5, 0.3],  # DOWN, predicted FLAT
        [0.1, 0.7, 0.2],  # FLAT, predicted FLAT
        [0.2, 0.2, 0.6],  # UP, predicted UP
        [0.5, 0.1, 0.4],  # UP, predicted DOWN
        [0.1, 0.1, 0.8],  # UP, predicted UP
    ])
    report = stats.classification_report(y, p)
    assert report["n"] == 6
    assert report["accuracy"] == pytest.approx(4 / 6)
    # recall: DOWN 1/2, FLAT 1/1, UP 2/3
    assert report["balanced_accuracy"] == pytest.approx((1 / 2 + 1 + 2 / 3) / 3)
    # precision: DOWN 1/2, FLAT 1/2, UP 2/2 -> F1: 1/2, 2/3, 4/5
    assert report["macro_f1"] == pytest.approx((1 / 2 + 2 / 3 + 4 / 5) / 3)
    assert report["log_loss"] == pytest.approx(-np.mean(np.log([0.6, 0.2, 0.7, 0.6, 0.4, 0.8])))
    brier = np.mean([
        0.4**2 + 0.3**2 + 0.1**2, 0.8**2 + 0.5**2 + 0.3**2, 0.1**2 + 0.3**2 + 0.2**2,
        0.2**2 + 0.2**2 + 0.4**2, 0.5**2 + 0.1**2 + 0.6**2, 0.1**2 + 0.1**2 + 0.2**2,
    ])
    assert report["brier"] == pytest.approx(brier)
    assert [c["support"] for c in report["classes"]] == [2, 1, 3]
    assert [c["predicted"] for c in report["classes"]] == [2, 2, 2]


def test_uniform_forecast_scores_log_three():
    y = [0, 1, 2, 1]
    p = np.full((4, 3), 1 / 3)
    assert stats.classification_report(y, p)["log_loss"] == pytest.approx(math.log(3))
    assert stats.row_log_loss(y, p) == pytest.approx(np.full(4, math.log(3)))


@pytest.mark.parametrize(
    "y, p",
    [
        ([0, 1], [[0.5, 0.5, 0.2], [0.2, 0.3, 0.5]]),      # probabilities do not sum to one
        ([0, 1], [[0.5, 0.6, -0.1], [0.2, 0.3, 0.5]]),     # negative probability
        ([0, 3], [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]]),      # unknown label
        ([0, -1], [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]]),     # unmatured label
        ([0], [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]]),         # length mismatch
        ([], np.empty((0, 3))),                            # nothing to score
        ([0, 1], [[0.5, 0.3, float("nan")], [0.2, 0.3, 0.5]]),
    ],
)
def test_invalid_forecasts_are_rejected_not_scored(y, p):
    with pytest.raises(ValueError):
        stats.classification_report(y, np.asarray(p, float))


def test_sharpe_ratio_is_annualised_mean_over_sample_deviation():
    r = np.array([0.01, -0.005, 0.002, 0.007, -0.003])
    assert stats.sharpe_ratio(r) == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(252))
    assert math.isnan(stats.sharpe_ratio(np.zeros(10)))


def test_expected_maximum_sharpe_matches_the_published_example():
    # Bailey and Lopez de Prado (2014), "The Deflated Sharpe Ratio", worked example:
    # 100 trials whose annualised Sharpe ratios have variance 1/2, daily data (250 per year).
    sr0 = stats.expected_maximum_sharpe(100, 0.5 / 250)
    assert sr0 == pytest.approx(0.1132, abs=5e-5)
    assert sr0 == pytest.approx(0.113172, abs=1e-6)


def test_deflated_sharpe_ratio_matches_the_published_example():
    # Same example: annualised Sharpe 2.5 over 1,250 daily observations, skewness -3, kurtosis 10 -> DSR about 0.9004.
    sr, n, skewness, kurt = 2.5 / math.sqrt(250), 1250, -3.0, 10.0
    sr0 = stats.expected_maximum_sharpe(100, 0.5 / 250)
    from scipy.stats import norm

    expected = norm.cdf((sr - sr0) * math.sqrt(n - 1) / math.sqrt(1 - skewness * sr + (kurt - 1) / 4 * sr**2))
    assert expected == pytest.approx(0.9004, abs=5e-5)

    # The function takes returns, so check it applies the same formula to the sample's own moments.
    rng = np.random.default_rng(7)
    r = rng.standard_t(5, 1500) * 0.01 + 0.0006
    from scipy.stats import kurtosis, skew

    s = r.mean() / r.std(ddof=1)
    by_hand = norm.cdf((s - sr0) * math.sqrt(len(r) - 1) / math.sqrt(1 - skew(r) * s + (kurtosis(r, fisher=False) - 1) / 4 * s**2))
    assert stats.deflated_sharpe_ratio(r, 100, 0.5 / 250) == pytest.approx(by_hand, abs=1e-12)
    # More trials can only lower it; a benchmark of zero recovers the probabilistic Sharpe ratio.
    assert stats.deflated_sharpe_ratio(r, 1000, 0.5 / 250) < stats.deflated_sharpe_ratio(r, 100, 0.5 / 250) < stats.probabilistic_sharpe_ratio(r)


def test_expected_maximum_sharpe_rejects_a_single_trial():
    with pytest.raises(ValueError):
        stats.expected_maximum_sharpe(1, 0.01)
    with pytest.raises(ValueError):
        stats.expected_maximum_sharpe(10, -0.01)


def test_short_samples_are_declared_uninterpretable():
    rng = np.random.default_rng(1)
    short = stats.return_report(rng.normal(0.001, 0.01, 41))
    assert short["sessions"] == 41
    assert not short["interpretable"]
    assert "far too few to interpret a Sharpe ratio" in short["interpretation"]

    noise = stats.return_report(rng.normal(0.0, 0.01, 600))
    assert noise["sharpe_ci_low"] < 0 < noise["sharpe_ci_high"]
    assert not noise["interpretable"]
    assert "includes zero" in noise["interpretation"]

    strong = stats.return_report(rng.normal(0.003, 0.01, 600))
    assert strong["sharpe_ci_low"] > 0
    assert strong["interpretable"]


def test_block_interval_is_reproducible_and_brackets_the_estimate():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0005, 0.01, 500)
    first, second = stats.sharpe_block_interval(r), stats.sharpe_block_interval(r)
    assert first == second
    assert first[0] < stats.sharpe_ratio(r) < first[1]


def test_holm_adjustment_is_monotone_and_never_below_the_raw_value():
    raw = [0.01, 0.04, 0.03]
    adjusted = stats.holm(raw)
    assert all(a >= r - 1e-15 for a, r in zip(adjusted, raw, strict=True))
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[2] == pytest.approx(0.06)
    assert adjusted[1] == pytest.approx(0.06)
