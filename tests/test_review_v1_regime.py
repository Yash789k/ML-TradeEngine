"""The rolling HMM: recovers a known two-state process, is causal, and fails closed on bad input."""

import numpy as np
import pandas as pd
import pytest

from review_v1.regime import RegimeConfig, fit_window, forward_filter, observations, refit_positions, rolling_regime, truncation_check

CONFIG = RegimeConfig(window=600, min_fit=300, refit_every=60, anchor="2015-01-01", seeds=(42, 43), n_iter=100)


def synthetic(n=1500, seed=11, start="2012-01-02"):
    """Prices from a known two-state Markov chain: quiet (0.6% a day) and turbulent (2.4% a day)."""
    rng = np.random.default_rng(seed)
    state = np.zeros(n, int)
    for t in range(1, n):
        stay = 0.99 if state[t - 1] == 0 else 0.98
        state[t] = state[t - 1] if rng.random() < stay else 1 - state[t - 1]
    returns = rng.normal(np.where(state == 0, 0.0006, -0.0004), np.where(state == 0, 0.006, 0.024))
    index = pd.bdate_range(start, periods=n)
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=index), pd.Series(state, index=index)


@pytest.fixture(scope="module")
def fitted():
    close, truth = synthetic()
    posterior, fits = rolling_regime(close, CONFIG)
    return close, truth, posterior, fits


def test_recovers_a_known_two_state_process(fitted):
    _, truth, posterior, fits = fitted
    actual = truth.reindex(posterior.index)
    # Realised volatility is a 21-session average, so the filter needs a few weeks to see a switch.
    # Score only sessions at least 30 sessions into a spell.
    spell = actual.groupby((actual != actual.shift()).cumsum()).cumcount()
    settled = spell >= 30
    predicted = (posterior.state == "turbulent").astype(int)
    assert settled.sum() > 400
    assert (predicted[settled] == actual[settled]).mean() > 0.9
    # Both states must actually occur, in the fitted order: state 0 is the quieter one.
    assert 0.1 < predicted.mean() < 0.9
    for fit in fits:
        calm, turbulent = fit["states"]
        assert calm["name"] == "calm" and turbulent["name"] == "turbulent"
        assert calm["typical_realised_volatility"] < turbulent["typical_realised_volatility"]
        assert calm["annualised_volatility"] < 0.5 * turbulent["annualised_volatility"]
    # The dashboard reports the median across refits, so that is what is checked against the truth.
    # The calm estimate runs high (observed 10.0% to 13.6% against a true 9.5%): the first sessions
    # after a switch into turbulence are still labelled calm until the 21-session volatility catches up.
    calm_volatility = np.median([fit["states"][0]["annualised_volatility"] for fit in fits])
    turbulent_volatility = np.median([fit["states"][1]["annualised_volatility"] for fit in fits])
    true_calm, true_turbulent = 0.006 * np.sqrt(252), 0.024 * np.sqrt(252)
    assert 0.95 * true_calm < calm_volatility < 1.2 * true_calm
    assert turbulent_volatility == pytest.approx(true_turbulent, rel=0.1)


def test_probabilities_are_valid_and_dated_by_the_fit_that_produced_them(fitted):
    _, _, posterior, fits = fitted
    p = posterior[["p_calm", "p_turbulent"]].to_numpy()
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    assert np.allclose(p.sum(axis=1), 1, atol=1e-9)
    assert posterior.index.is_monotonic_increasing and posterior.index.is_unique
    # A row may only use parameters fitted on or before its own date.
    assert (posterior.fit_date <= posterior.index).all()
    assert all(fit["converged"] for fit in fits)


def test_removing_later_data_does_not_change_earlier_probabilities(fitted):
    close, _, posterior, _ = fitted
    cuts = [posterior.index[100], posterior.index[len(posterior) // 2], posterior.index[-40]]
    rows = truncation_check(close, posterior, cuts, CONFIG)
    assert [r["rows_compared"] for r in rows] == [r["rows_expected"] for r in rows]
    assert all(r["rows_compared"] > 0 for r in rows)
    assert max(r["max_abs_difference"] for r in rows) < 1e-9


def test_a_look_ahead_estimator_would_fail_the_truncation_check(fitted):
    """The check has teeth: probabilities that depend on later data are caught."""
    close, _, posterior, _ = fitted
    leaked = posterior.copy()
    leaked[["p_calm", "p_turbulent"]] = leaked[["p_calm", "p_turbulent"]].shift(-5).ffill()
    rows = truncation_check(close, leaked, [posterior.index[len(posterior) // 2]], CONFIG)
    assert rows[0]["max_abs_difference"] > 1e-3


def test_refit_dates_do_not_depend_on_where_the_history_starts(fitted):
    close, _, posterior, _ = fitted
    later, _ = rolling_regime(close.iloc[120:], CONFIG)
    common = later.index.intersection(posterior.index)
    assert len(common) > 500
    assert (later.loc[common, "fit_date"] == posterior.loc[common, "fit_date"]).all()
    index = observations(close).index
    assert refit_positions(index, CONFIG)[0] == int(index.searchsorted(pd.Timestamp(CONFIG.anchor)))


def test_forward_filter_matches_direct_bayes_updates():
    emission = np.log(np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]]))
    start, transition = np.array([0.6, 0.4]), np.array([[0.95, 0.05], [0.10, 0.90]])
    belief = start * np.exp(emission[0])
    belief /= belief.sum()
    expected = [belief]
    for row in emission[1:]:
        belief = (belief @ transition) * np.exp(row)
        belief /= belief.sum()
        expected.append(belief)
    assert np.allclose(forward_filter(emission, start, transition), expected, atol=1e-12)


def test_forward_filter_survives_underflowing_likelihoods():
    emission = np.array([[-5.0, -6.0], [-2000.0, -1990.0], [-1.0, -3.0]])
    out = forward_filter(emission, np.array([0.5, 0.5]), np.array([[0.9, 0.1], [0.1, 0.9]]))
    assert np.isfinite(out).all()
    assert np.allclose(out.sum(axis=1), 1)
    assert out[1, 1] > 0.99


@pytest.mark.parametrize(
    "values",
    [
        [100.0, 101.0, -1.0, 102.0],
        [100.0, 0.0, 101.0, 102.0],
        [100.0, float("nan"), 101.0, 102.0],
        [100.0, float("inf"), 101.0, 102.0],
    ],
)
def test_bad_prices_are_rejected(values):
    with pytest.raises(ValueError):
        observations(pd.Series(values, index=pd.bdate_range("2024-01-02", periods=len(values))))


def test_unordered_or_duplicate_sessions_are_rejected():
    index = pd.bdate_range("2024-01-02", periods=4)
    with pytest.raises(ValueError):
        observations(pd.Series([1.0, 2.0, 3.0, 4.0], index=index[::-1]))
    with pytest.raises(ValueError):
        observations(pd.Series([1.0, 2.0, 3.0, 4.0], index=[index[0], index[0], index[1], index[2]]))


def test_too_little_history_emits_no_regime():
    close, _ = synthetic(n=250)
    with pytest.raises(ValueError):
        rolling_regime(close, CONFIG)
    with pytest.raises(ValueError):
        fit_window(observations(close), CONFIG)


def test_a_constant_price_series_cannot_be_fitted():
    index = pd.bdate_range("2012-01-02", periods=900)
    flat = pd.DataFrame({"log_return": np.zeros(900), "log_vol": np.full(900, -5.0)}, index=index)
    with pytest.raises(ValueError):
        fit_window(flat, CONFIG)
