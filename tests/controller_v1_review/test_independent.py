"""Independent post-run qualification; preserve the registered screen and results.

Synthetic refits are behavior tests, not additional research/selection trials.
Actual-data checks read the completed screen without retraining any estimator.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_controller_v1 import screen


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data/research-controller-v1/20261030-screen-v1"
CUTOFF = pd.Timestamp("2025-07-01", tz="UTC")
PREDICTION_COLUMNS = ["date", "symbol", "model", "p0", "p1", "p2"]


@pytest.fixture(scope="module")
def synthetic_panel():
    dates = pd.bdate_range("2024-01-02", "2025-07-18", tz="UTC")
    rng = np.random.default_rng(20261030)
    rows = []
    for i, date in enumerate(dates[:-6]):
        for j, symbol in enumerate(["SYNTHETIC_A", "SYNTHETIC_B"]):
            row = dict(date=date, endpoint=dates[i + 6], symbol=symbol, label=(i + j) % 3)
            for expert in screen.EXPERTS:
                probabilities = rng.dirichlet([2.5, 3.0, 2.0])
                row.update({f"{expert}_{c}": p for c, p in enumerate(probabilities)})
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_forecasts(synthetic_panel):
    result = screen.forecast(synthetic_panel, use_stack=True)
    assert len(result[2]) == 2
    assert pd.Timestamp(result[2][1]["decision"]) == CUTOFF
    return result


def assert_earlier_decisions_equal(original, changed):
    old_forecasts, old_weights, old_fits = original
    new_forecasts, new_weights, new_fits = changed
    pd.testing.assert_frame_equal(
        old_forecasts.loc[old_forecasts.date <= CUTOFF, PREDICTION_COLUMNS],
        new_forecasts.loc[new_forecasts.date <= CUTOFF, PREDICTION_COLUMNS],
    )
    pd.testing.assert_frame_equal(
        old_weights.loc[pd.to_datetime(old_weights.date) <= CUTOFF],
        new_weights.loc[pd.to_datetime(new_weights.date) <= CUTOFF],
    )
    # Includes fit boundaries, counts, training hash, scaler mean and learned coefficients.
    assert old_fits == new_fits


def test_full_stack_future_label_mutation_including_july_refit(
    synthetic_panel, synthetic_forecasts
):
    changed = synthetic_panel.copy(deep=True)
    unavailable = changed.endpoint >= CUTOFF
    assert (changed.endpoint == CUTOFF).any()
    changed.loc[unavailable, "label"] = (changed.loc[unavailable, "label"] + 1) % 3
    assert_earlier_decisions_equal(
        synthetic_forecasts, screen.forecast(changed, use_stack=True)
    )


def test_full_stack_future_probability_mutation_including_july_refit(
    synthetic_panel, synthetic_forecasts
):
    changed = synthetic_panel.copy(deep=True)
    future = changed.date > CUTOFF
    for expert in screen.EXPERTS:
        columns = [f"{expert}_{c}" for c in range(3)]
        changed.loc[future, columns] = np.roll(
            changed.loc[future, columns].to_numpy(), 1, axis=1
        )
    assert_earlier_decisions_equal(
        synthetic_forecasts, screen.forecast(changed, use_stack=True)
    )


@pytest.fixture(scope="module")
def actual_run():
    protocol = json.loads((RUN / "protocol.json").read_text())
    panel = screen.load_panel(protocol)
    fits = json.loads((RUN / "stack_fits.json").read_text())
    return protocol, panel, fits


def test_actual_stack_fit_excludes_same_day_endpoints(actual_run):
    _, panel, fits = actual_run
    assert len(fits) == 2
    for index, fit in enumerate(fits):
        decision = pd.Timestamp(fit["decision"])
        training = panel.loc[(panel.date < decision) & (panel.endpoint < decision)]
        if index == 0:
            training = training.loc[training.date.dt.year == 2024]
        else:
            training = training.loc[training.date < CUTOFF]
        same_day_endpoint = panel.loc[panel.endpoint == decision]
        assert len(same_day_endpoint) == 22
        assert set(training.index).isdisjoint(same_day_endpoint.index)
        assert training.endpoint.max() < decision
        assert len(training) == fit["n_rows"]
        assert training.date.nunique() == fit["n_dates"]
        assert str(training.endpoint.max()) == fit["latest_endpoint"]
        training_hash = hashlib.sha256(
            pd.util.hash_pandas_object(training, index=False).values.tobytes()
        ).hexdigest()
        assert training_hash == fit["training_hash"]


def test_actual_hedge_feedback_counts_unique_strictly_mature_dates(actual_run):
    _, panel, _ = actual_run
    dates = panel[["date", "endpoint"]].drop_duplicates()
    weights = pd.read_csv(RUN / "hedge_weights.csv")
    for row in weights.itertuples(index=False):
        decision = pd.Timestamp(row.date)
        expected = dates.loc[dates.endpoint < decision, "date"].nunique()
        assert row.n_feedback_dates == expected
    assert (weights["n_feedback_dates"].diff().dropna() >= 0).all()


def test_actual_baselines_match_distinct_registered_expert_sets(actual_run):
    _, panel, _ = actual_run
    evaluation = panel.loc[panel.date.dt.year == 2025].set_index(["date", "symbol"])
    forecasts = pd.read_parquet(RUN / "forecasts.parquet")
    p = screen.expert_probabilities(evaluation)
    expected = {
        "prior": p[:, 0],
        "fixed_mixture": (p[:, 1] + p[:, 2]) / 2,
        "uniform3": p.mean(axis=1),
    }
    for name, probabilities in expected.items():
        saved = forecasts.loc[forecasts.model == name].set_index(["date", "symbol"])
        assert saved.index.is_unique
        saved = saved.loc[evaluation.index]
        np.testing.assert_allclose(
            saved[["p0", "p1", "p2"]].to_numpy(), probabilities, rtol=0, atol=1e-15
        )


def test_completed_manifest_and_registered_source_inputs_unchanged(actual_run):
    protocol, _, _ = actual_run
    manifest = json.loads((RUN / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert protocol["sources"] == screen.source_hashes()
    for files in [manifest["files"], protocol["sources"], protocol["inputs"]]:
        for relative, expected in files.items():
            path = ROOT / relative
            assert path.resolve().is_relative_to(ROOT)
            assert screen.digest(path) == expected, relative
