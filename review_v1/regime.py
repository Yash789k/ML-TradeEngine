"""Causal rolling Gaussian HMM of market regimes from one price series (SPY).

What it does
------------
Observation on each session: the daily log return and the log of the 21-session
realised volatility. A Gaussian hidden Markov model is refit every 21 observed
sessions on a trailing window. Between refits the parameters are frozen and the
regime probability is the *filtered* posterior P(state at t | observations up to t),
never the smoothed one, so each value uses information available at that close.

Differences from ``src/features/regime.py`` (the summer prototype):
that module fits once on the first 252 rows and emits a hard state label. This one
refits on a rolling window and emits probabilities, which is what the FYP proposal
specifies for the gate inputs.

Assumptions (recorded so they can be challenged)
------------------------------------------------
* Two states by default. The proposal says the state count (2, 3 or 4) should be
  chosen on validation data; that selection has not been run, so two is a fixed
  prior choice, not a tuned one.
* States are ordered by the fitted mean of log realised volatility: state 0 is the
  calmer state, the last state the most turbulent. The proposal orders by mean
  return; volatility ordering is used because it is far more stable across refits.
* The refit schedule is anchored to a fixed calendar date and counted in observed
  sessions, so two price histories that agree from the anchor onward produce the
  same refit dates regardless of where each history starts.
* Standardisation uses the mean and standard deviation of the training window only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class RegimeConfig:
    version: int = 1
    n_states: int = 2
    window: int = 1008
    min_fit: int = 504
    refit_every: int = 21
    vol_window: int = 21
    anchor: str = "2019-01-02"
    seeds: tuple = (42, 43, 44)
    n_iter: int = 200


def observations(close: pd.Series, vol_window: int = 21) -> pd.DataFrame:
    """Log return and log realised volatility; rows with undefined values are dropped."""
    close = pd.Series(close).astype(float)
    if not close.index.is_unique or not close.index.is_monotonic_increasing:
        raise ValueError("Ordered unique sessions required")
    if (close <= 0).any() or not np.isfinite(close.to_numpy()).all():
        raise ValueError("Finite positive prices required")
    log_return = np.log(close).diff()
    vol = log_return.rolling(vol_window).std()
    with np.errstate(divide="ignore"):
        log_vol = np.log(vol)
    frame = pd.DataFrame({"log_return": log_return, "log_vol": log_vol})
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def forward_filter(log_emission: np.ndarray, startprob: np.ndarray, transmat: np.ndarray) -> np.ndarray:
    """Filtered state probabilities P(s_t | o_1..o_t) for frozen parameters."""
    log_emission = np.asarray(log_emission, float)
    transmat = np.asarray(transmat, float)
    belief = np.asarray(startprob, float)
    out = np.empty_like(log_emission)
    for t, emission in enumerate(log_emission):
        prior = belief if t == 0 else belief @ transmat
        weight = prior * np.exp(emission - emission.max())
        total = weight.sum()
        if not (np.isfinite(total) and total > 0):
            # Every state with prior mass has an underflowing likelihood: redo
            # this one step in log space rather than emit an undefined belief.
            with np.errstate(divide="ignore"):
                log_weight = np.log(prior) + emission
            weight, total = np.exp(log_weight - logsumexp(log_weight)), 1.0
        belief = weight / total
        out[t] = belief
    return out


def _emission_logprob(z: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [multivariate_normal(mean=means[k], cov=covars[k], allow_singular=True).logpdf(z) for k in range(len(means))]
    ).reshape(len(z), len(means))


def fit_window(train: pd.DataFrame, config: RegimeConfig = RegimeConfig()) -> dict:
    """Fit on one training window; returns frozen, volatility-ordered parameters."""
    from hmmlearn.hmm import GaussianHMM

    x = train[["log_return", "log_vol"]].to_numpy(float)
    if len(x) < config.min_fit or not np.isfinite(x).all():
        raise ValueError("Insufficient or non-finite regime training window")
    mean, std = x.mean(axis=0), x.std(axis=0)
    if (std <= 0).any():
        raise ValueError("Degenerate regime training window")
    z = (x - mean) / std
    best = None
    # EM on a flat likelihood logs a harmless "not converging" notice when the
    # last step changes the likelihood by about 1e-7; keep real errors only.
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)
    for seed in config.seeds:
        model = GaussianHMM(
            n_components=config.n_states,
            covariance_type="full",
            n_iter=config.n_iter,
            tol=1e-4,
            random_state=seed,
        )
        # One thread: these are tiny matrices, and a fixed thread count keeps
        # floating-point summation order, and so the fitted values, repeatable.
        with warnings.catch_warnings(), threadpool_limits(limits=1):
            warnings.simplefilter("ignore")
            model.fit(z)
            score = float(model.score(z))
        if not (np.isfinite(model.means_).all() and np.isfinite(model.transmat_).all() and np.isfinite(score)):
            continue
        if best is None or score > best[0]:
            best = (score, seed, model)
    if best is None:
        raise ValueError("HMM calibration failed for every seed; no regime may be emitted")
    score, seed, model = best
    order = np.argsort(model.means_[:, 1], kind="stable")  # ascending log-volatility
    means = model.means_[order]
    covars = np.asarray(model.covars_)[order]
    transmat = model.transmat_[np.ix_(order, order)]
    startprob = model.startprob_[order]
    return dict(
        mean=mean,
        std=std,
        means=means,
        covars=covars,
        transmat=transmat,
        startprob=startprob,
        log_likelihood=score,
        seed=int(seed),
        converged=bool(model.monitor_.converged),
        iterations=int(model.monitor_.iter),
        n=len(x),
    )


def describe_fit(params: dict) -> list:
    """State parameters in market units for display."""
    rows = []
    for k in range(len(params["means"])):
        mean_return = params["means"][k, 0] * params["std"][0] + params["mean"][0]
        std_return = float(np.sqrt(params["covars"][k, 0, 0])) * params["std"][0]
        mean_log_vol = params["means"][k, 1] * params["std"][1] + params["mean"][1]
        stay = float(params["transmat"][k, k])
        rows.append(
            dict(
                state=k,
                mean_daily_return=float(mean_return),
                annualised_return=float(mean_return * 252),
                annualised_volatility=float(std_return * np.sqrt(252)),
                typical_realised_volatility=float(np.exp(mean_log_vol) * np.sqrt(252)),
                stay_probability=stay,
                expected_duration_sessions=float(1 / (1 - stay)) if stay < 1 else float("inf"),
            )
        )
    return rows


def state_names(n_states: int) -> list:
    if n_states == 2:
        return ["calm", "turbulent"]
    if n_states == 3:
        return ["calm", "middle", "turbulent"]
    return [f"state_{k}" for k in range(n_states)]


def refit_positions(index: pd.DatetimeIndex, config: RegimeConfig) -> list:
    anchor = pd.Timestamp(config.anchor)
    anchor = anchor.tz_localize(index.tz) if index.tz is not None and anchor.tzinfo is None else anchor
    start = int(index.searchsorted(anchor))
    return [p for p in range(start, len(index), config.refit_every) if p + 1 >= config.min_fit]


def rolling_regime(close: pd.Series, config: RegimeConfig = RegimeConfig()):
    """Filtered regime probabilities for every session from the first eligible refit.

    Returns ``(posterior, fits)``. ``posterior`` has one probability column per state
    (``p_calm``, ``p_turbulent`` for two states), the most likely ``state`` and the
    ``fit_date`` whose parameters produced the row. ``fits`` documents every refit.
    """
    obs = observations(close, config.vol_window)
    names = state_names(config.n_states)
    positions = refit_positions(obs.index, config)
    pieces, fits = [], []
    for i, position in enumerate(positions):
        lo = max(0, position - config.window + 1)
        params = fit_window(obs.iloc[lo : position + 1], config)
        stop = positions[i + 1] if i + 1 < len(positions) else len(obs)
        block = obs.iloc[lo:stop]
        z = (block[["log_return", "log_vol"]].to_numpy(float) - params["mean"]) / params["std"]
        filtered = forward_filter(
            _emission_logprob(z, params["means"], params["covars"]), params["startprob"], params["transmat"]
        )[position - lo :]
        piece = pd.DataFrame(filtered, index=block.index[position - lo :], columns=[f"p_{n}" for n in names])
        piece["state"] = [names[k] for k in filtered.argmax(axis=1)]
        piece["fit_date"] = obs.index[position]
        pieces.append(piece)
        fits.append(
            dict(
                fit_date=obs.index[position],
                window_start=obs.index[lo],
                n=params["n"],
                seed=params["seed"],
                converged=params["converged"],
                iterations=params["iterations"],
                log_likelihood=params["log_likelihood"],
                states=[dict(name=names[r["state"]], **r) for r in describe_fit(params)],
            )
        )
    if not pieces:
        raise ValueError("No eligible regime refit: history too short for the configured anchor and window")
    posterior = pd.concat(pieces)
    total = posterior[[f"p_{n}" for n in names]].sum(axis=1)
    if not np.allclose(total, 1, atol=1e-9):
        raise ValueError("Regime probabilities do not sum to one")
    posterior.attrs["config"] = asdict(config)
    return posterior, fits


def truncation_check(close: pd.Series, posterior: pd.DataFrame, cut_dates, config: RegimeConfig = RegimeConfig()) -> list:
    """Recompute with every observation after each cut date removed.

    A causal estimator must give the same probabilities on and before the cut.
    The returned maximum absolute differences should be numerically zero: the
    observed values are about 1e-12 (floating-point summation order inside EM),
    and the tests require them to stay below 1e-9.
    """
    probability_columns = [c for c in posterior.columns if c.startswith("p_")]
    rows = []
    for cut in cut_dates:
        cut = pd.Timestamp(cut)
        cut = cut.tz_localize(close.index.tz) if close.index.tz is not None and cut.tzinfo is None else cut
        partial, _ = rolling_regime(close.loc[:cut], config)
        common = partial.index.intersection(posterior.loc[:cut].index)
        difference = (partial.loc[common, probability_columns] - posterior.loc[common, probability_columns]).abs()
        rows.append(
            dict(
                cut=cut,
                rows_compared=len(common),
                rows_expected=len(posterior.loc[:cut]),
                max_abs_difference=float(difference.to_numpy().max()) if len(common) else float("nan"),
            )
        )
    return rows
