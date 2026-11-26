"""
Phase 02 — Regime Detection via Hidden Markov Model

Uses a Gaussian HMM with n_states (default 3) fitted on two observable series:
  - daily log-return
  - 21-day realized volatility

The model labels each trading day as one of N hidden states.  States are then
mapped to human-readable labels by ranking on mean return:
  lowest-mean state  → 0 (bear)
  middle state(s)    → 1 (ranging)  [for n_states > 2]
  highest-mean state → 2 (bull)

Causal convention
-----------------
Fit once on the first 252 eligible observations. Warm-up rows remain unknown.
Thereafter use only forward-filter probabilities with frozen parameters. The
initial fitting window is deliberately never relabeled as historical features.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM  # type: ignore

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ---------------------------------------------------------------------------
# Core HMM fitter
# ---------------------------------------------------------------------------

def fit_hmm(
    df: pd.DataFrame,
    n_states: int = 3,
    random_state: int = 42,
    covariance_type: str = "full",
) -> GaussianHMM:
    """
    Fit a Gaussian HMM on (log_return, realized_vol_21).
    Requires these columns to already exist in `df`.

    Parameters
    ----------
    covariance_type : {"full", "diag", "tied", "spherical"}
        Passed directly to GaussianHMM.  Use "diag" as a fallback when
        "full" is near-singular (e.g. very short or IID synthetic data).

    Returns the fitted model — caller decides when to decode.
    """
    feat_cols = []
    if "log_return" in df.columns:
        feat_cols.append("log_return")
    if "realized_vol_21" in df.columns:
        feat_cols.append("realized_vol_21")

    if not feat_cols:
        raise ValueError(
            "HMM requires 'log_return' and/or 'realized_vol_21' columns. "
            "Call add_all_statistical() before fit_hmm()."
        )

    obs = df[feat_cols].dropna().values

    scale = np.maximum(np.std(obs, axis=0), 1e-6)
    obs = obs / scale
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=200,
        startprob_prior=1.1,
        transmat_prior=1.1,
        tol=1e-4,
        random_state=random_state,
    )
    model.fit(obs)
    model.feature_scale_ = scale
    if not np.isfinite(model.means_).all() or not np.isfinite(model.transmat_).all():
        raise ValueError("HMM calibration failed; no regime feature may be emitted")
    return model


def decode_hmm(
    model: GaussianHMM,
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Forward-filter a sequence using a previously fitted, frozen model.
    Returns an integer array of raw state indices (length = len(df)).
    Rows with NaN features receive state -1.
    """
    feat_cols = [c for c in ["log_return", "realized_vol_21"] if c in df.columns]
    obs_full  = df[feat_cols]
    valid     = obs_full.dropna()
    states    = np.full(len(df), -1, dtype=int)

    if len(valid) > 0:
        from scipy.special import logsumexp
        emissions = model._compute_log_likelihood(valid.to_numpy() / getattr(model, "feature_scale_", 1.0))
        with np.errstate(divide="ignore"):
            transition = np.log(model.transmat_)
            posterior = np.log(model.startprob_)
        raw = []
        for i, emission in enumerate(emissions):
            prior = posterior if i == 0 else logsumexp(posterior[:, None] + transition, axis=0)
            posterior = prior + emission
            posterior -= logsumexp(posterior)
            raw.append(int(np.argmax(posterior)))
        states[np.flatnonzero(obs_full.notna().all(axis=1).to_numpy())] = raw

    return states


def _map_states_to_regime(
    model: GaussianHMM,
    raw_states: np.ndarray,
    n_states: int,
) -> np.ndarray:
    """
    Remap raw HMM states so regime 0 = bear, max = bull.
    Ranking is by the mean of the first feature (log_return).
    """
    means = model.means_[:, 0]
    order = np.argsort(means)               # ascending: bear → bull
    remap = np.empty(n_states, dtype=int)
    for new_label, old_state in enumerate(order):
        remap[old_state] = new_label

    mapped = np.where(raw_states >= 0, remap[raw_states], -1)
    return mapped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_hmm_regime(
    df: pd.DataFrame,
    n_states: int = 3,
    random_state: int = 42,
    col_name: str = "hmm_regime",
    covariance_type: str = "full",
    train_size: int = 252,
) -> tuple[pd.DataFrame, GaussianHMM]:
    """
    Fit a Gaussian HMM and add a regime column to `df`.

    Parameters
    ----------
    covariance_type : {"full", "diag", "tied", "spherical"}
        Passed to the underlying GaussianHMM.  "full" is the default and
        generally best on real market data; "diag" is a useful fallback when
        the observation matrix is near-singular.

    Returns
    -------
    df      : DataFrame with `col_name` column added (0=bear, 1=ranging, 2=bull)
    model   : fitted GaussianHMM (persist for Phase 06 live inference)

    Regime mapping
    --------------
    0 → bear    (lowest mean return state)
    1 → ranging (middle state)
    2 → bull    (highest mean return state)
    """
    feat_cols = [c for c in ["log_return", "realized_vol_21"] if c in df.columns]
    valid = df[feat_cols].dropna()
    df[col_name] = -1
    if len(valid) <= train_size:
        return df, None
    calibration = valid.iloc[:train_size]
    model = fit_hmm(calibration, n_states=n_states, random_state=random_state,
                    covariance_type=covariance_type)
    cutoff = df.index.get_loc(calibration.index[-1]) + 1
    # Filtering the training prefix initializes state probabilities; it is not
    # published as causal history. Only observations after the cutoff are emitted.
    raw_states = decode_hmm(model, df)
    regime = _map_states_to_regime(model, raw_states, n_states)
    regime[:cutoff] = -1
    df[col_name] = regime
    return df, model
