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

Design decision — lookahead note
---------------------------------
hmmlearn's Viterbi decoder (predict) operates on the FULL training series at
once.  In a live setting you would use online filtering (forward pass only).
For the backtesting feature store we use the full-series Viterbi, which gives
slightly smoother labels and is the standard practise in academic quant research.
The ML model is trained on these labels in Phase 03, so any Viterbi smoothing
is consistent across train and test sets — as long as the HMM is fit only on
the training split and the test labels are decoded using that frozen model.
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
) -> GaussianHMM:
    """
    Fit a Gaussian HMM on (log_return, realized_vol_21).
    Requires these columns to already exist in `df`.

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

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=500,
        tol=1e-4,
        random_state=random_state,
    )
    model.fit(obs)
    return model


def decode_hmm(
    model: GaussianHMM,
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Run Viterbi decoding on the full DataFrame.
    Returns an integer array of raw state indices (length = len(df)).
    Rows with NaN features receive state -1.
    """
    feat_cols = [c for c in ["log_return", "realized_vol_21"] if c in df.columns]
    obs_full  = df[feat_cols]
    valid     = obs_full.dropna()
    states    = np.full(len(df), -1, dtype=int)

    if len(valid) > 0:
        raw = model.predict(valid.values)
        states[valid.index.get_indexer(valid.index)] = raw  # type: ignore[call-arg]

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
) -> tuple[pd.DataFrame, GaussianHMM]:
    """
    Fit a Gaussian HMM and add a regime column to `df`.

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
    model       = fit_hmm(df, n_states=n_states, random_state=random_state)
    raw_states  = decode_hmm(model, df)
    regime      = _map_states_to_regime(model, raw_states, n_states)
    df[col_name] = regime
    return df, model
