"""
Phase 02 — Statistical Features

Rolling z-score        : standardised price deviation from rolling mean
Hurst exponent         : R/S analysis in a rolling window — >0.5 trending,
                         <0.5 mean-reverting, ~0.5 random walk
Realized volatility    : annualised std-dev of log-returns over rolling windows
Skewness / kurtosis    : rolling third/fourth moment of returns

All windows are strictly backward-looking (no lookahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis  # type: ignore


# ---------------------------------------------------------------------------
# Rolling z-score
# ---------------------------------------------------------------------------

def add_zscore_features(
    df: pd.DataFrame,
    col: str = "Close",
    windows: tuple[int, ...] = (20, 60),
) -> pd.DataFrame:
    """Rolling z-score: (x - mean) / std over each window."""
    for w in windows:
        mu  = df[col].rolling(w, min_periods=w // 2).mean()
        sig = df[col].rolling(w, min_periods=w // 2).std()
        df[f"zscore_{w}"] = (df[col] - mu) / sig.replace(0, np.nan)
    return df


# ---------------------------------------------------------------------------
# Realized volatility
# ---------------------------------------------------------------------------

def add_realized_volatility(
    df: pd.DataFrame,
    log_return_col: str = "log_return",
    windows: tuple[int, ...] = (5, 21),
    ann_factor: float = 252.0,
) -> pd.DataFrame:
    """
    Annualised realized volatility = std(log_return, window) * sqrt(ann_factor).
    Requires `log_return` column — call add_price_structure_features first.
    """
    if log_return_col not in df.columns:
        df[log_return_col] = np.log(df["Close"] / df["Close"].shift(1))

    for w in windows:
        rv = df[log_return_col].rolling(w, min_periods=w // 2).std() * np.sqrt(ann_factor)
        df[f"realized_vol_{w}"] = rv
    return df


# ---------------------------------------------------------------------------
# Hurst exponent  (R/S method)
# ---------------------------------------------------------------------------

def _hurst_rs(series: np.ndarray) -> float:
    """
    Estimate the Hurst exponent via Rescaled Range (R/S) analysis.

    Partition the series into m sub-series for each lag scale n,
    compute R/S for each, fit log(R/S) ~ H * log(n) by OLS.

    Returns NaN if the series is too short or degenerate.
    """
    n = len(series)
    if n < 20:
        return np.nan

    lags   = []
    rs_vals = []

    for lag in range(10, n // 2):
        # Split series into non-overlapping chunks of length `lag`
        chunks = [series[i : i + lag] for i in range(0, n - lag + 1, lag)]
        if len(chunks) < 2:
            continue

        rs_list = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean_c = np.mean(chunk)
            dev    = np.cumsum(chunk - mean_c)
            r      = dev.max() - dev.min()
            s      = np.std(chunk, ddof=1)
            if s > 0:
                rs_list.append(r / s)

        if rs_list:
            lags.append(np.log(lag))
            rs_vals.append(np.log(np.mean(rs_list)))

    if len(lags) < 2:
        return np.nan

    coeffs = np.polyfit(lags, rs_vals, 1)
    # Clamp to (0, 1]: R/S estimator can overshoot on strongly trending series
    return float(np.clip(coeffs[0], 0.0, 1.0))


def add_hurst_exponent(
    df: pd.DataFrame,
    col: str = "Close",
    window: int = 100,
) -> pd.DataFrame:
    """
    Rolling Hurst exponent using a backward-looking window.
    Values >0.5 indicate trend persistence; <0.5 mean reversion.
    """
    prices = df[col].values
    hurst  = np.full(len(prices), np.nan)

    for i in range(window - 1, len(prices)):
        hurst[i] = _hurst_rs(prices[i - window + 1 : i + 1])

    df[f"hurst_{window}"] = hurst
    return df


# ---------------------------------------------------------------------------
# Rolling skewness & kurtosis of returns
# ---------------------------------------------------------------------------

def add_return_moments(
    df: pd.DataFrame,
    log_return_col: str = "log_return",
    window: int = 60,
) -> pd.DataFrame:
    """Rolling skewness and excess kurtosis of log-returns."""
    if log_return_col not in df.columns:
        df[log_return_col] = np.log(df["Close"] / df["Close"].shift(1))

    df[f"skew_{window}"]     = df[log_return_col].rolling(window, min_periods=window // 2).skew()
    df[f"kurtosis_{window}"] = df[log_return_col].rolling(window, min_periods=window // 2).kurt()
    return df


# ---------------------------------------------------------------------------
# Convenience: run all statistical features
# ---------------------------------------------------------------------------

def add_all_statistical(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all statistical feature groups and return the enriched DataFrame."""
    df = add_zscore_features(df)
    df = add_realized_volatility(df)
    df = add_hurst_exponent(df)
    df = add_return_moments(df)
    return df
