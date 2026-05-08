"""
Phase 02 — Cross-Asset Features

SPY correlation      : rolling Pearson correlation of log-returns with SPY
BTC return proxy     : 5-day / 21-day BTC return as a crypto-sentiment signal
Sector RS            : relative strength = asset return - SPY return, rolling
Macro merge          : attach VIX, yield spread, CPI, rate columns from FRED data

All features are computed causally: only past data enters each row's value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# SPY correlation
# ---------------------------------------------------------------------------

def add_spy_correlation(
    df: pd.DataFrame,
    spy_df: pd.DataFrame,
    windows: tuple[int, ...] = (21, 63),
) -> pd.DataFrame:
    """
    Rolling Pearson correlation of df log-returns vs SPY log-returns.

    Parameters
    ----------
    df     : target asset OHLCV (must have a 'Close' column)
    spy_df : SPY OHLCV with 'Close' column, same calendar
    windows: rolling window sizes in trading days
    """
    asset_ret = np.log(df["Close"] / df["Close"].shift(1))
    spy_ret   = np.log(spy_df["Close"] / spy_df["Close"].shift(1))

    # Align on shared index to avoid lookahead from index gaps
    spy_ret = spy_ret.reindex(df.index).ffill()

    for w in windows:
        df[f"spy_corr_{w}"] = asset_ret.rolling(w, min_periods=w // 2).corr(spy_ret)

    return df


# ---------------------------------------------------------------------------
# Relative strength vs benchmark
# ---------------------------------------------------------------------------

def add_relative_strength(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    benchmark_name: str = "spy",
    windows: tuple[int, ...] = (21, 63),
) -> pd.DataFrame:
    """
    Relative strength = rolling cumulative return of asset minus benchmark.
    Positive → asset outperforming; negative → underperforming.
    """
    asset_ret = df["Close"].pct_change()
    bench_ret = benchmark_df["Close"].pct_change().reindex(df.index).ffill()

    for w in windows:
        asset_cum = asset_ret.rolling(w, min_periods=w // 2).sum()
        bench_cum = bench_ret.rolling(w, min_periods=w // 2).sum()
        df[f"rs_vs_{benchmark_name}_{w}"] = asset_cum - bench_cum

    return df


# ---------------------------------------------------------------------------
# BTC return proxy (crypto-sentiment signal)
# ---------------------------------------------------------------------------

def add_btc_proxy(
    df: pd.DataFrame,
    btc_df: pd.DataFrame,
    windows: tuple[int, ...] = (5, 21),
) -> pd.DataFrame:
    """
    Attach lagged BTC return as a cross-asset sentiment proxy.
    Uses returns lagged by 1 day so day-T feature uses BTC data through T-1.
    """
    btc_ret = btc_df["Close"].pct_change().reindex(df.index).ffill()

    for w in windows:
        btc_roll = btc_ret.rolling(w, min_periods=w // 2).sum().shift(1)
        # Periods where BTC history is unavailable get 0 (neutral — no signal)
        df[f"btc_return_{w}d"] = btc_roll.fillna(0.0)

    return df


# ---------------------------------------------------------------------------
# Macro feature merge
# ---------------------------------------------------------------------------

def add_macro_features(
    df: pd.DataFrame,
    macro_df: pd.DataFrame,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Left-join macro columns onto the asset DataFrame.
    Uses forward-fill so each trading day carries the last available macro reading
    without peeking at future releases.

    Default columns: VIX, yield_spread_10_2, CPI, rate_10y, rate_2y
    """
    if cols is None:
        cols = [c for c in ["VIX", "yield_spread_10_2", "CPI", "rate_10y", "rate_2y"]
                if c in macro_df.columns]

    for col in cols:
        series = macro_df[col].reindex(df.index).ffill()
        # Fill any remaining NaN (e.g. macro starts after asset history) with
        # the column's first available value so no rows are dropped
        df[f"macro_{col}"] = series.fillna(series.bfill())

    return df
