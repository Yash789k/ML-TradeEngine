"""
Phase 02 — Technical Features
Computes momentum, trend, volatility, and volume indicators via pandas-ta.

All indicators are computed on strictly backward-looking windows — no lookahead.

Feature groups
--------------
Momentum / trend  : RSI(14), MACD(12,26,9), ADX(14), EMA ratios
Volatility        : Bollinger Bands(20,2), ATR(14), BB width, daily range
Volume            : OBV, volume ratio (vol / 20-day avg)
Price structure   : log-return, gap (open vs prior close)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Momentum & trend
# ---------------------------------------------------------------------------

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    RSI(14), MACD(12,26,9), ADX(14).
    Appended in-place; original columns untouched.
    """
    _require_cols(df, ["Close", "High", "Low"])

    rsi = ta.rsi(df["Close"], length=14)
    if rsi is not None:
        df["rsi_14"] = rsi

    macd_df = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    if macd_df is not None:
        df["macd_line"]   = macd_df.iloc[:, 0]
        df["macd_hist"]   = macd_df.iloc[:, 1]
        df["macd_signal"] = macd_df.iloc[:, 2]

    adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    if adx_df is not None:
        df["adx_14"] = adx_df.iloc[:, 0]

    return df


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bollinger Bands(20,2), ATR(14), BB width, daily high-low range ratio.
    """
    _require_cols(df, ["Close", "High", "Low"])

    bb = ta.bbands(df["Close"], length=20, std=2)
    if bb is not None:
        df["bb_lower"]  = bb.iloc[:, 0]
        df["bb_mid"]    = bb.iloc[:, 1]
        df["bb_upper"]  = bb.iloc[:, 2]
        df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)
        df["bb_pct"]    = (df["Close"] - df["bb_lower"]) / (
            (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        )

    atr = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    if atr is not None:
        df["atr_14"] = atr

    df["hl_range_pct"] = (df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)

    return df


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    OBV, volume ratio (current vol / 20-day rolling average).
    """
    _require_cols(df, ["Close", "Volume"])

    obv = ta.obv(df["Close"], df["Volume"])
    if obv is not None:
        df["obv"] = obv

    vol_ma = df["Volume"].rolling(20, min_periods=1).mean()
    df["volume_ratio_20"] = df["Volume"] / vol_ma.replace(0, np.nan)

    return df


# ---------------------------------------------------------------------------
# Price structure
# ---------------------------------------------------------------------------

def add_price_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    EMA ratios (20/50/200), log-return, gap (open vs prior close).
    """
    _require_cols(df, ["Open", "Close"])

    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    df["log_return"] = log_ret

    df["return_5d"]  = df["Close"].pct_change(5)
    df["return_21d"] = df["Close"].pct_change(21)

    for span in (20, 50, 200):
        ema = df["Close"].ewm(span=span, adjust=False).mean()
        df[f"close_ema{span}_ratio"] = df["Close"] / ema.replace(0, np.nan)

    df["gap_return"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1).replace(0, np.nan)

    return df


# ---------------------------------------------------------------------------
# Convenience: run all technical feature groups
# ---------------------------------------------------------------------------

def add_all_technical(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all technical feature groups and return the enriched DataFrame."""
    df = df.copy()
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_price_structure_features(df)
    return df
