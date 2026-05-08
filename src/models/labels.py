"""
Phase 03 — Label Generation

Produces a 3-class directional label from forward price returns.

Classes
-------
  2 = UP   : forward_return > +threshold
  0 = DOWN : forward_return < -threshold
  1 = FLAT : |forward_return| <= threshold

Label is intentionally forward-looking — it must NEVER appear as a training
feature.  The trailing `horizon` rows receive label -1 (no future available)
and are dropped before any model sees the data.

Usage
-----
    from src.models.labels import make_labels

    labels, fwd_ret = make_labels(df, horizon=5, threshold=0.005)
    # Drop un-labelable tail
    mask = labels != -1
    X, y = feat_df[mask], labels[mask]
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Stable integer encoding shared across all Phase 03 modules
CLASS_DOWN = 0
CLASS_FLAT = 1
CLASS_UP   = 2
CLASS_MASK = -1   # sentinel: no valid label (trailing rows)

CLASS_NAMES = {CLASS_DOWN: "DOWN", CLASS_FLAT: "FLAT", CLASS_UP: "UP"}


def make_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.005,
    close_col: str = "Close",
) -> tuple[pd.Series, pd.Series]:
    """
    Generate 3-class directional labels from forward close-to-close returns.

    Parameters
    ----------
    df        : OHLCV or feature DataFrame with a close price column
    horizon   : number of bars to look ahead for the return calculation
    threshold : symmetric percentage threshold; returns outside ±threshold
                are labelled UP/DOWN; inside is FLAT
    close_col : name of the close price column (default 'Close')

    Returns
    -------
    labels   : integer Series {0=DOWN, 1=FLAT, 2=UP, -1=masked}
               index matches df.index
    fwd_ret  : raw forward return Series (useful for analysis)
    """
    if close_col not in df.columns:
        raise ValueError(f"Column '{close_col}' not found in DataFrame.")

    close   = df[close_col]
    fwd_ret = close.shift(-horizon) / close - 1.0

    labels = pd.Series(CLASS_FLAT, index=df.index, dtype=np.int8, name="label")
    labels[fwd_ret >  threshold] = CLASS_UP
    labels[fwd_ret < -threshold] = CLASS_DOWN
    labels[fwd_ret.isna()]       = CLASS_MASK

    return labels, fwd_ret


def class_distribution(labels: pd.Series) -> pd.Series:
    """Return percentage distribution across valid classes (excludes -1)."""
    valid = labels[labels != CLASS_MASK]
    counts = valid.value_counts().sort_index()
    return (counts / len(valid) * 100).rename(CLASS_NAMES)


def compute_class_weights(labels: pd.Series) -> dict[int, float]:
    """
    Inverse-frequency class weights for imbalanced datasets.
    Returns a dict {class_int: weight_float} suitable for XGB sample_weight
    and PyTorch CrossEntropyLoss weight tensor.
    """
    valid = labels[labels != CLASS_MASK]
    counts = valid.value_counts()
    total  = len(valid)
    weights = {int(c): total / (len(counts) * n) for c, n in counts.items()}
    return weights
