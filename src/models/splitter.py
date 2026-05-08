"""
Phase 03 — Walk-Forward Cross-Validator

Implements PurgedGroupTimeSeriesSplit: a temporal cross-validator that
prevents data leakage when labels are computed from future prices.

Key concepts (Lopez de Prado, "Advances in Financial Machine Learning"):
  Purge   — remove training observations whose forward-label window overlaps
             with the test period (avoids labels "from the future" in train)
  Embargo — drop a buffer period immediately AFTER the test set to prevent
             spillover from any autocorrelation in features/returns

Walk-forward modes
------------------
  expanding : train window grows each fold (more historical data → later folds)
  rolling   : fixed-size train window slides forward (stationarity assumption)

Example (n_splits=5, expanding):

  fold 0:  train [0..399]   embargo  test [410..499]
  fold 1:  train [0..499]   embargo  test [510..599]
  ...

Usage
-----
    from src.models.splitter import WalkForwardSplit

    splitter = WalkForwardSplit(n_splits=5, embargo_days=10)
    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
"""

from __future__ import annotations

import numpy as np
from typing import Generator, Iterator


class WalkForwardSplit:
    """
    Purged walk-forward cross-validator for time series.

    Parameters
    ----------
    n_splits     : number of test folds (default 5)
    test_size    : rows per test fold; if None, uses len(X) // (n_splits+1)
    embargo_days : rows to remove from the END of each train window to prevent
                   label leakage (should be >= label horizon)
    window_type  : 'expanding' (all past data) or 'rolling' (fixed train size)
    min_train    : minimum number of training rows required to produce a fold
    """

    def __init__(
        self,
        n_splits:     int  = 5,
        test_size:    int | None = None,
        embargo_days: int  = 10,
        window_type:  str  = "expanding",
        min_train:    int  = 100,
    ) -> None:
        if window_type not in ("expanding", "rolling"):
            raise ValueError("window_type must be 'expanding' or 'rolling'")
        self.n_splits     = n_splits
        self.test_size    = test_size
        self.embargo_days = embargo_days
        self.window_type  = window_type
        self.min_train    = min_train

    # ------------------------------------------------------------------
    # sklearn-compatible interface
    # ------------------------------------------------------------------

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(
        self,
        X,
        y=None,
        groups=None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Yield (train_indices, test_indices) for each fold.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Only len(X) is used; values are not read.
        """
        n          = len(X)
        test_sz    = self.test_size or max(1, n // (self.n_splits + 1))
        embargo    = self.embargo_days

        for fold in range(self.n_splits):
            # Test window: step forward by test_sz each fold
            test_start = n - (self.n_splits - fold) * test_sz
            test_end   = test_start + test_sz
            test_end   = min(test_end, n)

            if test_start <= 0:
                continue

            # Train window ends `embargo` rows before the test window
            train_end = test_start - embargo
            if train_end < self.min_train:
                continue

            if self.window_type == "rolling":
                # Fixed-size train window slides with the test window
                train_start = max(0, train_end - test_sz * self.n_splits)
                train_idx   = np.arange(train_start, train_end)
            else:
                train_idx = np.arange(0, train_end)

            test_idx = np.arange(test_start, test_end)

            if len(train_idx) < self.min_train or len(test_idx) == 0:
                continue

            yield train_idx, test_idx

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def print_splits(self, n: int) -> None:
        """Print a human-readable summary of fold boundaries for a dataset of size n."""

        class _Stub:
            def __len__(self):
                return n

        print(f"WalkForwardSplit  n={n}  n_splits={self.n_splits}  "
              f"embargo={self.embargo_days}  mode={self.window_type}")
        for i, (tr, te) in enumerate(self.split(_Stub())):
            print(f"  fold {i}:  train [{tr[0]}..{tr[-1]}] ({len(tr)} rows)  "
                  f"test [{te[0]}..{te[-1]}] ({len(te)} rows)")
