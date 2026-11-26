"""
Phase 02 — FeatureEngineer
Orchestrates all feature groups, handles Parquet caching, and optionally
prunes low-importance columns via XGBoost SHAP values.

Cache pattern mirrors Phase 01 DataLoader:
  cache file exists AND age < max_age_hours?
    YES → read Parquet (fast, no network / computation)
    NO  → recompute → write Parquet → return

Feature matrix files are stored alongside the raw OHLCV files under:
  data/features/{asset_slug}_features.parquet

Feature inventory (32 features minimum per asset)
--------------------------------------------------
Technical   (14) : rsi_14, macd_line, macd_hist, macd_signal, adx_14,
                   bb_lower, bb_mid, bb_upper, bb_width, bb_pct, atr_14,
                   hl_range_pct, obv, volume_ratio_20
Price struct (7) : log_return, return_5d, return_21d, gap_return,
                   close_ema20_ratio, close_ema50_ratio, close_ema200_ratio
Statistical (8)  : zscore_20, zscore_60, realized_vol_5, realized_vol_21,
                   hurst_100, skew_60, kurtosis_60
Cross-asset (7)  : spy_corr_21, spy_corr_63, rs_vs_spy_21, rs_vs_spy_63,
                   btc_return_5d, btc_return_21d, macro_VIX ...
Regime      (1)  : hmm_regime
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.features.technical   import add_all_technical
from src.features.statistical  import add_all_statistical
from src.features.cross_asset  import (
    add_spy_correlation,
    add_relative_strength,
    add_btc_proxy,
    add_macro_features,
)
from src.features.regime import add_hmm_regime

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT   = Path(__file__).resolve().parents[2]
_FEATURE_ROOT   = _PROJECT_ROOT / "data" / "features"
_FEATURE_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Parquet helpers  (same pattern as loader.py)
# ---------------------------------------------------------------------------

def _feature_path(asset_slug: str) -> Path:
    return _FEATURE_ROOT / f"{asset_slug}_features.parquet"


def _is_stale(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return True
    return (time.time() - path.stat().st_mtime) / 3600 > max_age_hours


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(table, path, compression="snappy")


def _read_parquet(path: Path) -> pd.DataFrame:
    df = pq.read_table(path).to_pandas()
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


# ---------------------------------------------------------------------------
# FeatureEngineer
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Compute and cache the full feature matrix for any asset.

    Usage
    -----
    from src.data.loader import DataLoader
    from src.features.pipeline import FeatureEngineer

    loader = DataLoader()
    fe     = FeatureEngineer()

    aapl    = loader.load_equity("AAPL")
    spy     = loader.load_equity("SPY")
    btc     = loader.load_crypto("BTC/USDT")
    macro   = loader.load_macro()

    features = fe.compute_features(
        df       = aapl,
        asset_slug = "AAPL",
        spy_df   = spy,
        btc_df   = btc,
        macro_df = macro,
    )

    Parameters
    ----------
    cache_dir      : override default features store location
    max_age_hours  : stale threshold; default matches DataLoader (12 h)
    n_hmm_states   : number of hidden market regimes (default 3)
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_age_hours: float = 12.0,
        n_hmm_states: int = 3,
    ) -> None:
        self.cache_dir      = Path(cache_dir) if cache_dir else _FEATURE_ROOT
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_hours  = max_age_hours
        self.n_hmm_states   = n_hmm_states
        self._hmm_models: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Core compute
    # ------------------------------------------------------------------

    def compute_features(
        self,
        df: pd.DataFrame,
        asset_slug: str = "asset",
        spy_df:   Optional[pd.DataFrame] = None,
        btc_df:   Optional[pd.DataFrame] = None,
        macro_df: Optional[pd.DataFrame] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Build the full feature matrix for one asset.

        Parameters
        ----------
        df         : raw OHLCV DataFrame (output of DataLoader.load_equity/crypto)
        asset_slug : used for cache file naming
        spy_df     : SPY OHLCV (enables correlation + relative-strength features)
        btc_df     : BTC/USDT OHLCV (enables crypto-sentiment proxy features)
        macro_df   : FRED macro DataFrame (enables VIX, yield-curve features)
        """
        path = self.cache_dir / f"{asset_slug}_features.parquet"

        if not force_refresh and not _is_stale(path, self.max_age_hours):
            return _read_parquet(path)

        feat = df.copy()

        # ── technical ─────────────────────────────────────────────────
        feat = add_all_technical(feat)

        # ── statistical ───────────────────────────────────────────────
        feat = add_all_statistical(feat)

        # ── cross-asset ───────────────────────────────────────────────
        if spy_df is not None:
            feat = add_spy_correlation(feat, spy_df)
            feat = add_relative_strength(feat, spy_df)

        if btc_df is not None:
            feat = add_btc_proxy(feat, btc_df)

        if macro_df is not None:
            feat = add_macro_features(feat, macro_df)

        # ── regime ────────────────────────────────────────────────────
        feat, hmm = add_hmm_regime(feat, n_states=self.n_hmm_states)
        self._hmm_models[asset_slug] = hmm

        _write_parquet(feat, path)
        return feat

    # ------------------------------------------------------------------
    # Cache-aware load
    # ------------------------------------------------------------------

    def load_features(
        self,
        asset_slug: str,
        force_refresh: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load from cache if fresh, otherwise call compute_features(**kwargs).
        Pass the same keyword arguments as compute_features when a re-compute
        might be needed.
        """
        path = self.cache_dir / f"{asset_slug}_features.parquet"

        if not force_refresh and not _is_stale(path, self.max_age_hours):
            return _read_parquet(path)

        if "df" not in kwargs:
            raise ValueError(
                "Cache is stale or missing — supply df= (and optional spy_df, "
                "btc_df, macro_df) to recompute."
            )
        return self.compute_features(asset_slug=asset_slug, **kwargs)

    # ------------------------------------------------------------------
    # SHAP-based feature selection  (optional, requires xgboost + shap)
    # ------------------------------------------------------------------

    def select_features(
        self,
        feat_df: pd.DataFrame,
        target_col: str = "label",
        top_n: int = 20,
        importance_threshold: float = 0.005,
    ) -> list[str]:
        """
        Rank features by mean |SHAP| value from a quick XGBoost fit.
        Returns the list of feature column names that pass the threshold
        (or the top_n features, whichever is smaller).

        This is an approximation pass — the Phase 03 ML pipeline will refine
        selection further inside walk-forward CV.

        Parameters
        ----------
        feat_df              : feature DataFrame, must include `target_col`
        target_col           : binary or multi-class label column
        top_n                : hard cap on returned features
        importance_threshold : min mean |SHAP| to keep a feature
        """
        try:
            import xgboost as xgb  # type: ignore
            import shap             # type: ignore
        except ImportError as exc:
            raise ImportError("xgboost and shap are required for select_features.") from exc

        feature_cols = [c for c in feat_df.columns if c != target_col]
        X = feat_df[feature_cols].copy()
        y = feat_df[target_col].copy()

        # Drop rows where either X or y has NaN
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]

        if len(X) < 50:
            raise ValueError("Too few clean rows for SHAP selection (need >= 50).")

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)

        # SHAP >= 0.40 returns a 3-D array (n_samples, n_features, n_classes)
        # for multi-class models.  Older versions return a list of 2-D arrays.
        if isinstance(shap_vals, list):
            # list of (n_samples, n_features) — one array per class
            mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
        elif shap_vals.ndim == 3:
            # (n_samples, n_features, n_classes) → mean over samples and classes
            mean_abs = np.abs(shap_vals).mean(axis=(0, 2))
        else:
            # (n_samples, n_features) — binary or regression
            mean_abs = np.abs(shap_vals).mean(axis=0)

        importance = pd.Series(mean_abs, index=feature_cols).sort_values(ascending=False)
        selected   = importance[importance >= importance_threshold].head(top_n).index.tolist()
        return selected

    # ------------------------------------------------------------------
    # Convenience: compute features for all default assets
    # ------------------------------------------------------------------

    def compute_all(
        self,
        equity_dfs:  dict[str, pd.DataFrame],
        spy_df:      Optional[pd.DataFrame] = None,
        btc_df:      Optional[pd.DataFrame] = None,
        macro_df:    Optional[pd.DataFrame] = None,
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Compute and cache features for every asset in `equity_dfs`.

        Parameters
        ----------
        equity_dfs : {slug: raw_ohlcv_df}  — output of DataLoader.load_all()
        """
        results: dict[str, pd.DataFrame] = {}
        for slug, raw_df in equity_dfs.items():
            print(f"  [features] computing {slug} …")
            results[slug] = self.compute_features(
                df           = raw_df,
                asset_slug   = slug,
                spy_df       = spy_df,
                btc_df       = btc_df,
                macro_df     = macro_df,
                force_refresh= force_refresh,
            )
        return results
