"""
Phase 02 — Feature Engineering Tests
pytest tests/test_feature_pipeline.py

Covers:
  1.  Technical: all expected columns present after add_all_technical()
  2.  Technical: RSI values bounded [0, 100]
  3.  Technical: MACD signal is a smoothed version (variance < MACD line variance)
  4.  Technical: ATR is strictly positive after warm-up
  5.  Technical: BB upper >= BB mid >= BB lower
  6.  Technical: volume_ratio_20 > 0 where volume > 0
  7.  Statistical: z-score has ~zero mean over a long enough window
  8.  Statistical: realized_vol_21 is non-negative
  9.  Statistical: Hurst exponent is in (0, 1) for valid rows
  10. Statistical: return moments (skew/kurtosis) columns present
  11. Cross-asset: spy_corr values bounded [-1, 1]
  12. Cross-asset: relative strength column present after add_relative_strength()
  13. Cross-asset: btc proxy columns present after add_btc_proxy()
  14. Cross-asset: macro columns attached correctly
  15. Regime: hmm_regime values are in {0, 1, 2}
  16. Regime: regime column has no NaN after warm-up (>= n_observations for HMM)
  17. Pipeline: compute_features returns >= 30 feature columns
  18. Pipeline: feature Parquet round-trip preserves shape and dtypes
  19. Pipeline: cache hit — second compute_features() does not rewrite the file
  20. Pipeline: force_refresh updates the cache mtime
  21. No-lookahead: rolling features do not use future values
  22. No-lookahead: features on a row with NaN close remain NaN (not backfilled)
  23. FeatureEngineer.compute_all() processes multiple assets
  24. select_features() returns a non-empty list of column names
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Test fixtures — synthetic OHLCV data (reproducible, no network)
# ---------------------------------------------------------------------------

_TEST_CACHE = Path(__file__).parent / ".test_feature_cache"
_N = 300  # rows — enough for all rolling windows incl. Hurst(100)


def _make_ohlcv(n: int = _N, seed: int = 0, start_price: float = 100.0) -> pd.DataFrame:
    """Generate a synthetic daily OHLCV DataFrame with realistic structure."""
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n, tz="UTC")
    log_r = rng.normal(0.0005, 0.012, size=n)
    close = start_price * np.exp(np.cumsum(log_r))
    noise = rng.uniform(0.995, 1.005, size=n)

    df = pd.DataFrame(
        {
            "Open":      close * rng.uniform(0.99, 1.01, size=n),
            "High":      close * rng.uniform(1.00, 1.02, size=n),
            "Low":       close * rng.uniform(0.98, 1.00, size=n),
            "Close":     close,
            "Adj_Close": close * noise,
            "Volume":    rng.integers(1_000_000, 10_000_000, size=n).astype(float),
        },
        index=dates,
    )
    df.index.name = "Date"
    # Ensure High >= Close >= Low
    df["High"] = df[["High", "Close"]].max(axis=1)
    df["Low"]  = df[["Low",  "Close"]].min(axis=1)
    return df


@pytest.fixture(scope="module")
def ohlcv():
    return _make_ohlcv(seed=42)


@pytest.fixture(scope="module")
def spy_df():
    return _make_ohlcv(seed=1, start_price=400.0)


@pytest.fixture(scope="module")
def btc_df():
    return _make_ohlcv(seed=2, start_price=30_000.0)


@pytest.fixture(scope="module")
def macro_df(ohlcv):
    """Minimal synthetic macro DataFrame aligned to the same calendar."""
    idx = ohlcv.index
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        {
            "VIX":               rng.uniform(10, 40, len(idx)),
            "yield_spread_10_2": rng.uniform(-0.5, 2.0, len(idx)),
            "CPI":               rng.uniform(2.0, 9.0, len(idx)),
            "rate_10y":          rng.uniform(1.0, 5.0, len(idx)),
            "rate_2y":           rng.uniform(0.5, 4.5, len(idx)),
        },
        index=idx,
    )


@pytest.fixture(scope="module")
def tech_df(ohlcv):
    from src.features.technical import add_all_technical
    return add_all_technical(ohlcv.copy())


@pytest.fixture(scope="module")
def stat_df(tech_df):
    from src.features.statistical import add_all_statistical
    return add_all_statistical(tech_df.copy())


@pytest.fixture(scope="module")
def full_feat_df(ohlcv, spy_df, btc_df, macro_df):
    _TEST_CACHE.mkdir(exist_ok=True)
    from src.features.pipeline import FeatureEngineer
    fe = FeatureEngineer(cache_dir=_TEST_CACHE)
    return fe.compute_features(
        df         = ohlcv,
        asset_slug = "TEST",
        spy_df     = spy_df,
        btc_df     = btc_df,
        macro_df   = macro_df,
        force_refresh = True,
    )


@pytest.fixture(scope="module")
def fe():
    _TEST_CACHE.mkdir(exist_ok=True)
    from src.features.pipeline import FeatureEngineer
    return FeatureEngineer(cache_dir=_TEST_CACHE)


# ---------------------------------------------------------------------------
# 1. Technical: expected columns
# ---------------------------------------------------------------------------

def test_technical_columns_present(tech_df):
    expected = [
        "rsi_14", "macd_line", "macd_hist", "macd_signal", "adx_14",
        "bb_lower", "bb_mid", "bb_upper", "bb_width", "bb_pct",
        "atr_14", "hl_range_pct", "obv", "volume_ratio_20",
        "log_return", "return_5d", "return_21d", "gap_return",
        "close_ema20_ratio", "close_ema50_ratio", "close_ema200_ratio",
    ]
    missing = [c for c in expected if c not in tech_df.columns]
    assert not missing, f"Technical feature columns missing: {missing}"


# ---------------------------------------------------------------------------
# 2. RSI bounded [0, 100]
# ---------------------------------------------------------------------------

def test_rsi_bounds(tech_df):
    valid = tech_df["rsi_14"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all(), \
        "RSI values outside [0, 100]"


# ---------------------------------------------------------------------------
# 3. MACD signal smoother than MACD line
# ---------------------------------------------------------------------------

def test_macd_signal_smoother(tech_df):
    line_var   = tech_df["macd_line"].dropna().var()
    signal_var = tech_df["macd_signal"].dropna().var()
    assert signal_var <= line_var, \
        "MACD signal should have lower variance than MACD line (it's the EMA of the line)"


# ---------------------------------------------------------------------------
# 4. ATR strictly positive after warm-up
# ---------------------------------------------------------------------------

def test_atr_positive(tech_df):
    atr = tech_df["atr_14"].dropna()
    assert (atr > 0).all(), "ATR contains non-positive values"


# ---------------------------------------------------------------------------
# 5. Bollinger Band ordering
# ---------------------------------------------------------------------------

def test_bb_ordering(tech_df):
    valid = tech_df[["bb_lower", "bb_mid", "bb_upper"]].dropna()
    assert (valid["bb_upper"] >= valid["bb_mid"]).all(), "BB upper < BB mid"
    assert (valid["bb_mid"]   >= valid["bb_lower"]).all(), "BB mid < BB lower"


# ---------------------------------------------------------------------------
# 6. Volume ratio positive
# ---------------------------------------------------------------------------

def test_volume_ratio_positive(tech_df):
    vr = tech_df["volume_ratio_20"].dropna()
    assert (vr > 0).all(), "volume_ratio_20 contains non-positive values"


# ---------------------------------------------------------------------------
# 7. Z-score approximately zero mean
# ---------------------------------------------------------------------------

def test_zscore_zero_mean(stat_df):
    z = stat_df["zscore_20"].dropna()
    assert abs(z.mean()) < 0.5, f"zscore_20 mean {z.mean():.4f} is unexpectedly large"


# ---------------------------------------------------------------------------
# 8. Realized volatility non-negative
# ---------------------------------------------------------------------------

def test_realized_vol_nonneg(stat_df):
    for col in ["realized_vol_5", "realized_vol_21"]:
        assert col in stat_df.columns, f"Column {col} missing"
        assert (stat_df[col].dropna() >= 0).all(), f"{col} has negative values"


# ---------------------------------------------------------------------------
# 9. Hurst exponent in (0, 1)
# ---------------------------------------------------------------------------

def test_hurst_range(stat_df):
    h = stat_df["hurst_100"].dropna()
    assert len(h) > 0, "No valid Hurst values"
    assert (h > 0).all() and (h <= 1.0).all(), \
        f"Hurst exponent outside (0, 1]: min={h.min():.3f} max={h.max():.3f}"


# ---------------------------------------------------------------------------
# 10. Return moments columns present
# ---------------------------------------------------------------------------

def test_return_moments_present(stat_df):
    for col in ["skew_60", "kurtosis_60"]:
        assert col in stat_df.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# 11. SPY correlation bounded [-1, 1]
# ---------------------------------------------------------------------------

def test_spy_correlation_bounds(full_feat_df):
    for col in ["spy_corr_21", "spy_corr_63"]:
        assert col in full_feat_df.columns, f"Missing {col}"
        valid = full_feat_df[col].dropna()
        assert (valid >= -1.0).all() and (valid <= 1.0).all(), \
            f"{col} values outside [-1, 1]"


# ---------------------------------------------------------------------------
# 12. Relative strength column present
# ---------------------------------------------------------------------------

def test_relative_strength_present(full_feat_df):
    for col in ["rs_vs_spy_21", "rs_vs_spy_63"]:
        assert col in full_feat_df.columns, f"Missing {col}"


# ---------------------------------------------------------------------------
# 13. BTC proxy columns present
# ---------------------------------------------------------------------------

def test_btc_proxy_present(full_feat_df):
    for col in ["btc_return_5d", "btc_return_21d"]:
        assert col in full_feat_df.columns, f"Missing {col}"


# ---------------------------------------------------------------------------
# 14. Macro columns attached
# ---------------------------------------------------------------------------

def test_macro_columns_attached(full_feat_df):
    for col in ["macro_VIX", "macro_yield_spread_10_2"]:
        assert col in full_feat_df.columns, f"Missing {col}"


# ---------------------------------------------------------------------------
# 15. HMM regime values in {0, 1, 2}
# ---------------------------------------------------------------------------

def test_hmm_regime_values(full_feat_df):
    assert "hmm_regime" in full_feat_df.columns, "hmm_regime column missing"
    valid = full_feat_df["hmm_regime"][full_feat_df["hmm_regime"] >= 0]
    unique = set(valid.unique())
    assert unique.issubset({0, 1, 2}), \
        f"Unexpected regime values: {unique - {0, 1, 2}}"


# ---------------------------------------------------------------------------
# 16. Regime non-NaN after warm-up
# ---------------------------------------------------------------------------

def test_hmm_regime_coverage(full_feat_df):
    regime = full_feat_df["hmm_regime"]
    warmup = 50
    tail   = regime.iloc[warmup:]
    na_pct = tail.isna().mean()
    assert na_pct < 0.05, f"Too many NaN in hmm_regime after warm-up: {na_pct:.1%}"


# ---------------------------------------------------------------------------
# 17. Pipeline produces >= 30 feature columns
# ---------------------------------------------------------------------------

def test_feature_count(full_feat_df):
    # Exclude raw OHLCV input columns
    raw_cols = {"Open", "High", "Low", "Close", "Adj_Close", "Volume", "ticker"}
    feat_cols = [c for c in full_feat_df.columns if c not in raw_cols]
    assert len(feat_cols) >= 30, \
        f"Expected >= 30 feature columns, got {len(feat_cols)}: {feat_cols}"


# ---------------------------------------------------------------------------
# 18. Parquet round-trip
# ---------------------------------------------------------------------------

def test_feature_parquet_roundtrip(full_feat_df, tmp_path):
    from src.features.pipeline import _write_parquet, _read_parquet

    path = tmp_path / "feat_roundtrip.parquet"
    _write_parquet(full_feat_df, path)
    loaded = _read_parquet(path)

    assert loaded.shape == full_feat_df.shape, \
        f"Shape mismatch: {loaded.shape} != {full_feat_df.shape}"

    numeric_orig   = full_feat_df.select_dtypes(include="number")
    numeric_loaded = loaded.select_dtypes(include="number")

    pd.testing.assert_frame_equal(
        numeric_orig.sort_index(axis=1),
        numeric_loaded.sort_index(axis=1),
        check_names=False,
        check_freq=False,
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# 19. Cache hit: second call does not rewrite the file
# ---------------------------------------------------------------------------

def test_feature_cache_hit(fe, ohlcv, spy_df, btc_df, macro_df):
    # First call already performed by full_feat_df fixture; grab mtime
    cache_path = _TEST_CACHE / "TEST_features.parquet"
    assert cache_path.exists(), "Feature cache file missing"

    mtime_before = cache_path.stat().st_mtime
    time.sleep(0.05)
    fe.compute_features(
        df=ohlcv, asset_slug="TEST",
        spy_df=spy_df, btc_df=btc_df, macro_df=macro_df,
    )
    mtime_after = cache_path.stat().st_mtime
    assert mtime_before == mtime_after, \
        "Feature cache was rewritten on second call (expected cache hit)"


# ---------------------------------------------------------------------------
# 20. force_refresh updates mtime
# ---------------------------------------------------------------------------

def test_feature_force_refresh(fe, ohlcv, spy_df, btc_df, macro_df):
    cache_path = _TEST_CACHE / "TEST_features.parquet"
    mtime_before = cache_path.stat().st_mtime
    time.sleep(0.1)
    fe.compute_features(
        df=ohlcv, asset_slug="TEST",
        spy_df=spy_df, btc_df=btc_df, macro_df=macro_df,
        force_refresh=True,
    )
    mtime_after = cache_path.stat().st_mtime
    assert mtime_after > mtime_before, \
        "force_refresh=True did not update the cache file"


# ---------------------------------------------------------------------------
# 21. No-lookahead: inserting a NaN row does not back-propagate values
# ---------------------------------------------------------------------------

def test_no_lookahead_zscore():
    """
    Rolling z-score must not fill a NaN that appears at the START of a series
    with future values.
    """
    from src.features.statistical import add_zscore_features

    idx = pd.bdate_range("2023-01-02", periods=50, tz="UTC")
    df  = pd.DataFrame({"Close": np.linspace(100, 150, 50)}, index=idx)
    df.index.name = "Date"
    df.loc[df.index[0], "Close"] = np.nan  # NaN at row 0

    result = add_zscore_features(df.copy())
    assert pd.isna(result["zscore_20"].iloc[0]), \
        "Lookahead detected: z-score at row 0 was filled from future data"


# ---------------------------------------------------------------------------
# 22. No-lookahead: rolling realized vol on NaN prefix stays NaN
# ---------------------------------------------------------------------------

def test_no_lookahead_realized_vol():
    from src.features.statistical import add_realized_volatility

    idx = pd.bdate_range("2023-01-02", periods=30, tz="UTC")
    df  = pd.DataFrame(
        {"Close": np.linspace(100, 130, 30), "log_return": [np.nan] + list(np.random.default_rng(7).normal(0, 0.01, 29))},
        index=idx,
    )
    df.index.name = "Date"

    result = add_realized_volatility(df.copy(), windows=(5,))
    # First 5 rows (warm-up) should be NaN or at most partially filled
    first_valid_idx = result["realized_vol_5"].first_valid_index()
    # The first valid value must not be at row 0 (it needs min 3 observations)
    assert first_valid_idx != idx[0], \
        "Realized vol populated row 0 — possible lookahead"


# ---------------------------------------------------------------------------
# 23. compute_all processes multiple assets
# ---------------------------------------------------------------------------

def test_compute_all_multiple_assets(fe, spy_df, btc_df, macro_df):
    asset_a = _make_ohlcv(seed=10)
    asset_b = _make_ohlcv(seed=11)
    results = fe.compute_all(
        equity_dfs  = {"ASSET_A": asset_a, "ASSET_B": asset_b},
        spy_df      = spy_df,
        btc_df      = btc_df,
        macro_df    = macro_df,
        force_refresh = True,
    )
    assert set(results.keys()) == {"ASSET_A", "ASSET_B"}
    for slug, feat_df in results.items():
        assert len(feat_df) > 0, f"{slug} feature DataFrame is empty"
        assert "hmm_regime" in feat_df.columns, f"{slug} missing hmm_regime"


# ---------------------------------------------------------------------------
# 24. select_features returns non-empty list
# ---------------------------------------------------------------------------

def test_select_features_returns_columns(full_feat_df):
    from src.features.pipeline import FeatureEngineer

    raw_cols = {"Open", "High", "Low", "Close", "Adj_Close", "Volume", "ticker"}
    feat_cols = [c for c in full_feat_df.columns if c not in raw_cols]

    # Create a simple binary label: 1 if next-day return > 0
    df_sel = full_feat_df[feat_cols].copy()
    df_sel["label"] = (full_feat_df["Close"].shift(-1) > full_feat_df["Close"]).astype(int)
    df_sel = df_sel.dropna()

    fe_local = FeatureEngineer()
    selected = fe_local.select_features(df_sel, target_col="label", top_n=15)
    assert isinstance(selected, list), "select_features must return a list"
    assert len(selected) > 0, "select_features returned an empty list"
    for col in selected:
        assert col in feat_cols, f"select_features returned unknown column: {col}"
