"""
Phase 04 — Signal Generation

Converts ensemble probability output into a tradeable signal DataFrame.

Signal encoding
---------------
  2 = UP   → long  (buy)
  0 = DOWN → short (or cash in long-only mode)
  1 = FLAT → cash  (no position)
 -1 = insufficient confidence → carry previous signal

Signal source
-------------
Versioned, sealed out-of-sample predictions from causal training only.
Historical final-model backfill is prohibited. Legacy caches require rebuilding.

Confidence filtering
---------------------
Only emit a non-FLAT signal if max(p_down, p_flat, p_up) >= confidence_threshold.
Low-confidence bars are marked as FLAT (no position), reducing churning and noise.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODELS_ROOT  = _PROJECT_ROOT / "data" / "models"


# ---------------------------------------------------------------------------
# Load OOS predictions
# ---------------------------------------------------------------------------

def load_oos_predictions(ticker: str) -> Optional[pd.DataFrame]:
    """
    Load the walk-forward OOS probability DataFrame saved by ModelTrainer.

    Returns None if the file does not exist (pre-Phase 04 training run).
    Columns: p_down  p_flat  p_up  true_label
    """
    path = _MODELS_ROOT / ticker / "oos_predictions.parquet"
    if not path.exists():
        return None
    df = pq.read_table(str(path)).to_pandas()
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


# ---------------------------------------------------------------------------
# Final-model fallback inference
# ---------------------------------------------------------------------------

def _infer_with_final_models(
    ticker: str,
    feat_df: pd.DataFrame,
    feat_cols: list[str],
) -> pd.DataFrame:
    """
    Apply the saved final models to the full feature history.
    Warned: introduces look-ahead bias for early periods.
    """
    import json
    from src.models.ensemble import EnsembleClassifier
    from src.models.lgbm_model import LGBMModel
    from src.models.lstm_model import LSTMModel
    from src.models.xgb_model import XGBModel

    out_dir = _MODELS_ROOT / ticker
    xgb  = XGBModel.load(str(out_dir / "xgb_final.json"))
    lgbm = LGBMModel.load(str(out_dir / "lgbm_final.txt"))
    lstm_path = out_dir / "lstm_final.pt"
    lstm = LSTMModel.load(str(lstm_path)) if lstm_path.exists() else None

    X = feat_df[feat_cols].dropna()
    x_arr = X.values.astype(np.float32)

    xgb_p  = xgb.predict_proba(x_arr)
    lgbm_p = lgbm.predict_proba(x_arr)
    lstm_p = lstm.predict_proba(x_arr) if lstm else np.full((len(x_arr), 3), 1/3)

    weights = [1/3, 1/3, 1/3] if lstm else [0.5, 0.5, 0.0]
    ens = EnsembleClassifier(weights=weights)
    proba = ens.predict_proba(xgb_p, lgbm_p, lstm_p)

    return pd.DataFrame(
        {"p_down": proba[:, 0], "p_flat": proba[:, 1], "p_up": proba[:, 2]},
        index=X.index,
    )


# ---------------------------------------------------------------------------
# Signal DataFrame builder
# ---------------------------------------------------------------------------

def build_signal_df(
    ticker: str,
    feat_df: pd.DataFrame,
    confidence_threshold: float = 0.38,
    policy_engine=None,
) -> pd.DataFrame:
    """
    Build a time-indexed signal DataFrame for backtesting.

    Returns a DataFrame with columns:
      p_down, p_flat, p_up   : ensemble probabilities
      signal                 : {0=DOWN, 1=FLAT, 2=UP}
      confidence             : max(p_down, p_flat, p_up)
      filtered_signal        : signal if confidence >= threshold, else 1 (FLAT)
      conf_threshold         : the threshold applied per bar (scalar or dynamic)
      true_label             : actual class (only from OOS; NaN for fallback)

    Parameters
    ----------
    ticker               : asset symbol, must match data/models/{ticker}/ path
    feat_df              : Phase 02 feature matrix (provides close prices + dates)
    confidence_threshold : min max-probability to emit a non-FLAT signal
                           (ignored when policy_engine is supplied)
    policy_engine        : optional src.risk.policy_engine.PolicyEngine — when
                           given, the confidence threshold becomes dynamic
                           per bar (tightened while the HMM regime is
                           transitioning) and ineligible bars are forced FLAT.
    """
    import json
    feat_path = _MODELS_ROOT / ticker / "feature_cols.json"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"No trained models found for {ticker}. Run train.py first."
        )
    from src.models.bundle import validate_bundle
    validate_bundle(_MODELS_ROOT / ticker)
    if feat_df.attrs.get("causality_version") != 2:
        raise ValueError("Legacy features are not eligible for validated backtesting")
    feat_cols = json.loads(feat_path.read_text())

    # ── Try OOS file first ─────────────────────────────────────────────
    proba_df = load_oos_predictions(ticker)
    source   = "oos"

    if proba_df is None:
        raise FileNotFoundError(f"{ticker}: missing OOS predictions; final-model backfill is prohibited")

    # Align with feat_df close prices (inner join on index)
    close    = feat_df["Close"].rename("close")
    df       = proba_df.join(close, how="inner")
    df.index.name = "Date"

    df["signal"]    = np.argmax(df[["p_down", "p_flat", "p_up"]].values, axis=1)
    df["confidence"]= df[["p_down", "p_flat", "p_up"]].max(axis=1)

    if policy_engine is not None:
        # Note: only the dynamic threshold is applied here.  The policy's
        # eligibility gate is enforced by RiskEngine (blocks NEW entries
        # only) — flattening the signal here would force-close open
        # positions and churn re-entries.
        policy = policy_engine.compute_policy(feat_df).reindex(df.index).ffill()
        thresholds = policy["conf_threshold"].fillna(confidence_threshold).values
    else:
        thresholds = np.full(len(df), confidence_threshold)

    df["conf_threshold"] = thresholds
    df["filtered_signal"] = np.where(
        df["confidence"].values >= thresholds,
        df["signal"],
        1,   # fall back to FLAT when model is unsure
    )
    df["source"] = source

    log.info(
        "%s signals: %d rows  source=%s  "
        "DOWN=%.1f%%  FLAT=%.1f%%  UP=%.1f%%  "
        "(conf≥%.2f → active=%.1f%%)",
        ticker, len(df), source,
        (df["filtered_signal"] == 0).mean() * 100,
        (df["filtered_signal"] == 1).mean() * 100,
        (df["filtered_signal"] == 2).mean() * 100,
        float(np.mean(thresholds)),
        (df["confidence"].values >= thresholds).mean() * 100,
    )
    return df
