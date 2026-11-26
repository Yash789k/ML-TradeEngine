"""
Phase 06E — Live Signal Generator

Fetches fresh OHLCV, rebuilds the feature matrix, runs the Phase 03 ensemble
on the most recent bar, and returns a structured LiveSignal for today.

Signal encoding (matches Phase 04 / Phase 05):
  2 = UP   → long
  1 = FLAT → no position
  0 = DOWN → exit / short

No historical OOS predictions are used here — we always apply the saved final
models to the latest available feature row to avoid any look-ahead dependency
on future training data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODELS_ROOT  = _PROJECT_ROOT / "data" / "models"

_CONFIDENCE_THRESHOLD: float = 0.38   # matches Phase 04 default


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class LiveSignal:
    """All information needed to act on today's signal."""

    ticker:       str
    date:         datetime         # UTC timestamp of the signal bar

    direction:    int              # 2=UP, 1=FLAT, 0=DOWN
    confidence:   float           # max(p_down, p_flat, p_up)
    p_up:         float
    p_flat:       float
    p_down:       float

    kelly_frac:   float           # half-Kelly fraction (0 → skip trade)
    atr:          float           # ATR-14 for stop calculation
    stop_loss:    float           # entry_price - 2×ATR (set after fill price known)
    close:        float           # last close price used for sizing estimate

    # Set by MetaSignalGenerator (src/live/meta_signal_generator.py); default
    # values keep this dataclass fully backward-compatible with the static
    # ensemble path and with existing SignalLogger rows.
    size_cap:          float = 1.0
    eligible:          bool = True
    strategy_mode:     str           = "ensemble"   # "ensemble" | "meta"
    selected_strategy: Optional[str] = None         # zoo strategy name, or "EW"

    # Human-readable label
    @property
    def label(self) -> str:
        return {2: "UP", 1: "FLAT", 0: "DOWN"}.get(self.direction, "UNKNOWN")

    def is_actionable(self) -> bool:
        """True when the model emits a directional signal with positive Kelly edge."""
        return self.direction != 1 and self.kelly_frac > 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_feature_cols(ticker: str) -> list[str]:
    path = _MODELS_ROOT / ticker / "feature_cols.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No trained models found for {ticker}. Run train.py first."
        )
    return json.loads(path.read_text())


def _load_win_loss_ratio(ticker: str, fallback: float = 1.5) -> float:
    path = _PROJECT_ROOT / "data" / "backtest" / ticker / "backtest_summary.json"
    if not path.exists():
        return fallback
    data = json.loads(path.read_text())
    m = data.get("strategy_metrics", {})
    avg_win  = m.get("avg_win",  0.0)
    avg_loss = abs(m.get("avg_loss", 0.0))
    if avg_win <= 0 or avg_loss <= 0:
        return fallback
    return avg_win / avg_loss


def _kelly_fraction(p_win: float, win_loss_ratio: float, k: float = 0.5) -> float:
    if p_win <= 0 or p_win >= 1 or win_loss_ratio <= 0:
        return 0.0
    p_loss = 1.0 - p_win
    f_full = (p_win * win_loss_ratio - p_loss) / win_loss_ratio
    return float(max(f_full * k, 0.0))


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def _infer_latest_bar(
    ticker: str,
    feat_df: pd.DataFrame,
    feat_cols: list[str],
) -> dict[str, float]:
    """
    Run the saved final ensemble models on the most recent row of feat_df.

    Returns {"p_down": float, "p_flat": float, "p_up": float}
    """
    from src.models.ensemble import EnsembleClassifier
    from src.models.lgbm_model import LGBMModel
    from src.models.lstm_model import LSTMModel
    from src.models.xgb_model import XGBModel

    out_dir = _MODELS_ROOT / ticker

    from src.models.bundle import validate_bundle
    metadata = validate_bundle(out_dir)
    for key in ["feature_history_start", "hmm_calibration_sha256"]:
        if metadata.get(key) is not None and metadata[key] != feat_df.attrs.get(key):
            raise ValueError(f"{ticker}: feature calibration differs from model bundle ({key}); rebuild bundle")
    trained_through = pd.Timestamp(metadata["trained_through"])
    if pd.Timestamp(feat_df.index[-1]) <= trained_through:
        raise ValueError(f"{ticker}: inference date must follow model training cutoff")
    xgb  = XGBModel.load(str(out_dir / "xgb_final.json"))
    lgbm = LGBMModel.load(str(out_dir / "lgbm_final.txt"))
    lstm_path = out_dir / "lstm_final.pt"
    lstm = None if metadata["skip_lstm"] else LSTMModel.load(str(lstm_path))

    missing = set(feat_cols) - set(feat_df.columns)
    if missing:
        raise ValueError(f"{ticker}: required model features missing: {sorted(missing)}; rebuild matching feature bundle")
    needed = lstm.seq_len if lstm else 1
    X = feat_df[feat_cols].tail(needed)
    if len(X) < needed or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(f"{ticker}: need {needed} consecutive complete latest feature rows")
    x_arr = X.to_numpy(dtype=np.float32)
    xgb_p = xgb.predict_proba(x_arr[-1:])
    lgbm_p = lgbm.predict_proba(x_arr[-1:])
    lstm_p = lstm.predict_proba(x_arr)[-1:] if lstm else np.full((1,3), 1/3)

    weights = [1 / 3, 1 / 3, 1 / 3] if lstm else [0.5, 0.5, 0.0]
    ens     = EnsembleClassifier(weights=weights)
    proba   = ens.predict_proba(xgb_p, lgbm_p, lstm_p)[0]   # shape (3,)

    if not np.isfinite(proba).all() or (proba < 0).any() or not np.isclose(proba.sum(), 1):
        raise ValueError(f"{ticker}: invalid model probabilities")
    return {"p_down": float(proba[0]), "p_flat": float(proba[1]), "p_up": float(proba[2])}


# ---------------------------------------------------------------------------
# SignalGenerator
# ---------------------------------------------------------------------------

class SignalGenerator:
    """
    Generate a daily live signal for one or more tickers.

    Usage
    -----
    gen = SignalGenerator()
    signal = gen.generate("AAPL")
    """

    def __init__(
        self,
        confidence_threshold: float = _CONFIDENCE_THRESHOLD,
        atr_multiplier:       float = 2.0,
        policy_engine=None,
    ) -> None:
        """
        policy_engine : optional src.risk.policy_engine.PolicyEngine — when
                        given, confidence threshold and ATR stop multiplier
                        are computed per ticker from the latest market state
                        (dynamic policy layer) instead of the static values.
        """
        self.confidence_threshold = confidence_threshold
        self.atr_multiplier       = atr_multiplier
        self.policy_engine        = policy_engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        ticker: str,
        force_refresh: bool = True,
    ) -> LiveSignal:
        """
        Fetch fresh data, rebuild features, and infer today's signal.

        Parameters
        ----------
        ticker        : asset symbol, must have trained models in data/models/
        force_refresh : bypass DataLoader cache and fetch live data (default True)

        Returns
        -------
        LiveSignal dataclass
        """
        from src.data.loader import DataLoader
        from src.features.pipeline import FeatureEngineer

        log.info("[SignalGenerator] %s — fetching live data …", ticker)

        loader = DataLoader(max_age_hours=0.5 if force_refresh else 12.0)
        fe     = FeatureEngineer(max_age_hours=0.5 if force_refresh else 12.0)

        # ── 1. Load OHLCV + supporting series ──────────────────────────
        ohlcv = loader.load_equity(ticker, force_refresh=force_refresh)

        required_columns = _load_feature_cols(ticker)
        spy_df = macro = btc_df = None
        for name, fn in [
            ("SPY",      lambda: loader.load_equity("SPY", force_refresh=force_refresh)),
            ("macro",    lambda: loader.load_macro(force_refresh=force_refresh)),
            ("BTC/USDT", lambda: loader.load_crypto("BTC/USDT", force_refresh=force_refresh)),
        ]:
            if name == "macro" and not any(c.startswith("macro_") for c in required_columns):
                continue
            if name == "BTC/USDT" and not any(c.startswith("btc_") for c in required_columns):
                continue
            try:
                if name == "SPY":
                    spy_df = fn()
                elif name == "macro":
                    macro = fn()
                else:
                    btc_df = fn()
            except Exception as exc:
                log.debug("Could not load %s: %s", name, exc)

        # ── 2. Compute feature matrix ───────────────────────────────────
        feat_df = fe.compute_features(
            df         = ohlcv,
            asset_slug = ticker,
            spy_df     = spy_df,
            btc_df     = btc_df,
            macro_df   = macro,
            force_refresh = force_refresh,
        )

        # ── 3. Load feature columns expected by models ──────────────────
        feat_cols = _load_feature_cols(ticker)

        # ── 4. Run ensemble inference on latest bar ─────────────────────
        log.info("[SignalGenerator] %s — running ensemble inference …", ticker)
        proba = _infer_latest_bar(ticker, feat_df, feat_cols)

        p_down, p_flat, p_up = proba["p_down"], proba["p_flat"], proba["p_up"]
        raw_signal  = int(np.argmax([p_down, p_flat, p_up]))
        confidence  = max(p_down, p_flat, p_up)

        # ── 5. Resolve operating policy (static or dynamic) ─────────────
        conf_threshold = self.confidence_threshold
        atr_multiplier = self.atr_multiplier
        policy_eligible = True
        size_cap = 1.0
        if self.policy_engine is not None:
            decision = self.policy_engine.decide_latest(feat_df)
            conf_threshold  = decision.confidence_threshold
            atr_multiplier  = decision.atr_multiplier
            policy_eligible = decision.eligible
            size_cap = decision.size_cap

        # ── 5b. Confidence filter + eligibility gate ─────────────────────
        direction = raw_signal if (confidence >= conf_threshold and policy_eligible) else 1

        # ── 6. Kelly sizing (uses calibrated win/loss ratio) ─────────────
        wl_ratio   = 1.5  # fixed exploratory assumption; no full-history outcome calibration
        kelly_frac = _kelly_fraction(p_up, wl_ratio) if direction == 2 else 0.0

        # ── 7. ATR stop level (estimated from last close) ────────────────
        last_row  = feat_df.iloc[-1]
        if not np.isfinite(last_row.get("atr_14", np.nan)):
            raise ValueError(f"{ticker}: latest ATR missing")
        atr_val   = float(last_row.get("atr_14", 0.0))
        last_close = float(feat_df["Close"].iloc[-1])
        stop_loss  = last_close - atr_multiplier * atr_val  # for long positions

        signal_date = feat_df.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)

        signal = LiveSignal(
            ticker      = ticker,
            size_cap    = size_cap,
            eligible    = policy_eligible,
            date        = signal_date,
            direction   = direction,
            confidence  = round(confidence, 4),
            p_up        = round(p_up,    4),
            p_flat      = round(p_flat,  4),
            p_down      = round(p_down,  4),
            kelly_frac  = round(kelly_frac, 4),
            atr         = round(atr_val,    4),
            stop_loss   = round(stop_loss,  4),
            close       = round(last_close, 4),
        )

        log.info(
            "[SignalGenerator] %s → %s  conf=%.3f  kelly=%.3f  close=%.2f  stop=%.2f",
            ticker, signal.label, confidence, kelly_frac, last_close, stop_loss,
        )
        return signal

    def generate_batch(
        self,
        tickers: list[str],
        force_refresh: bool = True,
    ) -> dict[str, LiveSignal]:
        """Generate signals for multiple tickers. Skips tickers that fail."""
        results: dict[str, LiveSignal] = {}
        self.errors = {}
        for ticker in tickers:
            try:
                results[ticker] = self.generate(ticker, force_refresh=force_refresh)
            except Exception as exc:
                log.error("[SignalGenerator] %s failed: %s", ticker, exc)
                self.errors[ticker] = str(exc)
        return results
