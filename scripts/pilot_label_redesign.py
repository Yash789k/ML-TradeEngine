"""
Lever B pilot — label redesign experiment (Phase 03 refinement)

Trains XGB+LGBM (no LSTM, 30 Optuna trials) on 3 representative tickers with
redesigned labels:

    horizon   5 → 10 bars      (two-week move, less daily noise)
    threshold 0.5% → 1.5%      (only meaningful moves labelled UP/DOWN)

Models are written to data/models_pilot/ so production models under
data/models/ are untouched.

After training, compares old vs new on:
  - OOS accuracy / F1
  - UP-signal precision (when model says UP, was it UP?)
  - Simple next-bar long-only simulation from OOS predictions

Usage:
    python3 scripts/pilot_label_redesign.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pilot")

ROOT       = Path(__file__).resolve().parents[1]
PILOT_DIR  = ROOT / "data" / "models_pilot"
PROD_DIR   = ROOT / "data" / "models"

TICKERS   = ["AAPL", "NVDA", "SPY"]
HORIZON   = 10
THRESHOLD = 0.015
N_TRIALS  = 30


def quick_longonly_sim(oos: pd.DataFrame, close: pd.Series,
                       cost: float = 0.0015) -> dict:
    """
    Vectorised next-bar long-only simulation from OOS probabilities.
    Signal at t-1 decides exposure for t's close-to-close return.
    """
    common = oos.index.intersection(close.index)
    oos    = oos.loc[common]
    px     = close.loc[common]

    sig  = np.argmax(oos[["p_down", "p_flat", "p_up"]].values, axis=1)
    long = (pd.Series(sig, index=common).shift(1) == 2).astype(float)
    ret  = px.pct_change().fillna(0.0)

    churn  = long.diff().abs().fillna(0.0)
    strat  = long * ret - churn * cost

    rf   = 1.05 ** (1 / 252) - 1
    sd   = strat.std(ddof=1)
    sharpe = float((strat.mean() - rf) / sd * np.sqrt(252)) if sd > 0 else 0.0
    cagr   = float((1 + strat).prod() ** (252 / max(len(strat), 1)) - 1)
    eq     = (1 + strat).cumprod()
    mdd    = float(((eq - eq.cummax()) / eq.cummax()).min())
    exposure = float(long.mean())
    return {"sharpe": round(sharpe, 3), "cagr": round(cagr, 4),
            "max_dd": round(mdd, 4), "exposure": round(exposure, 3)}


def signal_stats(oos: pd.DataFrame) -> dict:
    sig = np.argmax(oos[["p_down", "p_flat", "p_up"]].values, axis=1)
    lbl = oos["true_label"].values
    up  = sig == 2
    return {
        "pct_up":   round(float(up.mean()), 3),
        "pct_down": round(float((sig == 0).mean()), 3),
        "up_precision": round(float((lbl[up] == 2).mean()), 3) if up.sum() else None,
        "n": len(oos),
    }


def main() -> None:
    from src.features.pipeline import _feature_path, _read_parquet
    from src.models.trainer import ModelTrainer

    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    trainer = ModelTrainer(models_dir=PILOT_DIR, experiment="phase03-pilot-labels")

    report: dict = {}

    for ticker in TICKERS:
        log.info("══════════ PILOT %s  (horizon=%d, threshold=%.3f) ══════════",
                 ticker, HORIZON, THRESHOLD)
        features = _read_parquet(_feature_path(ticker))

        trainer.run(
            features   = features,
            ticker     = ticker,
            horizon    = HORIZON,
            threshold  = THRESHOLD,
            n_splits   = 5,
            n_trials_xgb  = N_TRIALS,
            n_trials_lgbm = N_TRIALS,
            skip_lstm  = True,
        )

        close = features["Close"]

        new_oos = pd.read_parquet(PILOT_DIR / ticker / "oos_predictions.parquet")
        old_oos = pd.read_parquet(PROD_DIR / ticker / "oos_predictions.parquet")

        report[ticker] = {
            "old": {**signal_stats(old_oos), **quick_longonly_sim(old_oos, close)},
            "new": {**signal_stats(new_oos), **quick_longonly_sim(new_oos, close)},
        }

    print("\n" + "═" * 78)
    print("  PILOT RESULTS — old labels (h=5, ±0.5%) vs new labels (h=10, ±1.5%)")
    print("═" * 78)
    print(f"  {'':10} {'%UP':>6} {'%DN':>6} {'UPprec':>7} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'Expo':>6}")
    for ticker, r in report.items():
        for tag in ("old", "new"):
            m = r[tag]
            print(f"  {ticker + ' ' + tag:<10} {m['pct_up']*100:>5.1f}% {m['pct_down']*100:>5.1f}% "
                  f"{(m['up_precision'] or 0)*100:>6.1f}% {m['sharpe']:>7.3f} "
                  f"{m['cagr']*100:>6.1f}% {m['max_dd']*100:>6.1f}% {m['exposure']*100:>5.1f}%")
        print()

    (PILOT_DIR / "pilot_report.json").write_text(json.dumps(report, indent=2))
    print(f"  Report saved → {PILOT_DIR / 'pilot_report.json'}")


if __name__ == "__main__":
    main()
