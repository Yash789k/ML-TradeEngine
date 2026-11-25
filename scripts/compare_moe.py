"""
MoE gate benchmark — static soft-vote vs context-gated ensemble.

Two stages:

  1. (--train) Pilot retrain on a small ticker set with per-model OOS
     probability persistence (models isolated in data/models_moe/).
     Only needed once; skipped when the parquet files already exist.

  2. Post-hoc walk-forward gate evaluation on the saved per-model OOS
     predictions:
       - OOS rows are cut into sequential blocks.
       - For block k the gate trains on all rows ending 10 bars
         (embargo) before the block start — strictly past information,
         mirroring the expanding WalkForwardSplit used in training.
       - Block 0 has no history → uniform gate = static baseline.

     Both ensembles are turned into trading signals with the standard
     Phase 04 machinery (build signal → run_simulation → compute_metrics)
     so Sharpe/Calmar are directly comparable to existing results.

Outputs
-------
  data/research/moe/moe_comparison.json     per-ticker + aggregate metrics
  data/research/moe/gate_weights_{tkr}.parquet   per-bar weights (for plots)

Usage:
    python3 scripts/compare_moe.py --train      # first run (retrains pilot)
    python3 scripts/compare_moe.py              # evaluation only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("compare_moe")

ROOT      = Path(__file__).resolve().parents[1]
MOE_MODELS = ROOT / "data" / "models_moe"
OUT_DIR   = ROOT / "data" / "research" / "moe"

TICKERS   = ["AAPL", "NVDA", "SPY"]
N_TRIALS  = 20
N_BLOCKS  = 5        # sequential gate-evaluation blocks
EMBARGO   = 10       # bars between gate train window and eval block
CONF_THR  = 0.38     # match Phase 04 default so results are comparable


# ---------------------------------------------------------------------------
# Stage 1 — pilot retrain
# ---------------------------------------------------------------------------

def pilot_train() -> None:
    from src.features.pipeline import _feature_path, _read_parquet
    from src.models.trainer import ModelTrainer

    MOE_MODELS.mkdir(parents=True, exist_ok=True)
    trainer = ModelTrainer(models_dir=MOE_MODELS, experiment="phase03-moe-pilot")

    for ticker in TICKERS:
        pm = MOE_MODELS / ticker / "oos_predictions_per_model.parquet"
        if pm.exists():
            log.info("%s: per-model OOS already present — skipping retrain", ticker)
            continue
        log.info("══════════ MoE pilot retrain %s ══════════", ticker)
        features = _read_parquet(_feature_path(ticker))
        trainer.run(
            features      = features,
            ticker        = ticker,
            n_splits      = 5,
            n_trials_xgb  = N_TRIALS,
            n_trials_lgbm = N_TRIALS,
            skip_lstm     = False,
            lstm_epochs   = 30,
        )


# ---------------------------------------------------------------------------
# Stage 2 — walk-forward gate evaluation
# ---------------------------------------------------------------------------

def _simulate(proba: np.ndarray, index: pd.Index, close: pd.Series) -> dict:
    """Signals → simulation → metrics, using the Phase 04 pipeline."""
    from src.backtest.metrics import compute_metrics
    from src.backtest.simulator import run_simulation

    df = pd.DataFrame(
        {"p_down": proba[:, 0], "p_flat": proba[:, 1], "p_up": proba[:, 2]},
        index=index,
    )
    df["close"]      = close.reindex(index)
    df               = df.dropna(subset=["close"])
    df["signal"]     = np.argmax(df[["p_down", "p_flat", "p_up"]].values, axis=1)
    df["confidence"] = df[["p_down", "p_flat", "p_up"]].max(axis=1)
    df["filtered_signal"] = np.where(df["confidence"] >= CONF_THR, df["signal"], 1)

    result = run_simulation(df, mode="long_only")
    return compute_metrics(result.daily_returns)


def evaluate_ticker(ticker: str) -> dict:
    from src.models.gating import GatingModel, build_gate_context

    pm_path = MOE_MODELS / ticker / "oos_predictions_per_model.parquet"
    if not pm_path.exists():
        raise FileNotFoundError(f"{pm_path} missing — run with --train first")

    pm = pd.read_parquet(pm_path)
    pm.index = pd.to_datetime(pm.index, utc=True)
    feat = pd.read_parquet(ROOT / "data" / "features" / f"{ticker}_features.parquet")
    feat.index = pd.to_datetime(feat.index, utc=True)

    probs = [
        pm[["xgb_p_down", "xgb_p_flat", "xgb_p_up"]].values,
        pm[["lgbm_p_down", "lgbm_p_flat", "lgbm_p_up"]].values,
        pm[["lstm_p_down", "lstm_p_flat", "lstm_p_up"]].values,
    ]
    y = pm["true_label"].values.astype(int)
    n = len(y)

    ctx     = build_gate_context(feat, probs, y, index=pm.index)
    correct = np.stack([(np.argmax(p, axis=1) == y).astype(float) for p in probs], axis=1)

    # ── static baseline: equal weights over the three experts ──────────
    static_proba = sum(probs) / 3.0

    # ── walk-forward MoE ────────────────────────────────────────────────
    moe_proba = static_proba.copy()          # block 0 fallback
    W_all     = np.full((n, 3), 1.0 / 3)
    block     = n // N_BLOCKS

    for k in range(1, N_BLOCKS):
        lo, hi   = k * block, (k + 1) * block if k < N_BLOCKS - 1 else n
        train_hi = max(lo - EMBARGO, 0)
        gate = GatingModel(min_train=100)
        gate.fit(ctx.iloc[:train_hi], correct[:train_hi])
        W = gate.predict_weights(ctx.iloc[lo:hi])
        W_all[lo:hi] = W
        moe_proba[lo:hi] = (
            W[:, [0]] * probs[0][lo:hi]
            + W[:, [1]] * probs[1][lo:hi]
            + W[:, [2]] * probs[2][lo:hi]
        )
    moe_proba /= moe_proba.sum(axis=1, keepdims=True)

    # ── metrics ─────────────────────────────────────────────────────────
    close = feat["Close"]
    res = {
        "n_oos_rows":     n,
        "static": {
            "accuracy": float((np.argmax(static_proba, 1) == y).mean()),
            **{k: v for k, v in _simulate(static_proba, pm.index, close).items()
               if k in ("sharpe_ratio", "cagr", "max_drawdown", "calmar_ratio")},
        },
        "moe": {
            "accuracy": float((np.argmax(moe_proba, 1) == y).mean()),
            **{k: v for k, v in _simulate(moe_proba, pm.index, close).items()
               if k in ("sharpe_ratio", "cagr", "max_drawdown", "calmar_ratio")},
        },
    }

    # per-regime mean gate weights (evaluated region only)
    regime = feat["hmm_regime"].reindex(pm.index).fillna(-1).astype(int).values
    per_regime = {}
    for state, name in [(0, "bear"), (1, "ranging"), (2, "bull")]:
        mask = (regime == state) & (np.arange(n) >= block)   # skip uniform block 0
        if mask.sum() > 0:
            per_regime[name] = {
                "xgb":  round(float(W_all[mask, 0].mean()), 4),
                "lgbm": round(float(W_all[mask, 1].mean()), 4),
                "lstm": round(float(W_all[mask, 2].mean()), 4),
                "n_bars": int(mask.sum()),
            }
    res["mean_gate_weights_per_regime"] = per_regime

    # persist per-bar weights for plotting
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(W_all, index=pm.index, columns=["w_xgb", "w_lgbm", "w_lstm"]) \
        .to_parquet(OUT_DIR / f"gate_weights_{ticker}.parquet")

    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="Run the pilot retrain first.")
    args = ap.parse_args()

    if args.train:
        pilot_train()

    report: dict = {}
    for ticker in TICKERS:
        log.info("Evaluating gate on %s …", ticker)
        report[ticker] = evaluate_ticker(ticker)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "moe_comparison.json").write_text(json.dumps(report, indent=2))

    print("\n" + "═" * 78)
    print("  MoE gate vs static soft-vote  (walk-forward, no leakage)")
    print("═" * 78)
    print(f"  {'':14} {'acc':>7} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>8}")
    for tkr, r in report.items():
        for mode in ("static", "moe"):
            m = r[mode]
            print(f"  {tkr + ' ' + mode:<14} {m['accuracy']*100:>6.2f}% "
                  f"{m['sharpe_ratio']:>8.3f} {m['cagr']*100:>7.2f}% "
                  f"{m['max_drawdown']*100:>7.2f}% {m['calmar_ratio']:>8.2f}")
        print()
    print(f"  Saved → {OUT_DIR / 'moe_comparison.json'}")


if __name__ == "__main__":
    main()
