"""
Sentiment feature SHAP importance report.

For each ticker:
  1. Load the cached Phase 02 feature matrix.
  2. Append the five FinBERT sentiment columns (Alpaca News API +
     ProsusAI/finbert via src/features/sentiment.py).
  3. Build the production 3-class labels (make_labels, horizon 5,
     threshold 0.5%).
  4. Fit the same quick XGBoost used by FeatureEngineer.select_features
     and rank ALL features by mean |SHAP|.
  5. Report where the sentiment columns land and whether they clear the
     selection threshold (top_n=20, importance >= 0.005 — identical to
     the Phase 02 SHAP pruning pass).

News-history coverage is limited by the Alpaca News API, so rows before
the first available headline are excluded from the SHAP fit (otherwise
the zero-filled prefix would dilute the columns' apparent importance).

Outputs
-------
  data/research/sentiment/shap_report.json
  data/research/sentiment/{ticker}_sentiment_features.parquet

Usage:
    python3 scripts/sentiment_shap_report.py --tickers AAPL NVDA SPY
    python3 scripts/sentiment_shap_report.py            # full universe
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

from src.features.sentiment import SENTIMENT_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("sentiment_shap")

ROOT    = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "research" / "sentiment"

_NON_FEATURES = {"Open", "High", "Low", "Close", "Adj_Close", "Volume",
                 "ticker", "label"}


def shap_ranking(feat: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per feature from a quick XGBoost fit (mirrors
    FeatureEngineer.select_features)."""
    import shap
    import xgboost as xgb
    from src.models.labels import make_labels

    labels, _ = make_labels(feat, horizon=5, threshold=0.005)
    mask = labels != -1
    cols = [c for c in feat.columns if c not in _NON_FEATURES]
    X = feat.loc[mask, cols].astype(float)
    y = labels[mask]
    keep = X.notna().all(axis=1)
    X, y = X[keep], y[keep]

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", verbosity=0,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    # (n, features, classes) or list of per-class arrays, depending on version
    arr = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)
    imp = np.abs(arr).mean(axis=tuple(i for i in range(arr.ndim) if i != 1))
    return pd.Series(imp, index=cols).sort_values(ascending=False)


def _fetch_and_score(ticker: str) -> None:
    """
    Run FinBERT scoring in an isolated subprocess.  transformers/torch and
    xgboost/shap both load their own OpenMP runtimes; keeping them in
    separate processes avoids a native-library crash observed when both
    are active in one long-lived interpreter on macOS.
    """
    import subprocess

    out_path = OUT_DIR / f"{ticker}_sentiment_features.parquet"
    if out_path.exists():
        log.info("%s: sentiment features already cached — skipping fetch/score", ticker)
        return

    worker = Path(__file__).resolve().parent / "_score_sentiment_worker.py"
    result = subprocess.run(
        [sys.executable, str(worker), ticker, "--start", "2024-01-01"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"sentiment worker failed for {ticker} (exit {result.returncode}): "
            f"{result.stderr[-2000:]}"
        )
    log.info(result.stdout.strip())


def run_ticker(ticker: str) -> dict:
    feat = pd.read_parquet(ROOT / "data" / "features" / f"{ticker}_features.parquet")
    feat.index = pd.to_datetime(feat.index, utc=True)

    _fetch_and_score(ticker)
    sent = pd.read_parquet(OUT_DIR / f"{ticker}_sentiment_features.parquet")
    sent.index = pd.to_datetime(sent.index, utc=True)
    full = feat.join(sent)

    # Restrict SHAP fit to the news-covered era
    covered = sent["has_news"].cumsum() > 0
    full = full[covered]

    ranking = shap_ranking(full)
    total   = float(ranking.sum())
    norm    = ranking / total if total > 0 else ranking

    sent_report = {}
    for col in SENTIMENT_COLS:
        rank = int(np.where(ranking.index == col)[0][0]) + 1
        sent_report[col] = {
            "rank":       rank,
            "importance": round(float(norm[col]), 5),
            "selected":   bool(rank <= 20 and norm[col] >= 0.005),
        }

    n_days = int(sent["has_news"].sum())
    return {
        "n_headlines":     int(sent["event_count"].sum()),
        "news_days":       n_days,
        "coverage":        round(n_days / max(len(sent), 1), 3),
        "n_features":      int(len(ranking)),
        "shap_fit_rows":   int(covered.sum()),
        "sentiment":       sent_report,
        "top10":           {k: round(float(v), 5) for k, v in norm.head(10).items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=None)
    args = ap.parse_args()

    tickers = args.tickers or sorted(
        p.name for p in (ROOT / "data" / "models").iterdir() if p.is_dir()
    )

    report: dict = {}
    for ticker in tickers:
        log.info("═════ %s ═════", ticker)
        try:
            report[ticker] = run_ticker(ticker)
        except Exception as exc:
            log.error("%s failed: %s", ticker, exc)
            report[ticker] = {"error": str(exc)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "shap_report.json").write_text(json.dumps(report, indent=2))

    print("\n══════ Sentiment column SHAP ranks (of all features) ══════")
    for tkr, r in report.items():
        if "error" in r:
            print(f"  {tkr:<6} ERROR: {r['error']}")
            continue
        ranks = ", ".join(f"{c}:#{d['rank']}{'*' if d['selected'] else ''}"
                          for c, d in r["sentiment"].items())
        print(f"  {tkr:<6} ({r['n_headlines']} headlines, {r['coverage']:.0%} day coverage)  {ranks}")
    print("  * = clears the Phase 02 selection gate (top-20 & importance ≥ 0.005)")
    print(f"\nSaved → {OUT_DIR / 'shap_report.json'}")


if __name__ == "__main__":
    main()
