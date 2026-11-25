"""
Subprocess worker: fetch + score sentiment for ONE ticker, then exit.

Isolating this step in its own process keeps the transformers/torch
runtime out of the same process as xgboost/shap (which load their own
native OpenMP runtimes) — mixing both in one long-lived process has been
observed to crash on macOS. The parent script (sentiment_shap_report.py)
shells out to this worker per ticker and reads the parquet it writes.

Usage:
    python3 scripts/_score_sentiment_worker.py AAPL --start 2024-01-01
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT    = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "research" / "sentiment"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--max-pages", type=int, default=400)
    args = ap.parse_args()

    import pandas as pd
    from src.features.sentiment import SentimentPipeline

    feat = pd.read_parquet(ROOT / "data" / "features" / f"{args.ticker}_features.parquet")
    feat.index = pd.to_datetime(feat.index, utc=True)

    sp   = SentimentPipeline()
    news = sp.fetch_headlines(args.ticker, start=args.start, max_pages=args.max_pages)
    sent = sp.compute_daily_sentiment(news, trading_index=feat.index)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sent.to_parquet(OUT_DIR / f"{args.ticker}_sentiment_features.parquet")

    n_days = int(sent["has_news"].sum())
    print(f"{args.ticker}: {len(news)} headlines, {n_days} news-days "
          f"({n_days/max(len(sent),1):.1%} coverage)")


if __name__ == "__main__":
    main()
