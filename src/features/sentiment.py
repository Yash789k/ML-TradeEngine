"""
Phase 02+ — Sentiment Feature Pipeline (FinBERT)

Pulls news headlines from the Alpaca News API, scores them with FinBERT
(ProsusAI/finbert), and aggregates per trading day into feature columns
that append cleanly to the Phase 02 feature matrix (deck slides 33–34).

Feature columns
---------------
  sent_pos_1d   : mean positive-probability of headlines attributed to
                  the session (0 when no news)
  sent_neg_5d   : 5-session rolling mean of daily negative-probability
  event_count   : number of headlines attributed to the session
  guidance_tag  : 1 if any headline mentions guidance/outlook/forecast
                  language, else 0
  has_news      : 1 if the session had at least one headline, else 0
                  (lets the model separate "neutral news" from "no news")

Date alignment
--------------
Headlines are attributed to the trading session whose information they
can influence: anything published after 16:00 US/Eastern (market close)
belongs to the NEXT session, as do weekend/holiday headlines (they roll
forward to the next trading day in the target index).

Missing news days are zero-filled (neutral) and flagged has_news=0 —
never NaN, so downstream dropna() calls keep the full history.

Credentials
-----------
Uses ALPACA_API_KEY / ALPACA_SECRET_KEY environment variables (same as
Phase 06E live trading — the news endpoint is free with a paper
account).

Usage
-----
    from src.features.sentiment import SentimentPipeline

    sp   = SentimentPipeline()
    news = sp.fetch_headlines("AAPL", start="2021-01-01")
    sent = sp.compute_daily_sentiment(news, trading_index=feat_df.index)
    feat = feat_df.join(sent)          # or FeatureEngineer(sentiment=True)
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_NEWS_CACHE   = _PROJECT_ROOT / "data" / "news"

_NEWS_URL     = "https://data.alpaca.markets/v1beta1/news"
_PAGE_LIMIT   = 50          # max items per Alpaca news page
_MARKET_CLOSE_UTC_HOUR = 21  # 16:00 US/Eastern ≈ 21:00 UTC (20:00 in DST; the
                             # 1 h slack only shifts a few borderline headlines)

_GUIDANCE_RE = re.compile(
    r"\b(?:guidance|outlook|forecast|raises?\s+(?:full[- ]year|fy|q\d)|"
    r"lowers?\s+(?:full[- ]year|fy|q\d)|reaffirms?|cuts?\s+forecast)\b",
    re.IGNORECASE,
)

SENTIMENT_COLS = ["sent_pos_1d", "sent_neg_5d", "event_count", "guidance_tag", "has_news"]


# ---------------------------------------------------------------------------
# FinBERT scorer (lazy singleton — the model is ~440 MB)
# ---------------------------------------------------------------------------

class FinBERTScorer:
    """
    Thin wrapper around ProsusAI/finbert returning per-headline
    [p_positive, p_negative, p_neutral].
    """

    _MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, batch_size: int = 32) -> None:
        self.batch_size = batch_size
        self._pipe = None

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        from transformers import pipeline  # deferred: heavy import
        log.info("[FinBERT] loading %s …", self._MODEL_NAME)
        self._pipe = pipeline(
            "text-classification",
            model=self._MODEL_NAME,
            top_k=None,
            truncation=True,
            max_length=128,
            device="cpu",   # transformers ≥5 auto-selects MPS on macOS,
                            # which segfaults on long batch runs
        )

    def score(self, headlines: list[str]) -> np.ndarray:
        """
        Score a list of headlines.

        Returns (n, 3) array of [p_positive, p_negative, p_neutral].
        """
        if not headlines:
            return np.zeros((0, 3))
        self._ensure_loaded()

        import gc

        out = np.zeros((len(headlines), 3))
        n_batches = 0
        for lo in range(0, len(headlines), self.batch_size):
            batch   = headlines[lo:lo + self.batch_size]
            results = self._pipe(batch)
            for j, scores in enumerate(results):
                by_label = {s["label"].lower(): s["score"] for s in scores}
                out[lo + j] = [
                    by_label.get("positive", 0.0),
                    by_label.get("negative", 0.0),
                    by_label.get("neutral",  0.0),
                ]
            # Periodic GC: on long CPU inference runs (thousands of
            # headlines) unreleased tensor references have been observed
            # to accumulate and eventually crash the process — a cheap
            # collect every ~1k headlines keeps memory flat.
            n_batches += 1
            if n_batches % 32 == 0:
                gc.collect()
        return out


# ---------------------------------------------------------------------------
# SentimentPipeline
# ---------------------------------------------------------------------------

class SentimentPipeline:
    """
    Headlines → FinBERT scores → daily feature columns.

    Parameters
    ----------
    api_key / secret_key : Alpaca credentials; default to the
                           ALPACA_API_KEY / ALPACA_SECRET_KEY env vars
    cache_dir            : parquet cache for raw scored headlines
    scorer               : injectable FinBERTScorer (tests pass a stub)
    """

    def __init__(
        self,
        api_key:    Optional[str] = None,
        secret_key: Optional[str] = None,
        cache_dir:  Optional[Path] = None,
        scorer:     Optional[FinBERTScorer] = None,
    ) -> None:
        self.api_key    = api_key    or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self.cache_dir  = Path(cache_dir) if cache_dir else _NEWS_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.scorer     = scorer or FinBERTScorer()

    # ------------------------------------------------------------------
    # 1. Fetch headlines
    # ------------------------------------------------------------------

    def fetch_headlines(
        self,
        ticker: str,
        start:  str = "2021-01-01",
        end:    Optional[str] = None,
        force_refresh: bool = False,
        max_pages: int = 200,
    ) -> pd.DataFrame:
        """
        Pull all headlines for `ticker` from the Alpaca News API
        (paginated), caching to data/news/{ticker}_news.parquet.

        Returns DataFrame with columns: headline, created_at (UTC tz-aware).
        """
        cache = self.cache_dir / f"{ticker}_news.parquet"
        if cache.exists() and not force_refresh:
            df = pd.read_parquet(cache)
            df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
            return df

        if not self.api_key or not self.secret_key:
            raise RuntimeError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY not set — cannot fetch "
                "news and no cache exists for " + ticker
            )

        headers = {
            "APCA-API-KEY-ID":     self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }
        params: dict = {
            "symbols": ticker,
            "start":   start,
            "limit":   _PAGE_LIMIT,
            "sort":    "asc",
        }
        if end:
            params["end"] = end

        rows: list[dict] = []
        for page in range(max_pages):
            resp = requests.get(_NEWS_URL, headers=headers, params=params, timeout=30)
            if resp.status_code == 429:            # rate limited — back off once
                time.sleep(3.0)
                resp = requests.get(_NEWS_URL, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            for item in payload.get("news", []):
                rows.append({
                    "headline":   item.get("headline", ""),
                    "created_at": item.get("created_at"),
                    "available_at": item.get("updated_at") or item.get("created_at"),
                })
            token = payload.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        else:
            log.warning("%s: hit max_pages=%d — headline history truncated",
                        ticker, max_pages)

        df = pd.DataFrame(rows, columns=["headline", "created_at", "available_at"])
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        df = df.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)
        df.to_parquet(cache)
        log.info("%s: fetched %d headlines → %s", ticker, len(df), cache)
        return df

    # ------------------------------------------------------------------
    # 2. Session attribution
    # ------------------------------------------------------------------

    @staticmethod
    def attribute_to_session(
        created_at: pd.Series,
        trading_index: pd.DatetimeIndex,
    ) -> pd.Series:
        """
        Map each headline timestamp to the trading session it can
        influence:
          - published before the close → same calendar day
          - published after the close  → next calendar day
          - non-trading days           → rolled forward to the next
            session present in trading_index

        Returns a Series of normalized session dates (NaT when the
        headline falls after the last session in the index).
        """
        from src.data.validation import session_closes
        ts = pd.to_datetime(created_at, utc=True)
        sessions = pd.DatetimeIndex(trading_index).normalize().unique().sort_values()
        closes = session_closes(sessions)
        sessions, closes = sessions[closes.notna()], closes[closes.notna()]
        # At/after close is only usable in a later session; early closes and DST
        # come from XNYS rather than a fixed UTC hour.
        pos = closes.searchsorted(pd.DatetimeIndex(ts), side="right")
        out = pd.Series(pd.NaT, index=created_at.index, dtype="datetime64[ns, UTC]")
        valid = pos < len(sessions)
        out.iloc[np.flatnonzero(valid)] = sessions[pos[valid]]

        return out

    # ------------------------------------------------------------------
    # 3. Daily aggregation
    # ------------------------------------------------------------------

    def compute_daily_sentiment(
        self,
        news_df: pd.DataFrame,
        trading_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Score headlines with FinBERT and aggregate to one row per
        trading session.

        Parameters
        ----------
        news_df       : fetch_headlines() output (headline, created_at)
        trading_index : the target feature matrix index (defines sessions)

        Returns a DataFrame indexed by trading_index with the
        SENTIMENT_COLS columns — zero-filled, never NaN.
        """
        sessions = pd.DatetimeIndex(trading_index).normalize()
        out = pd.DataFrame(
            0.0, index=trading_index,
            columns=["sent_pos_1d", "sent_neg_1d", "event_count", "guidance_tag", "has_news"],
        )

        if len(news_df):
            news = news_df.copy()
            news["session"] = self.attribute_to_session(news.get("available_at", news["created_at"]), trading_index)
            news = news.dropna(subset=["session"])

            if len(news):
                scores = self.scorer.score(news["headline"].tolist())
                news["p_pos"] = scores[:, 0]
                news["p_neg"] = scores[:, 1]
                news["is_guidance"] = news["headline"].str.contains(_GUIDANCE_RE).astype(int)

                grouped = news.groupby("session").agg(
                    sent_pos_1d  = ("p_pos", "mean"),
                    sent_neg_1d  = ("p_neg", "mean"),
                    event_count  = ("headline", "size"),
                    guidance_tag = ("is_guidance", "max"),
                )
                grouped["has_news"] = 1.0

                aligned = grouped.reindex(sessions).fillna(0.0)
                aligned.index = trading_index
                out = aligned

        # 5-session rolling mean of daily negativity (0-filled ⇒ well defined)
        out["sent_neg_5d"] = out["sent_neg_1d"].rolling(5, min_periods=1).mean()
        return out[SENTIMENT_COLS].astype(float)

    # ------------------------------------------------------------------
    # 4. One-call convenience
    # ------------------------------------------------------------------

    def build_features(
        self,
        ticker: str,
        trading_index: pd.DatetimeIndex,
        start: str = "2021-01-01",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """fetch_headlines + compute_daily_sentiment for one ticker."""
        news = self.fetch_headlines(ticker, start=start, force_refresh=force_refresh)
        return self.compute_daily_sentiment(news, trading_index)
