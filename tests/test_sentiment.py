"""
Tests for the sentiment feature pipeline (src/features/sentiment.py).

FinBERT is replaced with a deterministic stub scorer and the Alpaca News
API with a mocked requests.get, so these tests need neither network nor
the 440 MB model download.

Coverage
--------
  attribute_to_session     — pre/post-close split, weekend roll-forward,
                             headlines past the last session dropped
  compute_daily_sentiment  — aggregation, zero-fill + has_news flag,
                             rolling 5d negativity, guidance regex
  fetch_headlines          — pagination, caching, missing-credentials error
  pipeline integration     — sentiment columns append without altering
                             existing feature columns
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.features.sentiment import (
    SENTIMENT_COLS,
    SentimentPipeline,
    _GUIDANCE_RE,
)


# ---------------------------------------------------------------------------
# Stub scorer — pos/neg keyed off simple words, avoids FinBERT download
# ---------------------------------------------------------------------------

class StubScorer:
    def score(self, headlines):
        out = np.zeros((len(headlines), 3))
        for i, h in enumerate(headlines):
            h = h.lower()
            if "beats" in h or "surges" in h:
                out[i] = [0.9, 0.05, 0.05]
            elif "misses" in h or "plunges" in h:
                out[i] = [0.05, 0.9, 0.05]
            else:
                out[i] = [0.1, 0.1, 0.8]
        return out


@pytest.fixture
def pipe(tmp_path):
    return SentimentPipeline(
        api_key="k", secret_key="s", cache_dir=tmp_path, scorer=StubScorer()
    )


def _trading_index(n=10, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


# ---------------------------------------------------------------------------
# Session attribution
# ---------------------------------------------------------------------------

class TestAttribution:
    def test_before_close_same_session(self):
        idx = _trading_index()
        ts = pd.Series([pd.Timestamp("2024-01-02 14:00", tz="UTC")])
        out = SentimentPipeline.attribute_to_session(ts, idx)
        assert out.iloc[0] == pd.Timestamp("2024-01-02", tz="UTC")

    def test_after_close_next_session(self):
        idx = _trading_index()
        ts = pd.Series([pd.Timestamp("2024-01-02 22:30", tz="UTC")])  # after 21:00 UTC
        out = SentimentPipeline.attribute_to_session(ts, idx)
        assert out.iloc[0] == pd.Timestamp("2024-01-03", tz="UTC")

    def test_weekend_rolls_to_monday(self):
        idx = _trading_index()          # starts Mon 2024-01-01
        ts = pd.Series([pd.Timestamp("2024-01-06 12:00", tz="UTC")])  # Saturday
        out = SentimentPipeline.attribute_to_session(ts, idx)
        assert out.iloc[0] == pd.Timestamp("2024-01-08", tz="UTC")    # Monday

    def test_after_last_session_dropped(self):
        idx = _trading_index(5)         # ends 2024-01-05
        ts = pd.Series([pd.Timestamp("2024-03-01 12:00", tz="UTC")])
        out = SentimentPipeline.attribute_to_session(ts, idx)
        assert out.isna().all()


# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------

class TestDailySentiment:
    def test_columns_and_no_nans(self, pipe):
        idx = _trading_index()
        news = pd.DataFrame({
            "headline":   ["AAPL beats estimates", "AAPL misses on revenue"],
            "created_at": [pd.Timestamp("2024-01-02 14:00", tz="UTC"),
                           pd.Timestamp("2024-01-03 14:00", tz="UTC")],
        })
        out = pipe.compute_daily_sentiment(news, idx)
        assert list(out.columns) == SENTIMENT_COLS
        assert len(out) == len(idx)
        assert not out.isna().any().any()

    def test_no_news_days_zero_filled_with_flag(self, pipe):
        idx = _trading_index()
        news = pd.DataFrame({
            "headline":   ["AAPL beats estimates"],
            "created_at": [pd.Timestamp("2024-01-02 14:00", tz="UTC")],
        })
        out = pipe.compute_daily_sentiment(news, idx)
        jan2 = out.loc[pd.Timestamp("2024-01-02", tz="UTC")]
        assert jan2["has_news"] == 1.0
        assert jan2["sent_pos_1d"] == pytest.approx(0.9)
        rest = out.drop(pd.Timestamp("2024-01-02", tz="UTC"))
        assert (rest["has_news"] == 0).all()
        assert (rest["sent_pos_1d"] == 0).all()
        assert (rest["event_count"] == 0).all()

    def test_event_count_and_mean(self, pipe):
        idx = _trading_index()
        day = pd.Timestamp("2024-01-04 13:00", tz="UTC")
        news = pd.DataFrame({
            "headline":   ["X beats", "X plunges", "X something neutral"],
            "created_at": [day, day, day],
        })
        out = pipe.compute_daily_sentiment(news, idx)
        row = out.loc[pd.Timestamp("2024-01-04", tz="UTC")]
        assert row["event_count"] == 3
        assert row["sent_pos_1d"] == pytest.approx((0.9 + 0.05 + 0.1) / 3)

    def test_neg_5d_rolls(self, pipe):
        idx = _trading_index()
        news = pd.DataFrame({
            "headline":   ["X misses badly"],
            "created_at": [pd.Timestamp("2024-01-02 14:00", tz="UTC")],
        })
        out = pipe.compute_daily_sentiment(news, idx)
        # negativity 0.9 on Jan 2 decays over the rolling window
        assert out["sent_neg_5d"].loc[pd.Timestamp("2024-01-02", tz="UTC")] == pytest.approx(0.45)
        assert out["sent_neg_5d"].loc[pd.Timestamp("2024-01-03", tz="UTC")] == pytest.approx(0.30)
        assert out["sent_neg_5d"].loc[pd.Timestamp("2024-01-09", tz="UTC")] == 0.0

    def test_empty_news_frame(self, pipe):
        idx = _trading_index()
        out = pipe.compute_daily_sentiment(
            pd.DataFrame(columns=["headline", "created_at"]), idx
        )
        assert (out["has_news"] == 0).all()
        assert not out.isna().any().any()

    def test_guidance_tag(self, pipe):
        idx = _trading_index()
        news = pd.DataFrame({
            "headline":   ["Acme raises full-year guidance"],
            "created_at": [pd.Timestamp("2024-01-02 14:00", tz="UTC")],
        })
        out = pipe.compute_daily_sentiment(news, idx)
        assert out["guidance_tag"].loc[pd.Timestamp("2024-01-02", tz="UTC")] == 1.0

    def test_guidance_regex_examples(self):
        hits = ["Q3 outlook improves", "cuts forecast", "reaffirms guidance"]
        misses = ["stock rises on earnings", "CEO steps down"]
        for h in hits:
            assert _GUIDANCE_RE.search(h), h
        for h in misses:
            assert not _GUIDANCE_RE.search(h), h


# ---------------------------------------------------------------------------
# Fetching + cache
# ---------------------------------------------------------------------------

def _fake_response(news, token=None):
    resp = mock.Mock()
    resp.status_code = 200
    resp.json.return_value = {"news": news, "next_page_token": token}
    resp.raise_for_status = mock.Mock()
    return resp


class TestFetch:
    def test_pagination_and_cache(self, pipe, tmp_path):
        pages = [
            _fake_response(
                [{"headline": "A", "created_at": "2024-01-02T14:00:00Z"}], token="t1"
            ),
            _fake_response(
                [{"headline": "B", "created_at": "2024-01-03T14:00:00Z"}], token=None
            ),
        ]
        with mock.patch("src.features.sentiment.requests.get", side_effect=pages) as g:
            df = pipe.fetch_headlines("TEST")
        assert g.call_count == 2
        assert list(df["headline"]) == ["A", "B"]
        assert (tmp_path / "TEST_news.parquet").exists()

        # second call must hit the cache, not the network
        with mock.patch("src.features.sentiment.requests.get") as g2:
            df2 = pipe.fetch_headlines("TEST")
        g2.assert_not_called()
        assert len(df2) == 2

    def test_missing_credentials_raises(self, tmp_path, monkeypatch):
        # Explicit empty credentials must NOT silently fall back to a
        # real ALPACA_API_KEY/SECRET that happens to be set in the
        # ambient environment (e.g. a developer's shell with .env sourced).
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        sp = SentimentPipeline(api_key="", secret_key="", cache_dir=tmp_path,
                               scorer=StubScorer())
        with pytest.raises(RuntimeError, match="ALPACA_API_KEY"):
            sp.fetch_headlines("TEST")


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_sentiment_columns_appended_not_modified(self, tmp_path, monkeypatch):
        """FeatureEngineer(sentiment=True) must append the five new columns
        and leave every pre-existing column bit-identical."""
        from src.features.pipeline import FeatureEngineer

        n   = 300
        rng = np.random.default_rng(0)
        idx = pd.date_range("2023-01-02", periods=n, freq="B", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "Open":   100 + rng.normal(0, 1, n).cumsum(),
                "High":   101 + rng.normal(0, 1, n).cumsum(),
                "Low":    99  + rng.normal(0, 1, n).cumsum(),
                "Close":  100 + rng.normal(0, 1, n).cumsum(),
                "Volume": rng.integers(1e6, 5e6, n).astype(float),
            },
            index=idx,
        )
        ohlcv["High"] = ohlcv[["Open", "High", "Close"]].max(axis=1)
        ohlcv["Low"]  = ohlcv[["Open", "Low",  "Close"]].min(axis=1)

        def fake_build(self, ticker, trading_index, **kw):
            out = pd.DataFrame(0.0, index=trading_index, columns=SENTIMENT_COLS)
            out["has_news"] = 1.0
            return out

        monkeypatch.setattr(
            "src.features.sentiment.SentimentPipeline.build_features", fake_build
        )

        fe_plain = FeatureEngineer(cache_dir=tmp_path / "a")
        fe_sent  = FeatureEngineer(cache_dir=tmp_path / "b", sentiment=True)
        base = fe_plain.compute_features(ohlcv, asset_slug="TESTX")
        with_s = fe_sent.compute_features(ohlcv, asset_slug="TESTX")

        assert set(with_s.columns) - set(base.columns) == set(SENTIMENT_COLS)
        shared = [c for c in base.columns if c != "hmm_regime"]  # HMM fit is stochastic
        pd.testing.assert_frame_equal(base[shared], with_s[shared])

    def test_sentiment_failure_is_soft(self, tmp_path, monkeypatch):
        """If the news fetch raises, the pipeline must still return features."""
        from src.features.pipeline import FeatureEngineer

        def boom(self, *a, **k):
            raise RuntimeError("no credentials")

        monkeypatch.setattr(
            "src.features.sentiment.SentimentPipeline.build_features", boom
        )

        n   = 300
        rng = np.random.default_rng(1)
        idx = pd.date_range("2023-01-02", periods=n, freq="B", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "Open": 100.0, "High": 101.0, "Low": 99.0,
                "Close": 100 + rng.normal(0, 1, n).cumsum(),
                "Volume": 1e6,
            },
            index=idx,
        )
        fe = FeatureEngineer(cache_dir=tmp_path, sentiment=True)
        feat = fe.compute_features(ohlcv, asset_slug="TESTY")
        assert "rsi_14" in feat.columns
        assert "sent_pos_1d" not in feat.columns
