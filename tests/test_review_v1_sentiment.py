"""Sentiment diagnostics: timing of headlines, caching, rank correlations, wording of the verdicts, FinBERT parity."""

import os

import numpy as np
import pandas as pd
import pytest

from review_v1 import audit
from review_v1 import sentiment_link as link

SESSIONS = pd.bdate_range("2026-10-01", "2026-10-08")  # Thu, Fri, Mon, Tue, Wed, Thu


def scored(times, positive, negative):
    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    return pd.DataFrame(dict(available_at=pd.to_datetime(times, utc=True), p_pos=positive, p_neg=negative, p_neu=1 - positive - negative))


def test_a_headline_counts_only_for_the_first_session_that_closes_after_it():
    frame = scored(
        [
            "2026-10-01T14:00:00Z",  # Thursday, during the session
            "2026-10-01T19:59:00Z",  # one minute before the 20:00 UTC close
            "2026-10-01T20:00:00Z",  # at the close: too late for Thursday
            "2026-10-01T20:30:00Z",  # after the close
            "2026-10-03T12:00:00Z",  # Saturday
            "2026-10-08T21:00:00Z",  # after the last session's close: unusable
        ],
        [0.8, 0.6, 0.1, 0.2, 0.5, 0.9],
        [0.1, 0.2, 0.7, 0.6, 0.25, 0.05],
    )
    out = link.daily(frame, SESSIONS)
    assert list(out.columns) == link.COLUMNS
    assert out.headlines.tolist() == [2, 2, 1, 0, 0, 0]
    assert out.loc["2026-10-01", ["positive", "negative", "tone"]].tolist() == pytest.approx([0.7, 0.15, 0.55])
    assert out.loc["2026-10-02", ["positive", "negative", "tone"]].tolist() == pytest.approx([0.15, 0.65, -0.5])
    assert out.loc["2026-10-05", ["positive", "negative", "tone"]].tolist() == pytest.approx([0.5, 0.25, 0.25])
    assert out.has_news.tolist() == [1, 1, 1, 0, 0, 0]
    # Sessions without news are zero-filled and flagged, never forward-filled.
    assert out.loc["2026-10-06":, ["positive", "negative", "tone"]].eq(0).all().all()
    assert out.negative_5d.tolist() == pytest.approx([0.15, 0.4, 0.35, 0.2625, 0.21, 0.18])


def test_later_headlines_never_change_an_earlier_session():
    early = scored(["2026-10-01T14:00:00Z", "2026-10-02T15:00:00Z"], [0.8, 0.3], [0.1, 0.5])
    late = scored(["2026-10-02T20:00:00Z", "2026-10-06T13:00:00Z", "2026-10-07T23:00:00Z"], [0.05, 0.9, 0.9], [0.9, 0.05, 0.05])
    before = link.daily(early, SESSIONS)
    after = link.daily(pd.concat([early, late], ignore_index=True), SESSIONS)
    pd.testing.assert_frame_equal(before.loc[:"2026-10-02"], after.loc[:"2026-10-02"])
    assert not before.loc["2026-10-05":].equals(after.loc["2026-10-05":])


def test_no_headlines_gives_an_explicit_empty_day_not_a_missing_one():
    out = link.daily(scored([], [], []), SESSIONS)
    assert out.shape == (len(SESSIONS), len(link.COLUMNS))
    assert out.eq(0).all().all()


def test_scores_are_cached_by_headline_content(tmp_path, monkeypatch):
    monkeypatch.setattr(link, "NEWS", tmp_path)

    class Stub:
        calls = 0

        def score(self, headlines):
            Stub.calls += 1
            return np.tile([0.2, 0.5, 0.3], (len(headlines), 1))

    news = pd.DataFrame(dict(headline=["Profit warning", "Record quarter", None], created_at=pd.to_datetime(["2026-10-01T12:00Z", "2026-10-01T13:00Z", "2026-10-02T09:00Z"])))
    first = link.score(news, "unit", Stub())
    again = link.score(news, "unit", Stub())
    assert Stub.calls == 1
    pd.testing.assert_frame_equal(first, again)
    assert first[["p_pos", "p_neg", "p_neu"]].to_numpy() == pytest.approx(np.tile([0.2, 0.5, 0.3], (3, 1)))
    assert str(first.available_at.dt.tz) == "UTC"
    # A different headline set is a different cache entry.
    link.score(news.assign(headline=["Profit warning", "Record quarter", "Guidance cut"]), "unit", Stub())
    assert Stub.calls == 2
    assert len(list((tmp_path / "scores").glob("unit-*.parquet"))) == 2


def test_rank_correlation_and_its_interval():
    x = np.arange(200, dtype=float)
    assert link.spearman(x, x**3) == pytest.approx(1.0)
    assert link.spearman(x, -x) == pytest.approx(-1.0)
    assert np.isnan(link.spearman([1.0, 2.0], [2.0, 1.0]))          # too few points
    assert np.isnan(link.spearman(x, np.ones(200)))                 # no variation
    short = link.spearman_interval(x[:20], x[:20])
    assert short["ci_low"] is None and short["ci_high"] is None and short["n"] == 20

    rng = np.random.default_rng(4)
    noise = rng.normal(size=400)
    unrelated = link.spearman_interval(noise, rng.normal(size=400))
    assert unrelated["ci_low"] < 0 < unrelated["ci_high"]
    related = link.spearman_interval(noise, noise + rng.normal(scale=0.5, size=400))
    assert related["ci_low"] > 0.5
    assert unrelated["n"] == related["n"] == 400
    # The resampling is seeded: the same inputs give the same interval.
    assert link.spearman_interval(noise, noise**3 + 0.1 * noise) == link.spearman_interval(noise, noise**3 + 0.1 * noise)
    assert link.spearman_interval(noise, noise * 2 + 1)["rho"] == pytest.approx(1.0)


def test_interval_wording_never_claims_more_than_the_interval():
    assert audit.interval_verdict(0.01, 0.05) == "worse"
    assert audit.interval_verdict(-0.05, -0.01) == "better"
    assert audit.interval_verdict(-0.01, 0.05) == "not distinguishable"
    assert audit.interval_verdict(0.01, 0.05, lower_is_better=False) == "better"
    assert audit.interval_verdict(-0.05, -0.01, lower_is_better=False) == "worse"
    assert audit.interval_verdict(None, 0.05) == "not assessed"

    def text(low, high):
        return audit.sentiment_ablation_text(dict(status="run", rows=[dict(symbol="NVDA+SPY", difference=(low + high) / 2, ci_low=low, ci_high=high)]))

    assert "the forecasts got worse" in text(0.01, 0.08)
    assert "the forecasts improved" in text(-0.08, -0.01)
    assert "no difference could be detected" in text(-0.02, 0.03)
    assert "NVDA and SPY" in text(0.01, 0.08)
    assert audit.sentiment_ablation_text(dict(status="skipped")) == "The classifier ablation with sentiment columns was not run."

    rows = [
        dict(symbol="AAA", tone_vs_model_score_rho=0.04, tone_vs_model_score_ci_low=-0.05, tone_vs_model_score_ci_high=0.12,
             tone_vs_trend_flag_rho=0.2, tone_vs_trend_flag_ci_low=0.1, tone_vs_trend_flag_ci_high=0.3,
             tone_vs_forward_return_rho=-0.01, tone_vs_forward_return_ci_low=None, tone_vs_forward_return_ci_high=None),
    ]
    score, flag, forward = audit.sentiment_link_text(rows)
    assert "AAA +0.04" in score and "no interval excludes zero" in score
    assert "interval excludes zero for AAA" in flag
    # A sample too short for an interval is reported as such, not as "no relationship" and not as a finding.
    assert "no interval could be computed for AAA" in forward
    assert "excludes zero" not in forward


def test_pilot_features_contain_no_sentiment_column():
    from runtime_v2.model import features

    rng = np.random.default_rng(0)
    index = pd.bdate_range("2023-01-02", periods=420)

    def bars():
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index))))
        return pd.DataFrame(dict(Open=close * 0.999, High=close * 1.01, Low=close * 0.99, Close=close, Volume=rng.integers(1_000_000, 2_000_000, len(index)).astype(float)), index=index)

    columns = set(features(bars(), bars()).columns)
    assert len(columns) == 17
    assert not columns & set(link.COLUMNS)
    assert not [c for c in columns if "sent" in c or "news" in c or "tone" in c or "headline" in c]


def test_batched_finbert_matches_the_repository_scorer():
    """Same model, same truncation: the batched scorer must reproduce the repository pipeline (needs the cached model)."""
    pytest.importorskip("transformers")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from review_v1.finbert import BatchedFinBERT
    from src.features.sentiment import FinBERTScorer

    headlines = [
        "Shares plunge after the company cuts its full-year guidance",
        "Record quarterly revenue beats every analyst estimate",
        "Board to meet on Tuesday",
        "",
        "Regulator opens an investigation into accounting practices at the chipmaker, sending the stock lower in early trading " * 6,
        "Dividend raised by 10%",
        "CEO resigns",
    ]
    try:
        batched = BatchedFinBERT(batch_size=3).score(headlines)
        reference = FinBERTScorer(batch_size=2).score(headlines)
    except OSError as exc:  # model files are not cached on this machine
        pytest.skip(f"FinBERT is not available offline: {type(exc).__name__}")
    assert batched.shape == reference.shape == (len(headlines), 3)
    assert np.allclose(batched.sum(axis=1), 1, atol=1e-5)
    assert np.abs(batched - reference).max() < 1e-4
    # Column order is [positive, negative, neutral]: a guidance cut reads negative, a revenue beat positive.
    assert batched[0].argmax() == 1 and batched[1].argmax() == 0
    assert BatchedFinBERT().score([]).shape == (0, 3)
