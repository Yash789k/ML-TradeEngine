"""Sentiment: what it is connected to, and what it is not.

Fact established by this stage: **no recorded decision used sentiment.** The pilot
feature set (``runtime_v2.model.features``) has 17 price and volume columns and
no news input, and every recorded decision reproduces exactly from price bars
alone. Sentiment exists in the repository as a separate FinBERT pipeline
(``src/features/sentiment.py``) that was never wired into the pilot.

So this stage is a diagnostic join, in three parts:

1. ``decision dates``  FinBERT tone for each instrument at the five recorded
                       decision dates, from headlines published before each
                       cutoff, next to what the classifier and the trend rule said.
2. ``history``         the same join for the three tickers with a cached news
                       history (AAPL, NVDA, SPY) against the saved forecasts.
3. ``ablation``        optional: does adding the sentiment columns change the
                       classifier's log loss? (NVDA and SPY, runtime settings.)

Headline text never leaves the gitignored workspace. Only daily aggregates are
written to ``data/review-v1/sentiment/``.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from review_v1 import receipts, registration, regime_report
from review_v1.paths import ADAPTIVE_RELEASE, OUT, RECONSTRUCTED, ROOT, WORK, write_json
from review_v1.registration import BLOCK, REPS, SYMBOLS
from review_v1.stats import block_effect, block_indices, row_log_loss

DIR = OUT / "sentiment"
NEWS = WORK / "news"
PILOT_WINDOW = ("2026-09-21", "2026-10-23")
HISTORY_TICKERS = ["AAPL", "NVDA", "SPY"]
# Earliest headline that can matter: the first 2024 refit trains on 1,008 sessions,
# which reach back to the end of 2019. Older cached SPY headlines are not scored.
HISTORY_SINCE = "2019-10-01"
ABLATION_TICKERS = ["NVDA", "SPY"]
COLUMNS = ["tone", "positive", "negative", "negative_5d", "headlines", "has_news"]


def credentials():
    """Alpaca keys from the gitignored .env; values are never logged or written."""
    from dotenv import dotenv_values

    values = dotenv_values(ROOT / ".env")
    return values.get("ALPACA_API_KEY") or "", values.get("ALPACA_SECRET_KEY") or ""


def pipeline(cache_dir):
    from src.features.sentiment import SentimentPipeline

    key, secret = credentials()
    return SentimentPipeline(api_key=key, secret_key=secret, cache_dir=cache_dir)


_SCORER = None


def default_scorer():
    """One FinBERT instance per process (the model is about 440 MB).

    Same model as ``src.features.sentiment.FinBERTScorer``, run in real batches.
    """
    global _SCORER
    if _SCORER is None:
        from review_v1.finbert import BatchedFinBERT

        _SCORER = BatchedFinBERT()
    return _SCORER


def score(news: pd.DataFrame, name: str, scorer=None) -> pd.DataFrame:
    """FinBERT probabilities per headline, cached by the content of the headline set."""
    news = news.reset_index(drop=True)
    digest = hashlib.sha256(pd.util.hash_pandas_object(news[["headline", "created_at"]], index=False).values.tobytes()).hexdigest()[:16]
    cache = NEWS / "scores" / f"{name}-{digest}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    values = (scorer or default_scorer()).score(news.headline.fillna("").tolist())
    out = pd.DataFrame(dict(created_at=news.created_at, available_at=news.get("available_at", news.created_at), p_pos=values[:, 0], p_neg=values[:, 1], p_neu=values[:, 2]))
    out["available_at"] = pd.to_datetime(out.available_at.fillna(out.created_at), utc=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out


def daily(scored: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per session from the headlines usable at that session's close.

    A headline belongs to the first session whose close is after the time it
    became available (the repository's own attribution rule), so a session's row
    never contains news published after its decision cutoff.
    """
    from src.features.sentiment import SentimentPipeline

    out = pd.DataFrame(0.0, index=sessions, columns=["positive", "negative", "headlines"])
    if len(scored):
        session = SentimentPipeline.attribute_to_session(scored.available_at, sessions)
        frame = scored.assign(session=session).dropna(subset=["session"])
        if len(frame):
            grouped = frame.groupby("session").agg(positive=("p_pos", "mean"), negative=("p_neg", "mean"), headlines=("p_pos", "size"))
            out = grouped.reindex(sessions).fillna(0.0)
    out["has_news"] = (out.headlines > 0).astype(float)
    out["tone"] = out.positive - out.negative
    out["negative_5d"] = out.negative.rolling(5, min_periods=1).mean()
    return out[COLUMNS]


def spearman(a, b) -> float:
    a, b = pd.Series(np.asarray(a, float)), pd.Series(np.asarray(b, float))
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.rank().corr(b.rank()))


def spearman_interval(a, b, block: int = BLOCK, reps: int = REPS) -> dict:
    """Rank correlation with a circular date-block percentile interval."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    value = spearman(a, b)
    if not np.isfinite(value) or len(a) < 4 * block:
        return dict(rho=value, ci_low=None, ci_high=None, n=int(len(a)))
    index = block_indices(len(a), block, reps)
    boot = np.array([spearman(a[i], b[i]) for i in index])
    boot = boot[np.isfinite(boot)]
    low, high = np.quantile(boot, [0.025, 0.975])
    return dict(rho=value, ci_low=float(low), ci_high=float(high), n=int(len(a)))


# --------------------------------------------------------------------------
# 1. Recorded decision dates
# --------------------------------------------------------------------------

def decision_dates(scorer=None) -> dict:
    cases = [receipts.load_decision(c) | dict(case=c) for c in receipts.registered_decisions()]
    last = cases[-1]
    frames, _ = receipts.load_snapshot(last["snapshot"], SYMBOLS, last["result"]["session"])
    sessions = frames["SPY"].index[frames["SPY"].index >= pd.Timestamp(PILOT_WINDOW[0], tz="UTC")]
    source = pipeline(NEWS / "pilot-window")
    rows, coverage = [], {}
    for symbol in SYMBOLS:
        news = source.fetch_headlines(symbol, start=PILOT_WINDOW[0], end=PILOT_WINDOW[1])
        scored = score(news, f"pilot-{symbol}", scorer)
        features = daily(scored, sessions)
        coverage[symbol] = int(len(news))
        for item in cases:
            date = pd.Timestamp(item["result"]["session"], tz="UTC")
            signal = next(s for s in item["plan"]["signals"] if s["symbol"] == symbol)
            p = signal["probabilities"]
            row = features.loc[date]
            rows.append(
                dict(
                    case_id=item["case"]["case_id"], session=item["result"]["session"], symbol=symbol,
                    headlines=int(row.headlines), tone=float(row.tone), positive=float(row.positive), negative=float(row.negative),
                    negative_5d=float(row.negative_5d), has_news=bool(row.has_news),
                    p_down=p[0], p_flat=p[1], p_up=p[2], model_score=p[2] - p[0], trend=bool(signal["trend"]),
                    target_weight=float(item["plan"]["shadow_weights"][symbol]), in_portfolio=bool(item["plan"]["shadow_weights"][symbol] > 0),
                )
            )
    table = pd.DataFrame(rows)
    by_date = []
    for session, g in table.groupby("session"):
        news = g.loc[g.has_news]
        by_date.append(
            dict(
                session=session, instruments_with_news=int(len(news)), headlines=int(g.headlines.sum()),
                spearman_tone_vs_model_score=spearman(news.tone, news.model_score),
                mean_tone_in_portfolio=float(news.loc[news.in_portfolio, "tone"].mean()) if news.in_portfolio.any() else None,
                mean_tone_not_in_portfolio=float(news.loc[~news.in_portfolio, "tone"].mean()) if (~news.in_portfolio).any() else None,
            )
        )
    return dict(table=table, by_date=by_date, headlines_fetched=coverage,
                fetched_at=pd.Timestamp.now(tz="UTC").isoformat(), window=PILOT_WINDOW)


# --------------------------------------------------------------------------
# 2. History link for the tickers with a cached news history
# --------------------------------------------------------------------------

def sealed_frame(symbol: str) -> pd.DataFrame:
    return pd.read_parquet(ADAPTIVE_RELEASE / f"market/adjusted/{symbol}.parquet")


def cached_news(symbol: str) -> pd.DataFrame:
    news = pd.read_parquet(ROOT / f"data/news/{symbol}_news.parquet")
    news["created_at"] = pd.to_datetime(news.created_at, utc=True)
    return news.loc[news.created_at >= pd.Timestamp(HISTORY_SINCE, tz="UTC")].reset_index(drop=True)


def history_link(scorer=None) -> dict:
    from runtime_v2.model import features as price_features

    spy = sealed_frame("SPY")
    rows, tables = [], []
    for symbol in HISTORY_TICKERS:
        news = cached_news(symbol)
        raw = sealed_frame(symbol)
        sentiment = daily(score(news, f"history-{symbol}", scorer), raw.index)
        predictions = pd.read_parquet(ADAPTIVE_RELEASE / f"experiment/{symbol}_predictions.parquet")
        x = price_features(raw, spy)
        frame = predictions.join(sentiment).assign(trend=((x.average_200 > 0) & (x.momentum_12_1 > 0)).reindex(predictions.index).astype(float))
        # Stop at the last cached headline: later sessions have no news by construction.
        frame = frame.loc[: news.created_at.max().normalize()]
        frame = frame.loc[(frame.label >= 0) & (frame.has_news > 0)]
        frame["model_score"] = frame.hybrid_2 - frame.hybrid_0
        row = dict(
            symbol=symbol, first=str(frame.index[0].date()), last=str(frame.index[-1].date()), sessions_with_news=int(len(frame)),
            median_headlines_per_session=float(frame.headlines.median()), mean_tone=float(frame.tone.mean()),
        )
        for name, column in [("model_score", "model_score"), ("trend_flag", "trend"), ("forward_return", "forward_return")]:
            result = spearman_interval(frame.tone.to_numpy(), frame[column].to_numpy())
            row.update({f"tone_vs_{name}_rho": result["rho"], f"tone_vs_{name}_ci_low": result["ci_low"], f"tone_vs_{name}_ci_high": result["ci_high"]})
        rows.append(row)
        keep = frame[["tone", "negative_5d", "headlines", "model_score", "trend", "forward_return", "label"]].copy()
        keep.insert(0, "symbol", symbol)
        keep.index = regime_report.naive(keep.index).strftime("%Y-%m-%d")
        tables.append(keep.rename_axis("date").reset_index())
    return dict(rows=pd.DataFrame(rows), table=pd.concat(tables, ignore_index=True))


# --------------------------------------------------------------------------
# 3. Optional ablation: price-only versus price-plus-sentiment classifier
# --------------------------------------------------------------------------

def walk_forward(x: pd.DataFrame, close: pd.Series, start: str, end) -> pd.DataFrame:
    """The sealed walk-forward schedule: refit every 21 sessions from ``start``."""
    from runtime_v2.model import AdaptiveConfig, AdaptiveModel, execution_labels

    config = AdaptiveConfig()
    y, _ = execution_labels(close, config)
    ids = np.flatnonzero((x.index >= pd.Timestamp(start, tz="UTC")) & (x.index <= end))
    rows, last_fit, model = [], -config.retrain_every, None
    for i in ids:
        if i - last_fit >= config.retrain_every:
            model, last_fit = AdaptiveModel(config).fit_at(x, close, x.index[i]), i
        p = model.predict(x.iloc[[i]])
        rows.append(dict(date=x.index[i], label=int(y.iloc[i]), **{f"hybrid_{c}": p["hybrid"][0, c] for c in range(3)}, **{f"prior_{c}": p["prior"][0, c] for c in range(3)}))
    return pd.DataFrame(rows).set_index("date")


def ablation(scorer=None) -> dict:
    from runtime_v2.model import features as price_features
    from threadpoolctl import threadpool_limits

    spy = sealed_frame("SPY")
    pieces, parity = [], {}
    with threadpool_limits(limits=1):
        for symbol in ABLATION_TICKERS:
            news = cached_news(symbol)
            raw = sealed_frame(symbol)
            end = news.created_at.max().normalize()
            sentiment = daily(score(news, f"history-{symbol}", scorer), raw.index)
            x = price_features(raw, spy)
            price = walk_forward(x, raw.Close, "2024-01-02", end)
            both = walk_forward(x.join(sentiment), raw.Close, "2024-01-02", end)
            sealed = pd.read_parquet(ADAPTIVE_RELEASE / f"experiment/{symbol}_predictions.parquet").loc[price.index]
            parity[symbol] = float(np.abs(price[[f"hybrid_{c}" for c in range(3)]].to_numpy() - sealed[[f"hybrid_{c}" for c in range(3)]].to_numpy()).max())
            keep = price.label >= 0
            y = price.label[keep].to_numpy()
            frame = pd.DataFrame(
                dict(
                    symbol=symbol,
                    price_only=row_log_loss(y, price.loc[keep, [f"hybrid_{c}" for c in range(3)]].to_numpy()),
                    with_sentiment=row_log_loss(y, both.loc[keep, [f"hybrid_{c}" for c in range(3)]].to_numpy()),
                    prior=row_log_loss(y, price.loc[keep, [f"prior_{c}" for c in range(3)]].to_numpy()),
                ),
                index=price.index[keep],
            )
            pieces.append(frame)
    data = pd.concat(pieces)
    rows = []
    for symbol, g in list(data.groupby("symbol")) + [("NVDA+SPY", data)]:
        difference = (g.with_sentiment - g.price_only).groupby(level=0).mean()
        effect = block_effect(difference.to_numpy(), BLOCK, REPS)
        rows.append(
            dict(
                symbol=symbol, sessions=effect["sessions"], n=int(len(g)),
                price_only_log_loss=float(g.price_only.mean()), with_sentiment_log_loss=float(g.with_sentiment.mean()), prior_log_loss=float(g.prior.mean()),
                difference=effect["effect"], ci_low=effect["ci_low"], ci_high=effect["ci_high"], p_two_sided=effect["p_two_sided"],
            )
        )
    return dict(
        status="run", rows=rows, parity_with_sealed_price_only=parity,
        parity_passed=bool(max(parity.values()) < 1e-12),
        reading="Negative difference would mean the sentiment columns lowered log loss. Two tickers; exploratory.",
        caveat=(
            "Cached NVDA headlines begin in November 2021, so its earliest training windows contain sessions with no "
            "news coverage (zero-filled and flagged). SPY headlines are capped at the 20,000 most recent items of the "
            "July download and end on 14 April 2026."
        ),
    )


def build(run_ablation: bool = True, scorer=None) -> dict:
    reg = registration.require()
    DIR.mkdir(parents=True, exist_ok=True)
    from runtime_v2.model import features as price_features

    columns = list(price_features(sealed_frame("SPY").tail(300), sealed_frame("SPY").tail(300)).columns)
    summary = dict(
        registration_sha256=reg["sha256"],
        provenance=RECONSTRUCTED,
        role=(
            "Diagnostic only. No recorded decision used sentiment: the pilot feature set has no news column and "
            "every recorded decision reproduces exactly from price bars alone."
        ),
        pilot_feature_columns=columns,
        pilot_features_contain_sentiment=bool(any("sent" in c or "news" in c or "tone" in c for c in columns)),
        scorer="ProsusAI/finbert, run locally on CPU",
        attribution="a headline counts for the first session whose XNYS close is after the time it became available",
    )
    if receipts.available():
        try:
            decisions = decision_dates(scorer)
            decisions["table"].to_csv(DIR / "decision_dates.csv", index=False)
            summary["decision_dates"] = dict(
                available=True, source="Alpaca News API", fetched_at=decisions["fetched_at"], window=list(decisions["window"]),
                headlines_fetched=int(sum(decisions["headlines_fetched"].values())),
                instruments_without_headlines=sorted(s for s, n in decisions["headlines_fetched"].items() if n == 0),
                by_date=decisions["by_date"],
                note=(
                    "Headlines were fetched for this review, not at decision time. Each is used only if it was "
                    "available before that session's close. Twenty-two instruments on five dates: descriptive."
                ),
            )
        except Exception as exc:  # network or credentials: say so, never invent values
            summary["decision_dates"] = dict(available=False, reason=f"{type(exc).__name__}: {exc}")
    else:
        summary["decision_dates"] = dict(available=False, reason="recorded receipts are not present on this machine")
    link = history_link(scorer)
    link["rows"].to_csv(DIR / "history_link.csv", index=False)
    link["table"].to_csv(DIR / "history_daily.csv", index=False)
    summary["history"] = dict(
        tickers=HISTORY_TICKERS, source="data/news/*.parquet (Alpaca headlines cached in July 2026)", rows=link["rows"].to_dict("records"),
        note="Sessions with at least one headline, matured labels only. Intervals are date-block percentile intervals; exploratory.",
    )
    summary["ablation"] = ablation(scorer) if run_ablation else dict(status="not run")
    write_json(DIR / "summary.json", summary)
    return summary


if __name__ == "__main__":
    out = build()
    print(out["decision_dates"])
    print(pd.DataFrame(out["history"]["rows"]).round(3).to_string())
    print(out["ablation"])
