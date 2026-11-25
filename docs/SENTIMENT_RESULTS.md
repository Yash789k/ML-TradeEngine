# Sentiment Feature Pipeline — Results

**Feature:** FinBERT news-sentiment features (deck slides 33–34) appended
to the Phase 02 feature matrix.

**Code:** `src/features/sentiment.py` (new: `SentimentPipeline`,
`FinBERTScorer`), `src/features/pipeline.py`
(`FeatureEngineer(sentiment=True)` — appends columns without touching
existing ones), `scripts/sentiment_shap_report.py` +
`scripts/_score_sentiment_worker.py` (benchmark).
Tests: `tests/test_sentiment.py` (15 tests: session attribution, daily
aggregation, fetch/cache, pipeline integration, soft-failure).

---

## 1. Design

- **Source:** Alpaca News API (`/v1beta1/news`), free with the paper
  account, paginated 50 headlines/page.
- **Scorer:** `ProsusAI/finbert` via `transformers`, batched (32/batch),
  with periodic `gc.collect()` — see §4.
- **Feature columns** (all zero-filled, never NaN):
  `sent_pos_1d`, `sent_neg_5d` (5-session rolling mean of daily
  negativity), `event_count`, `guidance_tag` (regex on
  guidance/outlook/forecast language), `has_news`.
- **Session alignment:** headlines published after 21:00 UTC (≈16:00
  US/Eastern close) attribute to the next session; weekend/holiday
  headlines roll forward to the next trading day actually present in
  the target index. Headlines beyond the last known session are
  dropped rather than guessed at.
- **Integration:** `FeatureEngineer(sentiment=True)` appends the five
  columns after HMM regime, wrapped in try/except — a fetch failure
  (missing credentials, API outage) logs a warning and returns the
  feature matrix unchanged rather than breaking the pipeline.

## 2. Benchmark protocol

Pulled real headlines (2024-01-01 → present, Alpaca News API) for AAPL,
NVDA, SPY; scored all of them with FinBERT; joined onto the cached
Phase 02 feature matrix; fit the same quick XGBoost + `TreeExplainer`
SHAP pass used by `FeatureEngineer.select_features` (top-20,
importance ≥ 0.005 gate) restricted to the news-covered era so the
zero-filled pre-history doesn't dilute the sentiment columns'
apparent importance. Script: `scripts/sentiment_shap_report.py`.

## 3. Results

| Ticker | Headlines | News-day coverage | `sent_pos_1d` | `sent_neg_5d` | `event_count` | `guidance_tag` | `has_news` |
|---|---:|---:|---|---|---|---|---|
| AAPL | 8,253 | 48% | rank 5 ✅ | rank 12 ✅ | rank 30 | rank 44 | rank 45 |
| NVDA | 11,269 | 24% | rank 40 | rank 10 ✅ | rank 38 | rank 43 | rank 45 |
| SPY | 20,000 | 46% | rank 17 ✅ | rank 11 ✅ | rank 38 | rank 39 | rank 45 |

✅ = clears the Phase 02 selection gate (top-20 of 45 features, mean
|SHAP| ≥ 0.005). Full ranking + per-feature scores:
`data/research/sentiment/shap_report.json`.

**`sent_neg_5d` clears the gate on all three tickers** — the smoothed
negative-sentiment signal is a consistently useful feature, landing in
the top 12 of 45 alongside technical/macro features like `adx_14`,
`realized_vol_5`, and `macro_VIX`. `sent_pos_1d` clears the gate on
AAPL (rank 5, ahead of `bb_lower`/`macro_VIX`) and SPY (rank 17) but
not NVDA (rank 40) — plausibly because NVDA's news-day coverage is
lower (24% vs 46–48%) and dominated by high-volume earnings-cycle
noise. `event_count`, `guidance_tag`, and `has_news` never clear the
gate — headline *volume* and the binary guidance flag carry little
independent signal once the sentiment *polarity* columns are present.

## 4. Engineering note: FinBERT + XGBoost/SHAP process isolation

Running `transformers`/`torch` (FinBERT) and `xgboost`/`shap` in the
same long-lived Python process reproducibly segfaulted (exit 139)
partway through scoring a large headline batch on this machine (both
libraries load their own native OpenMP runtime). The fix:
`scripts/_score_sentiment_worker.py` runs the fetch+FinBERT-scoring
step in its own subprocess per ticker and writes a parquet cache;
`sentiment_shap_report.py` only ever imports `xgboost`/`shap` in the
parent process. `FinBERTScorer.score()` also runs a periodic
`gc.collect()` every 32 batches as a defensive measure. This isolation
is cheap (one subprocess per ticker, cached thereafter) and is the
recommended pattern for any future caller that needs both stacks.

## 5. Limitations (cite in FYP §7.1)

- **Coverage is not 100%.** 24–48% of trading days have at least one
  headline for these three tickers; the rest are zero-filled
  (`has_news=0`). Coverage will be lower for less-covered names in the
  22-ticker universe (ETFs like `XLE`/`XLP` in particular).
- **History is truncated to 2024+.** The Alpaca News API's practical
  pagination limits (~20k headlines/query) made a full 5-year pull
  (2021–2026) incomplete/inconsistent for high-volume tickers; the
  benchmark instead uses a shorter but *contiguous* 2024→present
  window. Backtesting the sentiment features over the full Phase 06D
  sample would require either a paid news tier or multiple date-range
  queries.
- **Not yet wired into training.** The SHAP evidence supports adding
  `sent_pos_1d`/`sent_neg_5d` to the production feature set, but the
  ML ensemble (Phase 03) has not been retrained with them — that is
  the natural next step now that the pipeline and evidence exist.
- **FinBERT is a general financial-news model**, not fine-tuned on
  this dataset's headlines; sentiment quality is only as good as
  ProsusAI's pretraining corpus.

## 6. Reproduce

```bash
# .env must contain ALPACA_API_KEY / ALPACA_SECRET_KEY
python3 scripts/sentiment_shap_report.py --tickers AAPL NVDA SPY
python3 -m pytest tests/test_sentiment.py -q
```
