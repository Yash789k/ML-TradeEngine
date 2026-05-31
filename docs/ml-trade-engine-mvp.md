# ML Trade Engine — MVP Blueprint

A production-grade, Python-native algorithmic trading system using ML ensembles, multi-strategy signals, rigorous backtesting, a full risk management layer, and a systematic quantitative research programme targeting academic publication. No Pine Script. No TradingView lock-in.

**5 Phases Complete · 4 Research Phases In Progress · 3 ML Models (Ensemble) · 30+ Engineered Features · 1000× Monte Carlo Paths · 20–30 Ticker Universe**

---

## System Architecture

```
[01 Data] → [02 Features] → [03 ML Model] → [04 Backtest] → [05 Risk]
 Parquet      pandas-ta       XGB+LGBM        Vectorized      Kelly+ATR
 store        + HMM           + LSTM          Simulator       + VaR/CVaR

         ↓ Research Branch (Phase 06 — in progress)

[06A Strategy Zoo] → [06B Ranking] → [06C Env Analysis] → [06D Paper]
 8 classic strats     Unified          HMM regime × cost    "Regime-Gated
 on 20–30 tickers     scorecard        sensitivity           Alpha Trends"

[06E Live Execution]  ← runs in parallel with 06A–06D
 Alpaca paper trading
```

---

## Build Phases

### Phase 01 — Data Layer `[1–2 weeks]` · Foundation

> Ingest multi-asset OHLCV + macro data into a clean, versioned local store.

- Pull daily/hourly OHLCV via yfinance (equities) + ccxt (crypto)
- Fetch FRED macro data (VIX, yield curve, CPI) via pandas-datareader
- Persist to Parquet partitioned by asset/date for fast column reads
- Build a DataLoader class with caching, gap-fill, and adjust-for-splits logic
- Unit-test data pipeline with pytest — assert no lookahead leaks

**Libraries:** `yfinance` `ccxt` `pandas-datareader` `pyarrow` `pytest`

---

### Phase 02 — Feature Engineering `[1 week]` · Signals

> Generate 30+ TA + statistical features; select the most predictive subset.

- Compute RSI, MACD, Bollinger Bands, ATR, ADX, OBV via pandas-ta
- Add cross-asset features: BTC dominance, SPY correlation, sector ETF relative strength
- Statistical features: rolling z-score, Hurst exponent, realized volatility
- Regime features: HMM-detected market state (bull/bear/ranging) via hmmlearn
- Feature importance via XGBoost SHAP values; drop low-importance cols
- Store feature matrix as Parquet alongside raw OHLCV

**Libraries:** `pandas-ta` `hmmlearn` `shap` `scipy` `numpy`

---

### Phase 03 — ML Modeling `[2 weeks]` · Alpha

> Train a 3-class directional classifier (up/down/flat) with proper temporal CV.

- Label generation: forward return over N bars, threshold into 3 classes
- Walk-forward cross-validation (PurgedGroupTimeSeriesSplit) — no leakage
- Model 1 — XGBoost classifier; tune via Optuna (100+ trials, VectorBT speed)
- Model 2 — LightGBM for ensemble diversity
- Model 3 — LSTM (PyTorch) on 60-bar sequences for regime-aware signals
- Ensemble: soft-vote probability average across all 3 models
- Track MLflow experiments: params, metrics, artifacts

**Libraries:** `xgboost` `lightgbm` `torch` `optuna` `mlflow` `scikit-learn`

---

### Phase 04 — Backtesting `[1–2 weeks]` · Validation

> Simulate ML signals on 5+ years of out-of-sample data; validate edge is real.

- VectorBT: vectorized portfolio simulation; sweep 10k+ param combos in seconds
- PyBroker: ML-native backtest with walk-forward validation in one framework
- Compute: Sharpe ratio, Sortino ratio, max drawdown, CAGR, Calmar ratio, win rate
- Monte Carlo simulation: 1000 bootstrap paths to get confidence intervals on metrics
- Transaction cost model: realistic slippage + commission per asset class
- Benchmark comparison: Buy & hold SPY, 60/40, trend-following CTA index

**Libraries:** `vectorbt` `pybroker` `quantstats` `numpy`

---

### Phase 05 — Risk Management `[1 week]` · Guard Rails

> Size positions intelligently; hard-stop runaway losses before they compound.

- Kelly Criterion fractional sizing (half-Kelly) per signal confidence score
- ATR-based stop-loss: 2× ATR trailing stop per position
- Portfolio heat limit: max 20% of capital at risk simultaneously
- Max drawdown circuit breaker: pause trading if equity drops >15% from peak
- Correlation filter: block new positions that increase portfolio correlation above 0.7
- VaR / CVaR at 95% confidence computed daily via historical simulation

**Libraries:** `numpy` `scipy` `vectorbt`

---

### Phase 06A — Strategy Zoo `[2–3 weeks]` · Research

> Implement and benchmark 8 classic quant strategies on an expanded 20–30 ticker universe.

Each strategy is self-contained: own signal generation, position sizing, and full OOS backtest. The expanded universe (S&P500 large-caps, sector ETFs, crypto) is required for statistical credibility in publication.

| Strategy | Signal Logic | Regime Hypothesis |
|----------|-------------|-------------------|
| Momentum (12-1) | 12-month return minus most recent month; long top quintile | Trend-following regime |
| Mean Reversion | Bollinger Band squeeze + RSI extremes | High-vol ranging regime |
| EMA Crossover | Fast/slow EMA crossover + ADX filter | Trending / low-noise regime |
| Turtle Trading | Donchian channel breakout, ATR-sized units | Breakout regime |
| Pairs / Stat Arb | Cointegrated pair z-score entry/exit | Low-correlation regime |
| Carry Proxy | Yield spread as equity carry signal | Rate environment dependent |
| Volatility Breakout | ATR expansion + volume surge | Volatility compression → expansion |
| Alpha Trends | To be defined from literature + original research | TBD |

**Libraries:** `statsmodels` `scipy` `numpy` `pandas`

---

### Phase 06B — Strategy Ranking System `[1 week]` · Research

> Score all strategies on a unified multi-metric scorecard; compare against Phase 03–05 ML ensemble.

- CAGR, Sharpe, Sortino, Max Drawdown, Calmar
- Alpha (vs SPY via Jensen's alpha), Beta, t-statistic on mean daily return
- Information Ratio (active return / tracking error)
- Regime breakdown — all metrics decomposed per HMM state (bull / bear / ranging)
- Output: ranked table + visualisations for the paper

**Libraries:** `statsmodels` `scipy` `pandas`

---

### Phase 06C — Environment Characterisation `[1–2 weeks]` · Research

> Understand the algorithmic trading environment the engine operates in.

- **Regime analysis** — per-strategy Sharpe and CAGR by HMM state
- **Transaction cost sensitivity** — edge survival across commission/slippage levels and AUM scales
- **Signal stability** — rolling 90-day Sharpe and Alpha windows; detect strategy decay
- **Factor attribution** — Fama-French 3-factor decomposition; isolate true alpha
- **Cross-strategy correlation** — identify diversifying pairs for portfolio construction

**Libraries:** `statsmodels` `pandas-datareader` `scipy` `numpy`

---

### Phase 06D — Research Paper `[2–3 weeks]` · Publication

> Synthesise 06A–06C into a publishable academic paper.

**Title:** *Regime-Gated Alpha Trends: A Unified Framework for Strategy Selection Under Non-Stationary Market States*

**Core thesis:** The HMM regime classifier (Phase 02) acts as a meta-strategy gating layer — dynamically allocating to the strategy with the best historical edge in the current regime, sized by Kelly Criterion. This outperforms any single strategy and the passive benchmark on a risk-adjusted basis.

**Outline:**
1. Introduction — motivation, research question, contributions
2. Related work — HMM in finance, regime-switching models, strategy selection literature
3. Data and methodology — universe construction, feature pipeline, HMM calibration, strategy implementations
4. Strategy Zoo results — individual scorecards, regime decomposition
5. Regime-Gated Meta-Strategy — allocation mechanism, Kelly sizing, backtested performance
6. Environment characterisation — cost sensitivity, signal decay, factor attribution
7. Discussion — practical limitations, overfitting risks, generalisability
8. Conclusion — key findings, future directions

**Target:** Academic journal (Quantitative Finance, Journal of Portfolio Management, or similar)

---

### Phase 06E — Live Signal Engine `[1 week]` · Production *(parallel)*

> Run the trained model daily; push signals to paper trading via Alpaca. Runs in parallel with 06A–06D to generate live validation data for the paper.

- Cron job (GitHub Actions or Cloud Scheduler) fires daily at market close
- Fetch last N bars → run feature pipeline → model inference → signal output
- Alpaca paper trading API: submit bracket orders with ATR-based stop + limit
- Signal log stored in SQLite (prod) or Supabase (cloud)
- Alert via email/Slack webhook when signal fires or circuit breaker trips

**Libraries:** `alpaca-trade-api` `schedule` `sqlite3` `supabase-py` `requests`

---

### Phase 07 — Dashboard `[1 week]` · Deliverable

> Interactive Streamlit app showing model health, equity curve, live signals, and strategy ranking.

- Equity curve chart (Plotly): strategy vs benchmark overlaid
- Rolling Sharpe + max drawdown rolling window chart
- Live signal table: asset, direction, confidence score, position size, stop level
- Strategy ranking table from Phase 06B
- Model feature importance bar chart (SHAP values)
- Trade log with P&L per trade, win/loss streak
- Deploy to Streamlit Community Cloud or GCP Cloud Run (containerized)   

**Libraries:** `streamlit` `plotly` `pandas` `docker` `gcp`

---

## Quantitative Strategies

All 8 strategies are benchmarked independently in Phase 06A. The HMM regime classifier then acts as a meta-gating layer in Phase 06D, allocating to the highest-edge strategy per state.

| Strategy | Signal Logic | Timeframe | Regime Hypothesis |
|---|---|:---:|---|
| Momentum (12-1) | 12-month return minus recent month; rank top quintile | Monthly | Trend-following regime |
| Mean Reversion | Bollinger Band squeeze + RSI extremes | Daily | High-vol ranging regime |
| EMA Crossover | Fast/slow EMA + ADX filter | Daily | Trending / low-noise regime |
| Turtle Trading | Donchian channel breakout, ATR units | Daily | Breakout regime |
| Pairs / Stat Arb | Cointegrated pair z-score entry/exit | Daily | Low-correlation regime |
| Carry Proxy | Yield spread as equity carry signal | Weekly | Rate environment |
| Volatility Breakout | ATR expansion + volume surge | Daily | Vol compression → expansion |
| Alpha Trends | TBD from literature + original research | TBD | TBD |

---

## Risk Management Framework

### Circuit Breakers (Hard Limits)
- Max drawdown limit: 15% from equity peak
- Max portfolio heat: 20% capital at risk simultaneously
- Correlation cap: block trades raising portfolio correlation above 0.7

### Position Sizing
- Half-Kelly on each signal's predicted probability
- ATR-based stop: 2× ATR trailing per position
- Max single position: 5% of portfolio NAV

### Risk Reporting
- Daily VaR (95%) via historical simulation
- CVaR / Expected Shortfall at 99%
- Sharpe, Sortino, Calmar — rolling 90-day window

---

## Full Technology Stack

| Layer | Libraries / Tools | Purpose |
|---|---|---|
| Data Ingestion | yfinance, ccxt, pandas-datareader | OHLCV equity + crypto + macro |
| Feature Store | pandas-ta, scipy, hmmlearn | TA indicators, stats, HMM regime |
| ML Framework | XGBoost, LightGBM, PyTorch, scikit-learn | Tree ensembles + deep learning |
| Hyperparameter Tuning | Optuna | Bayesian search, 100+ trial sweeps |
| Experiment Tracking | MLflow | Params, metrics, model artifacts, registry |
| Backtesting | VectorBT, PyBroker | Vectorized speed + ML-native walk-forward |
| Performance Analytics | QuantStats, numpy | Sharpe, Sortino, max drawdown, CAGR |
| Risk Engine | scipy, numpy | VaR, CVaR, Kelly sizing, correlation filter |
| Live Execution | alpaca-trade-api | Paper trading API, bracket orders |
| Scheduling | GitHub Actions / GCP Cloud Scheduler | Daily cron, signal generation |
| Storage | Parquet + SQLite / Supabase | Feature store + trade log |
| Dashboard | Streamlit + Plotly | Equity curve, signals, SHAP explainability |
| Deployment | Docker + GCP Cloud Run / Streamlit Cloud | Containerized, scalable |
| Interpretability | SHAP | Feature attribution, model transparency |
| Testing | pytest | Data pipeline, feature, signal validation |

---

## Performance Targets (Out-of-Sample)

Minimum thresholds before live paper deployment.

| Metric | Target | Current Status |
|---|:---:|---|
| Sharpe Ratio | > 1.0 | ⏳ Phase 06 portfolio required |
| Max Drawdown | < 20% | ⚠️ AAPL ✅, others wider |
| Win Rate | > 55% | ✅ 61–64% on AAPL, MSFT, GOOGL, QQQ |
| CAGR | > 15% | ⏳ 7–10% single-asset |
| Calmar Ratio | > 0.5 | ✅ AAPL: 1.11 |
| t-stat on returns | > 2.0 | ⏳ Phase 06B |

---

## Roadmap

### Completed

| Phase | Milestone | Key Deliverable |
|:---:|---|---|
| 01 | Data Layer | Parquet store, 5 tickers, 5 years, FRED macro |
| 02 | Feature Engineering | 30+ features, HMM regime, SHAP selection |
| 03 | ML Models | XGB + LGBM + LSTM ensemble, walk-forward CV, MLflow |
| 04 | Backtesting | Vectorized OOS sim, 1000× Monte Carlo, threshold sweep |
| 05 | Risk Management | Kelly filter, ATR hard stop, circuit breaker, VaR/CVaR |

### In Progress

| Phase | Milestone | Key Deliverable | Gate to Proceed |
|:---:|---|---|---|
| 06A | Strategy Zoo | 8 strategies × 20–30 tickers, full OOS scorecards | Each strategy has clean, reproducible backtest |
| 06B | Ranking System | Unified scorecard table: Sharpe, Alpha, IR, t-stat, regime breakdown | All strategies statistically comparable |
| 06C | Env Characterisation | Cost sensitivity, signal decay, factor attribution | Factor α isolated from market β |
| 06D | Research Paper | Draft: Regime-Gated Alpha Trends | Results reproducible; novelty confirmed |
| 06E | Live Execution | Alpaca paper orders daily *(parallel)* | Orders execute cleanly; logs clean |
| 07 | Dashboard | Streamlit app showing equity, signals, rankings | Public URL, all charts functional |

---

## Post-MVP Extensions

### Strategy Expansion
- Options flow sentiment via unusual-whales API
- NLP sentiment from earnings call transcripts (FinBERT)
- Order book imbalance features from Level 2 data
- Reinforcement learning agent (Stable-Baselines3) for dynamic allocation
- Alpha Trends — original strategy derived from Phase 06 research findings

### Infrastructure Upgrades
- Migrate to NautilusTrader for institutional-grade event-driven execution
- Real-time feature streaming via Kafka + Faust
- Feature store on Feast (GCP) for low-latency serving
- A/B test live models with shadow deployment framework

---

## Full Technology Stack

| Layer | Libraries / Tools | Purpose |
|---|---|---|
| Data Ingestion | yfinance, ccxt, pandas-datareader | OHLCV equity + crypto + macro |
| Feature Store | pandas-ta, scipy, hmmlearn | TA indicators, stats, HMM regime |
| ML Framework | XGBoost, LightGBM, PyTorch, scikit-learn | Tree ensembles + deep learning |
| Hyperparameter Tuning | Optuna | Bayesian search |
| Experiment Tracking | MLflow | Params, metrics, model artifacts |
| Backtesting | Custom NumPy/Pandas simulator | Vectorized OOS simulation |
| Performance Analytics | QuantStats, numpy | Sharpe, Sortino, Max DD, CAGR |
| Risk Engine | scipy, numpy | VaR, CVaR, Kelly sizing |
| Research / Stats | statsmodels | Factor attribution, t-stats, OLS |
| Live Execution | alpaca-trade-api | Paper trading API, bracket orders |
| Scheduling | GitHub Actions / GCP Cloud Scheduler | Daily cron, signal generation |
| Storage | Parquet + SQLite / Supabase | Feature store + trade log |
| Dashboard | Streamlit + Plotly | Equity curve, signals, rankings |
| Deployment | Docker + GCP Cloud Run | Containerized, scalable |
| Interpretability | SHAP | Feature attribution |
| Testing | pytest | Data, feature, signal validation |

---

*ML Trade Engine — built on pandas-ta + XGBoost/LightGBM/PyTorch + custom simulator + Streamlit. No TradingView. No Pine Script. Full Python control. Research target: academic publication.*
