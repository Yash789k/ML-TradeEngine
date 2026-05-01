# ML Trade Engine — MVP Blueprint

A production-grade, Python-native algorithmic trading system using ML ensembles, multi-strategy signals, rigorous backtesting, and a live paper trading pipeline. No Pine Script. No TradingView lock-in.

**7 Build Phases · 8 Quant Strategies · 3 ML Models (Ensemble) · 30+ Engineered Features · 1000× Monte Carlo Paths · 10–12 Week Timeline**

---

## System Architecture

Seven sequential layers — each layer's output feeds the next. Build and validate one layer before moving on.

```
[01 Data] → [02 Features] → [03 ML Model] → [04 Backtest] → [05 Risk] → [06 Execution] → [07 Dashboard]
 Parquet      pandas-ta       XGB+LGBM        VectorBT        Kelly+ATR    Alpaca paper     Streamlit
 store        + HMM           + LSTM          + PyBroker      + VaR        API              + Plotly
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

### Phase 06 — Live Signal Engine `[1 week]` · Production

> Run the trained model daily; push signals to paper trading via Alpaca.

- Cron job (GitHub Actions or Cloud Scheduler) fires daily at market close
- Fetch last N bars → run feature pipeline → model inference → signal output
- Alpaca paper trading API: submit bracket orders with ATR-based stop + limit
- Signal log stored in SQLite (prod) or Supabase (cloud)
- Alert via email/Slack webhook when signal fires or circuit breaker trips

**Libraries:** `alpaca-trade-api` `schedule` `sqlite3` `supabase-py` `requests`

---

### Phase 07 — Dashboard `[1 week]` · Deliverable

> Interactive Streamlit app showing model health, equity curve, and live signals.

- Equity curve chart (Plotly): strategy vs benchmark overlaid
- Rolling Sharpe + max drawdown rolling window chart
- Live signal table: asset, direction, confidence score, position size, stop level
- Model feature importance bar chart (SHAP values)
- Trade log with P&L per trade, win/loss streak
- Deploy to Streamlit Community Cloud or GCP Cloud Run (containerized)

**Libraries:** `streamlit` `plotly` `pandas` `docker` `gcp`

---

## Quantitative Strategies

All strategies feed into the ensemble — the model learns which signals are regime-appropriate.

| Strategy | Signal Logic | Timeframe | ML Layer |
|---|---|:---:|---|
| Trend Following | EMA crossover + ADX filter | Daily | XGBoost classifier |
| Mean Reversion | Bollinger Band squeeze + RSI extremes | Hourly | LightGBM |
| Momentum | Cross-sectional 12-1 month momentum, sector rotation | Weekly | Rank-based scoring |
| Regime-Aware | HMM market state gates all other signals | Daily | hmmlearn + ensemble |
| Volatility Breakout | ATR expansion + volume surge entry | Daily | XGBoost + threshold |
| Macro Factor | Yield curve, VIX term structure, dollar index as features | Weekly | LightGBM |
| Statistical Arb (pairs) | Cointegrated pair z-score entry/exit (crypto) | Hourly | OLS residuals + rule |
| LSTM Sequence | 60-bar temporal pattern recognition on OHLCV+vol | Daily | PyTorch LSTM |

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

Minimum thresholds before considering live paper deployment.

| Metric | Target |
|---|:---:|
| Sharpe Ratio | > 1.0 |
| Max Drawdown | < 20% |
| Win Rate | > 55% |
| Annual Return (CAGR) | > 15% |
| Calmar Ratio | > 0.5 |

---

## 10-Week MVP Roadmap

| Week | Milestone | Key Deliverable | Gate to Proceed |
|:---:|---|---|---|
| 1–2 | Data Layer complete | Parquet store for 5 assets, 5 years | Zero NaN rows, split-adjusted |
| 3 | Feature Engineering complete | 30+ feature matrix per asset | No lookahead bias in any feature |
| 4–5 | ML Models trained | XGB + LGBM + LSTM with MLflow runs | Val accuracy > random baseline |
| 6–7 | Backtest validated | VectorBT + PyBroker equity curves | Sharpe > 1.0 out-of-sample |
| 8 | Risk engine integrated | Kelly sizing + circuit breakers live | Max drawdown < 20% in sim |
| 9 | Live signal engine running | Alpaca paper orders daily | Orders execute, logs clean |
| 10 | Dashboard deployed | Streamlit app on Cloud Run | Public URL, all charts functional |

---

## Post-MVP Extensions

### Strategy Expansion
- Options flow sentiment via unusual-whales API
- NLP sentiment from earnings call transcripts (FinBERT)
- Order book imbalance features from Level 2 data
- Reinforcement learning agent (Stable-Baselines3) for dynamic allocation

### Infrastructure Upgrades
- Migrate to NautilusTrader for institutional-grade event-driven execution
- Real-time feature streaming via Kafka + Faust
- Feature store on Feast (GCP) for low-latency serving
- A/B test live models with shadow deployment framework

---

*ML Trade Engine MVP — built on pandas-ta + XGBoost/LightGBM/PyTorch + VectorBT/PyBroker + Streamlit. No TradingView. No Pine Script. Full Python control.*
