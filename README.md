# ML Trade Engine

A production-grade Python algorithmic trading system — ML ensemble signals, rigorous out-of-sample backtesting, full risk management, and a systematic quant-strategy research programme aimed at an academic publication.

**5 Phases Complete · 3 ML Models · 30+ Engineered Features · 1000× Monte Carlo Paths · 5 Tickers · Research Paper In Progress**

---

## Architecture

```
[01 Data] → [02 Features] → [03 ML Models] → [04 Backtest] → [05 Risk]
 Parquet      pandas-ta       XGB+LGBM+LSTM    Vectorized      Kelly+ATR
 store        + HMM regime    Ensemble          simulator       + VaR

        ↓ Research Branch (Phase 06 — in progress)

[06A Strategy Zoo] → [06B Ranking] → [06C Env Analysis] → [06D Paper]
 8 classic strats     CAGR/Sharpe/    HMM regime × cost    "Regime-Gated
 benchmarked OOS      Alpha/IR table  sensitivity           Alpha Trends"

[06E Live Execution]  ← runs in parallel
 Alpaca paper trading
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Apple Silicon:** PyTorch uses MPS automatically. No CUDA required.

### 2. Fetch data

```bash
python3 data_retrieval.py
```

Downloads 5 years of OHLCV for AAPL, MSFT, GOOGL, SPY, QQQ plus FRED macro data (VIX, yield spread, CPI). Caches to `data/raw/` as Parquet.

### 3. Train models

```bash
python3 train.py --all-tickers --n-trials 50 --threshold 0.01
```

Runs walk-forward cross-validation for all tickers. Saves models to `data/models/{ticker}/` and OOS predictions to `data/models/{ticker}/oos_predictions.parquet`. Tracks experiments via MLflow.

> **Note:** Training takes 30–90 minutes per ticker (XGBoost + LightGBM + LSTM with Optuna tuning).

To train a single ticker with custom settings:
```bash
python3 train.py --ticker AAPL --n-trials 100 --horizon 5 --threshold 0.01
```

### 4. Run backtest

```bash
python3 backtest.py --all-tickers
```

Uses the saved OOS predictions for a fully honest (look-ahead-free) backtest. Outputs equity curves, trade logs, Monte Carlo paths, and a confidence-threshold sweep to `data/backtest/{ticker}/`.

### 5. Run risk analysis

```bash
python3 risk.py --all-tickers
```

Applies Kelly filter, ATR hard stop, and circuit breaker on top of the Phase 04 signals. Reports VaR (95%) and CVaR (99%). Saves results to `data/risk/{ticker}/`.

### Run the test suite

```bash
python3 -m pytest tests/ -v
```

---

## Project Structure

```
ml-trade-engine/
├── data_retrieval.py      Phase 01 CLI — fetch + cache OHLCV and macro data
├── train.py               Phase 03 CLI — walk-forward ML training pipeline
├── backtest.py            Phase 04 CLI — OOS backtest + Monte Carlo
├── risk.py                Phase 05 CLI — risk-managed simulation + VaR
│
├── src/
│   ├── data/              DataLoader: OHLCV (yfinance + ccxt), macro (FRED)
│   ├── features/          FeatureEngineer: TA indicators, stats, HMM regime
│   ├── models/            XGBModel, LGBMModel, LSTMModel, EnsembleClassifier
│   ├── backtest/          Simulator, SignalBuilder, Metrics, MonteCarlo, Runner
│   ├── risk/              RiskEngine, Kelly sizing, ATR stops, VaR/CVaR
│   └── research/          [Phase 06] Strategy Zoo, Ranker, EnvAnalyzer (coming)
│
├── tests/                 pytest suite — data, features, ML, backtest, risk
├── docs/                  PHASE_0x_LOG.md per-phase build logs + research notes
└── requirements.txt
```

---

## What Each Phase Does

### Phase 01 — Data Layer
Fetches and caches multi-asset OHLCV and macro data:
- **Equities** via `yfinance` (daily OHLCV, split-adjusted)
- **Crypto** via `ccxt` (Kraken — used as a cross-asset feature proxy)
- **Macro** via `pandas-datareader` + FRED: VIX, 10Y-2Y yield spread, CPI
- Parquet storage with gap-filling and stale-cache detection

### Phase 02 — Feature Engineering
Generates 30+ features per asset, cached to `data/features/`:

| Group | Features |
|-------|----------|
| Technical | RSI, MACD, Bollinger Bands, ATR, ADX, OBV, EMA ratio, log returns |
| Statistical | Rolling z-score, Hurst exponent (R/S), realized vol, skew/kurtosis |
| Cross-asset | SPY correlation, relative strength, BTC return proxy |
| Macro | VIX, yield spread, CPI (lagged, merged) |
| Regime | 3-state HMM market state (bull / bear / ranging) via `hmmlearn` |

Feature selection via XGBoost SHAP values — drops low-importance columns automatically.

### Phase 03 — ML Modeling
Trains a 3-class directional classifier (UP / FLAT / DOWN) on each ticker:

- **Labels:** forward 5-day return thresholded at ±1% → UP / FLAT / DOWN
- **CV:** Walk-forward split (PurgedGroupTimeSeriesSplit, embargo=5 days) — no data leakage
- **XGBoost:** Bayesian tuning via Optuna (50–100 trials), `RobustScaler`
- **LightGBM:** Same setup, provides ensemble diversity
- **LSTM:** PyTorch, 60-bar sequences, 2-layer unidirectional, MPS / CPU auto-detect
- **Ensemble:** Soft-vote probability average, weights ∝ OOF F1-macro score
- **Tracking:** MLflow logs all params, metrics, and model artifacts
- **Output:** `oos_predictions.parquet` — honest OOS probabilities for backtesting

### Phase 04 — Backtesting
Vectorized simulation on OOS predictions (no look-ahead bias):

- **Signals:** Confidence-filtered (`p_up > 0.38` default) UP / FLAT / DOWN
- **Simulator:** NumPy/Pandas vectorized engine — commission + slippage model
- **Metrics:** Sharpe, Sortino, Max Drawdown, CAGR, Calmar, win rate
- **Benchmark:** Buy-and-hold comparison
- **Monte Carlo:** 1000 block-bootstrap paths, 3-year horizon, confidence intervals
- **Sweep:** Grid over confidence thresholds and commission rates

**Phase 04 OOS results (5 years, 10K capital, long-only):**

| Ticker | Sharpe | CAGR   | Max DD | Win Rate |
|--------|--------|--------|--------|----------|
| AAPL   | +0.374 | +9.7%  | -8.8%  | 64%      |
| MSFT   | +0.084 | +5.0%  | -17.7% | 53%      |
| GOOGL  | +0.257 | +7.9%  | -20.2% | 63%      |
| SPY    | -0.645 | -1.3%  | -12.4% | 63%      |
| QQQ    | +0.202 | +6.9%  | -14.1% | 61%      |

### Phase 05 — Risk Management
Layered risk controls applied on top of Phase 04 signals:

1. **Kelly filter** — skip UP signals where `half-Kelly ≤ 0` (negative expected edge)
2. **ATR hard stop** — exit if close < entry price − 2×ATR (caps per-trade downside)
3. **Circuit breaker** — pause entries after 15% drawdown from peak; auto-reset after 63 bars (~3 months)
4. **Portfolio heat** — max 20% of capital at risk simultaneously (for multi-asset deployment)
5. **VaR / CVaR** — rolling 252-day historical simulation, 95% VaR and 99% CVaR

**Highlight:** SPY's Kelly filter correctly identifies avg_loss ($247) > avg_win ($188) as negative edge — blocks 98% of trades, cutting Max DD from −12.4% → −2.4% and turning CAGR positive (+2.5% vs −1.3%).

---

## Research Programme — Phase 06

The project has expanded from a single ML strategy into a full quantitative research programme with four parallel workstreams and a target academic publication.

### Research Question

> *Which classic quant strategies outperform in which market regimes, and can a regime-gating meta-strategy built on HMM state detection + Kelly allocation systematically capture that edge?*

**Target paper:** *"Regime-Gated Alpha Trends: A Unified Framework for Strategy Selection Under Non-Stationary Market States"*

---

### Phase 06A — Strategy Zoo

Implement and backtest 8 classic quant strategies on a 20–30 ticker universe (equities + crypto) over 5+ years OOS. Each strategy is self-contained: its own signal generation, position sizing, and backtest run.

| Strategy | Signal Logic | Regime Hypothesis |
|----------|-------------|-------------------|
| Momentum (12-1) | 12-month return minus most recent month; long top quintile | Trend-following regime |
| Mean Reversion | Bollinger Band squeeze + RSI extremes | High-vol ranging regime |
| EMA Crossover | Fast/slow EMA crossover + ADX filter | Trending / low-noise regime |
| Turtle Trading | Donchian channel breakout, ATR-sized units | Breakout regime |
| Pairs / Stat Arb | Cointegrated pair z-score entry/exit | Low-correlation regime |
| Carry Proxy | Yield spread as equity carry signal | Rate environment dependent |
| Volatility Breakout | ATR expansion + volume surge | Volatility compression → expansion |
| Alpha Trends | TBD — to be defined from literature + research | TBD |

**Asset universe:** 20–30 tickers (S&P500 large-caps, sector ETFs, crypto) to ensure statistically credible results for publication.

### Phase 06B — Strategy Ranking System

Rank all strategies against each other and against the Phase 03–05 ML ensemble using a unified scorecard:

| Metric | Description |
|--------|-------------|
| CAGR | Compound annual growth rate (OOS) |
| Sharpe | Risk-adjusted return (annualised) |
| Sortino | Downside-deviation-adjusted return |
| Max Drawdown | Worst peak-to-trough decline |
| Calmar | CAGR / Max DD |
| Alpha (vs SPY) | Jensen's alpha, regression against benchmark |
| Beta | Market exposure coefficient |
| t-stat on returns | Statistical significance of mean daily return ≠ 0 |
| Information Ratio | Active return / tracking error |
| Regime breakdown | Sharpe / CAGR decomposed per HMM state |

### Phase 06C — Environment Characterisation

Characterise the algorithmic trading environment this engine operates in:

- **Regime analysis** — how strategy performance varies across the 3 HMM states (bull / bear / ranging)
- **Transaction cost sensitivity** — how much edge survives realistic friction (commission, slippage, market impact) at different AUM levels
- **Signal stability** — rolling Sharpe and rolling Alpha windows; detect strategy decay over time
- **Factor attribution** — Fama-French 3-factor decomposition; isolate alpha from size/value/market exposure
- **Cross-strategy correlation** — identify diversifying pairs for eventual portfolio construction

### Phase 06D — Research Paper

Synthesise findings from 06A–06C into a publishable paper:

**Title:** *Regime-Gated Alpha Trends: A Unified Framework for Strategy Selection Under Non-Stationary Market States*

**Core thesis:** The HMM regime classifier built in Phase 02 can act as a meta-strategy gating layer — dynamically allocating capital to the strategy that historically performs best in the current regime state, sized by Kelly Criterion. This outperforms any single strategy and the passive benchmark on a risk-adjusted basis.

**Outline:**
1. Introduction — motivation, research question, contributions
2. Related work — HMM in finance, regime-switching models, strategy selection literature
3. Data and methodology — universe, feature construction, HMM calibration, strategy implementations
4. Strategy Zoo results — individual strategy scorecards, regime decomposition
5. Regime-Gated Meta-Strategy — allocation mechanism, Kelly sizing, backtested performance
6. Environment characterisation — cost sensitivity, signal decay, factor attribution
7. Discussion — practical limitations, transaction costs, overfitting risks
8. Conclusion — key findings, future directions

**Target:** Academic journal (Journal of Financial Economics, Quantitative Finance, or similar)

### Phase 06E — Live Execution *(runs in parallel)*

Paper trading on Alpaca while 06A–06D are in progress — generates real data for the paper's "live validation" section:

- Daily cron (GitHub Actions) — fetch data → features → model inference → signal output
- Alpaca paper trading API — bracket orders with ATR stop + limit
- Signal log in SQLite / Supabase
- Email / Slack alert on signal fire or circuit breaker trip

---

## Performance Targets (MVP)

| Metric | Target | Status |
|--------|:------:|--------|
| Sharpe Ratio | > 1.0 | ⏳ Phase 06 portfolio required |
| Max Drawdown | < 20% | ⚠️ Met per-ticker for AAPL; wider on others |
| Win Rate | > 55% | ✅ 61–64% on AAPL, MSFT, GOOGL, QQQ |
| CAGR | > 15% | ⏳ Currently 7–10% single-asset |
| Calmar Ratio | > 0.5 | ✅ AAPL: 1.11 |
| t-stat on returns | > 2.0 | ⏳ Phase 06B ranking |

Sharpe ≥ 1.0 is a **portfolio-level gate**. A 20–30 ticker deployment with per-strategy allocation provides the diversification needed:
- Portfolio Sharpe ∝ √N for N uncorrelated assets
- Regime gating removes allocation to strategies with negative edge in current state

---

## Technology Stack

| Layer | Libraries |
|-------|-----------|
| Data | `yfinance`, `ccxt`, `pandas-datareader`, `pyarrow` |
| Features | `pandas-ta`, `hmmlearn`, `scipy`, `numpy` |
| ML | `xgboost`, `lightgbm`, `torch`, `scikit-learn` |
| Tuning | `optuna` |
| Tracking | `mlflow` |
| Backtest | custom NumPy/Pandas simulator, `quantstats` |
| Risk | `numpy`, `scipy` |
| Research | `statsmodels` (factor attribution), `pandas` |
| Live execution | `alpaca-trade-api`, `schedule` |
| Testing | `pytest` |

---

## CLI Reference

```bash
# Data
python3 data_retrieval.py [--ticker AAPL MSFT ...] [--start 2018-01-01]

# Training
python3 train.py --ticker AAPL                   # single ticker
python3 train.py --all-tickers --n-trials 50     # all default tickers
python3 train.py --ticker AAPL --skip-lstm        # skip LSTM (faster)

# Backtesting
python3 backtest.py --ticker AAPL                # single ticker
python3 backtest.py --all-tickers --no-mc        # skip Monte Carlo
python3 backtest.py --all-tickers --confidence 0.45 --mode long_short

# Risk
python3 risk.py --ticker AAPL                    # single ticker
python3 risk.py --all-tickers                    # all tickers
python3 risk.py --all-tickers --atr-mult 3.0     # wider stop (3×ATR)
python3 risk.py --all-tickers --kelly 0.25       # quarter-Kelly filter

# Tests
python3 -m pytest tests/ -v
python3 -m pytest tests/test_risk.py -v          # risk tests only
```

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 01 Data | ✅ Done | Parquet store, gap-fill, split-adjust |
| 02 Features | ✅ Done | 30+ TA + stat + regime features, SHAP selection |
| 03 ML Models | ✅ Done | XGB + LGBM + LSTM ensemble, walk-forward CV, MLflow |
| 04 Backtest | ✅ Done | Vectorized OOS sim, Monte Carlo, param sweep |
| 05 Risk | ✅ Done | Kelly filter, ATR stop, circuit breaker, VaR/CVaR |
| 06A Strategy Zoo | 🔜 Next | 8 classic quant strategies benchmarked on 20–30 tickers |
| 06B Ranking | 🔜 Next | Unified scorecard: Sharpe, Alpha, IR, t-stat, regime breakdown |
| 06C Env Analysis | 🔜 Next | Cost sensitivity, signal decay, factor attribution |
| 06D Research Paper | 🔜 Next | "Regime-Gated Alpha Trends" — target academic journal |
| 06E Live Execution | 🔁 Parallel | Alpaca paper trading, daily cron, bracket orders |
| 07 Dashboard | 🔜 Planned | Streamlit + Plotly, equity curves, live signals |

---

## Build Logs

Detailed implementation notes, design decisions, and bug postmortems for each phase:

- [`docs/PHASE_01_LOG.md`](docs/PHASE_01_LOG.md) — Data Layer
- [`docs/PHASE_02_LOG.md`](docs/PHASE_02_LOG.md) — Feature Engineering
- [`docs/PHASE_03_LOG.md`](docs/PHASE_03_LOG.md) — ML Modeling
- [`docs/PHASE_04_LOG.md`](docs/PHASE_04_LOG.md) — Backtesting
- [`docs/PHASE_05_LOG.md`](docs/PHASE_05_LOG.md) — Risk Management
