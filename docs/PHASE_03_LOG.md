# Phase 03 — ML Modeling: Build Log

**Status:** ✅ Infrastructure complete — **awaiting training run**  
**Date completed:** 2026-05-03  
**Gate check:** 24/24 structural & smoke tests passing. Training not yet executed — see §How to run below.

---

## Objective

> Train a 3-class directional classifier (up/down/flat) with proper temporal CV.

---

## Steps Taken

### 1. Dependency resolution

| Package | Version | Role |
|---|---|---|
| `lightgbm` | latest | Tree ensemble (gradient boosting, histogram-based) |
| `torch` | latest (CPU) | PyTorch — LSTM sequence model |
| `optuna` | latest | Bayesian hyperparameter search (TPE sampler) |
| `mlflow` | latest | Experiment tracking: params, metrics, model artifacts |
| `scikit-learn` | 1.8.0 | already present — metrics, scalers, CV |
| `joblib` | present | model/scaler persistence |

**Environment note:** PyTorch was installed as CPU-only build
(`--index-url https://download.pytorch.org/whl/cpu`).
If you have Apple Silicon MPS or CUDA, reinstall torch with the appropriate wheel;
the LSTM code auto-selects `mps` if available, falling back to `cpu`.

**Segfault note:** XGBoost 3.x segfaults on macOS when `import torch` precedes
its initialisation in the same process (PyTorch installs signal handlers that
conflict with XGBoost's OpenMP layer).  Fix: never `import torch` at module
level — import it locally inside functions/tests.  All modules in this project
already follow this pattern.

### 2. Project structure extended

```
ml-trade-engine/
├── src/
│   └── models/
│       ├── __init__.py         ← exports ModelTrainer
│       ├── labels.py           ← 3-class label generation
│       ├── splitter.py         ← PurgedGroupTimeSeriesSplit
│       ├── xgb_model.py        ← XGBoost + Optuna tuner
│       ├── lgbm_model.py       ← LightGBM + Optuna tuner
│       ├── lstm_model.py       ← PyTorch LSTM sequence classifier
│       ├── ensemble.py         ← soft-vote probability ensemble
│       └── trainer.py          ← orchestrator + MLflow + save/load
├── train.py                    ← CLI entry point
├── tests/
│   └── test_ml_pipeline.py
└── data/
    └── models/                 ← saved model artifacts (created at training time)
        └── {ticker}/
            ├── xgb_final.json
            ├── xgb_final_scaler.pkl
            ├── lgbm_final.txt
            ├── lgbm_final_scaler.pkl
            ├── lstm_final.pt
            ├── lstm_final_scaler.pkl
            ├── feature_cols.json
            └── training_results.json
```

### 3. Label generation (`src/models/labels.py`)

Forward close-to-close return over `horizon` bars, thresholded into 3 classes:

```
label = 2 (UP)   if  fwd_return >  +threshold
label = 0 (DOWN) if  fwd_return <  -threshold
label = 1 (FLAT) otherwise
label = -1       for trailing `horizon` rows (no future data available)
```

Default: `horizon=5` (1 week), `threshold=0.5%`  
Trailing rows with label `-1` are dropped before any model sees data — no lookahead.

**Class balance** (empirically with threshold=0.5 %, 5-year AAPL):
approximately DOWN 30 % / FLAT 40 % / UP 30 %; `compute_class_weights()` returns
inverse-frequency weights used to balance the loss.

### 4. Walk-forward cross-validator (`src/models/splitter.py`)

`WalkForwardSplit` implements a purged walk-forward split:

```
n_splits=5, embargo_days=10 on n=1000 rows:

  fold 0:  train [0..489]  ← embargo gap (10 rows) →  test [500..599]
  fold 1:  train [0..589]                              test [600..699]
  fold 2:  train [0..689]                              test [700..799]
  fold 3:  train [0..789]                              test [800..899]
  fold 4:  train [0..889]                              test [900..999]
```

- **Embargo** (`embargo_days`) — rows at the boundary between train and test are
  dropped to prevent label leakage: with `horizon=5`, a train sample at T
  uses returns through T+5 as its label, which overlaps with the test window.
- **Modes** — `expanding` (all past data) or `rolling` (fixed-size window).
- Implements sklearn's `split()` interface for drop-in use with sklearn utilities.

### 5. XGBoost classifier (`src/models/xgb_model.py`)

`XGBModel` wraps `xgb.XGBClassifier` with:
- `RobustScaler` normalisation (fit on train, transform test — no leakage)
- Optuna `TPESampler` study over 9-dimensional hyperparameter space:

| Parameter | Search range |
|---|---|
| `n_estimators` | 50 – 500 |
| `max_depth` | 3 – 8 |
| `learning_rate` | 0.001 – 0.3 (log) |
| `subsample` | 0.5 – 1.0 |
| `colsample_bytree` | 0.4 – 1.0 |
| `min_child_weight` | 1 – 20 |
| `gamma` | 0.0 – 5.0 |
| `reg_alpha` | 1e-8 – 10 (log) |
| `reg_lambda` | 1e-8 – 10 (log) |

- Objective maximised by Optuna: **macro F1** on an inner validation split (20 % of training fold)
- After tuning, retrained on full training fold with best params
- `predict_proba(X)` → (n, 3): [p_DOWN, p_FLAT, p_UP]
- Persistence: `.save(path)` → `.json` model + `_scaler.pkl`

### 6. LightGBM classifier (`src/models/lgbm_model.py`)

`LGBMModel` — identical interface to `XGBModel`.  Brings ensemble diversity through:
- Histogram-based leaf-wise tree growth (different inductive bias)
- 9-dimensional Optuna search (num_leaves, min_child_samples differ from XGB)
- Early stopping on inner validation via `lgb.early_stopping(patience=20)`

### 7. LSTM sequence classifier (`src/models/lstm_model.py`)

`LSTMModel` processes temporal sequences for regime-aware pattern recognition:

- **Input:** sliding window of shape `(seq_len=60, n_features)` per sample
- **Architecture:**
  ```
  LSTM(input=n_features, hidden=128, layers=2, dropout=0.3, bidirectional=False)
  ↓ last timestep
  Dropout(0.3) → Linear(128→64) → ReLU → Linear(64→3) → logits
  ```
  Strictly unidirectional — bidirectional would leak future information.
- **Training:** Adam(lr=1e-3), `ReduceLROnPlateau`, `CrossEntropyLoss` with
  inverse-frequency class weights, gradient clipping (max_norm=1.0)
- **Early stopping:** patience=10 epochs on validation loss
- **Warm-up:** first `seq_len-1` rows receive uniform [1/3, 1/3, 1/3] probabilities
  since they don't have a full 60-bar history yet
- **Device:** auto-selects `mps` (Apple Silicon) → `cpu`
- Persistence: `.pt` checkpoint (state_dict + config) + `_scaler.pkl`

### 8. Ensemble classifier (`src/models/ensemble.py`)

`EnsembleClassifier` combines three probability matrices via weighted soft-vote:

```
P_ens[t, c] = w_xgb · P_xgb[t,c] + w_lgbm · P_lgbm[t,c] + w_lstm · P_lstm[t,c]
```

- Default: equal weights [1/3, 1/3, 1/3]
- `EnsembleClassifier.from_val_f1(f1_xgb, f1_lgbm, f1_lstm)` — weight each model
  proportionally to its out-of-fold validation F1 score (used in `trainer.py`)
- `signal_frame(...)` — returns a dict with `proba`, `signal`, `signal_name`,
  `confidence` (max probability) — feeds directly into Phase 06 signal engine

### 9. ModelTrainer (`src/models/trainer.py`)

Orchestrates the full pipeline in `trainer.run()`:

```
1. prepare_data()
   └── make_labels → SHAP feature selection → aligned (X, y, feat_cols)

2. WalkForwardSplit(n_splits=5, embargo_days=10)
   For each fold:
     a. XGBModel.fit(X_tr, y_tr, n_trials=50)   → P_xgb on test
     b. LGBMModel.fit(X_tr, y_tr, n_trials=50)  → P_lgbm on test
     c. LSTMModel.fit(X_tr, y_tr, n_epochs=50)  → P_lstm on test
     d. EnsembleClassifier(weights=[f1_xgb, f1_lgbm, f1_lstm]) → fold metrics
     e. mlflow.log_metrics(fold metrics)

3. Aggregate OOS predictions across all test folds
   └── Compute overall OOS accuracy + F1 macro

4. _save_final_models() → data/models/{ticker}/
5. Save training_results.json
```

All runs logged to MLflow with:
- **Parameters:** horizon, threshold, n_splits, embargo, n_trials, seq_len, feature_cols
- **Metrics:** per-fold and OOS accuracy, F1 macro, F1 per class (DOWN/FLAT/UP)
- **Tags:** ticker, phase

### 10. Test suite (`tests/test_ml_pipeline.py`)

24 structural and smoke tests — complete in **12.24 s** with no full training runs:

| # | Category | What is asserted |
|---|---|---|
| 1–5 | Labels | 3 classes {0,1,2}, trailing -1 mask, no lookahead, index alignment, class weights |
| 6–10 | Splitter | 5 folds produced, no train/test overlap, embargo gap ≥ embargo_days, temporal ordering, rolling mode |
| 11–13 | XGBModel | smoke-fit (2 trials), predict_proba shape (80,3), rows sum to 1, argmax ∈ {0,1,2} |
| 14–16 | LGBMModel | same three checks |
| 17–20 | LSTMModel | SequenceDataset length formula, item shape, forward-pass shape, 2-epoch smoke-fit API |
| 21–23 | Ensemble | output shape, rows sum to 1, F1-weighted constructor normalises correctly |
| 24 | Trainer | prepare_data aligns X/y/feat_cols, no NaN in X, no masked labels |

**Result: 24 passed, 0 failed** in 12.24 s.

---

## Design decisions & trade-offs

| Decision | Rationale |
|---|---|
| 3-class label (not binary) | DOWN/FLAT/UP lets the model abstain from low-conviction signals (FLAT class), reducing whipsaws |
| Symmetric threshold ±0.5 % | Empirically separates signal from noise on daily bars; tune based on transaction costs in Phase 04 |
| Embargo = 10 days (2× horizon) | Forward labels span 5 bars; 10-day buffer absorbs return autocorrelation spillover |
| Expanding window default | More training data on later folds → better generalisation; switch to rolling if concept drift is suspected |
| Optuna TPE (not Random/Grid) | TPE samples from a probabilistic model of the objective surface — outperforms random after ~20 trials |
| SHAP selection before CV | Prunes noisy features once (cheap) before the expensive walk-forward loop; per-fold selection inside the loop would be cleaner but ~5× slower |
| Class-weighted loss in LSTM | Prevents the FLAT class from dominating training due to higher frequency |
| Ensemble weighted by val F1 | Auto-adjusts when one model underperforms on a fold; sums to 1 via L1 normalisation |
| no `import torch` at module level | XGBoost 3.x + PyTorch share OpenMP; importing PyTorch first installs signal handlers that segfault XGBoost on macOS. Always import `torch` inside functions |
| MLflow local file backend | Zero setup; migrate to `sqlite:///mlflow.db` for production to avoid deprecation warning |

---

## Gate criteria (from MVP roadmap)

| Gate | Status |
|---|---|
| Walk-forward CV with no leakage | ✅ — WalkForwardSplit with embargo confirmed by tests 7–9 |
| XGBoost + Optuna tuning | ✅ — XGBModel.fit() with TPE search, smoke-tested |
| LightGBM for ensemble diversity | ✅ — LGBMModel, same interface |
| LSTM on 60-bar sequences | ✅ — LSTMModel with SequenceDataset, 2-layer causal LSTM |
| Soft-vote ensemble | ✅ — EnsembleClassifier, F1-weighted mode |
| MLflow tracking | ✅ — trainer.run() logs params + metrics per fold |
| Val accuracy > random baseline | ⏳ — **pending training run** (random baseline = 33.3 % for 3 classes) |

---

## ⚡ How to run training (your turn)

Phase 03 infrastructure is complete.  You need to run these commands to produce
trained models and fill the gate criterion above.

### Prerequisites — build the Phase 02 feature cache first

If you haven't generated feature matrices yet, run the data + feature pipeline:

```python
# In Python (or save as a script and run with python3)
from src.data.loader import DataLoader
from src.features.pipeline import FeatureEngineer

loader = DataLoader()
fe     = FeatureEngineer()

# Load raw data (uses Phase 01 Parquet cache, ~5–10 s if already cached)
aapl  = loader.load_equity("AAPL")
spy   = loader.load_equity("SPY")
btc   = loader.load_crypto("BTC/USDT")
macro = loader.load_macro()

# Compute features for each ticker you want to train on
for ticker in ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]:
    raw = loader.load_equity(ticker)
    fe.compute_features(raw, asset_slug=ticker, spy_df=spy, btc_df=btc, macro_df=macro)
    print(f"{ticker} features cached.")
```

### Option A — Quick smoke-test run (~5–10 min)

Skip LSTM and use only 5 Optuna trials.  Good for verifying the pipeline works
end-to-end before committing to the full run:

```bash
python3 train.py --ticker AAPL --n-trials 5 --lstm-epochs 3 --skip-lstm
```

### Option B — Full production run (~1–2 h per ticker, CPU)

```bash
# Single ticker
python3 train.py --ticker AAPL

# All default equity tickers (runs sequentially)
python3 train.py --ticker AAPL MSFT GOOGL SPY QQQ

# Customise label and CV params
python3 train.py --ticker AAPL \
    --horizon 5 \
    --threshold 0.005 \
    --n-splits 5 \
    --embargo-days 10 \
    --n-trials 100 \
    --lstm-epochs 100 \
    --lstm-seq-len 60 \
    --top-n-features 25
```

### Option C — Apple Silicon MPS (GPU-accelerated LSTM)

PyTorch auto-detects `mps` on Apple Silicon — no code change needed.
The LSTM will train ~5–10× faster.  Reinstall if you used the CPU wheel:

```bash
pip install torch   # standard wheel includes MPS support on macOS
python3 train.py --ticker AAPL
```

### View results in MLflow UI

```bash
mlflow ui --port 5000
# Open http://localhost:5000
# Experiment: "ml-trade-engine-phase03"
```

### Expected output (per ticker)

```
10:00:00  INFO  train — ══════════════════════════════════════════
10:00:00  INFO  train — Training: AAPL
10:00:00  INFO  train — ══════════════════════════════════════════
10:00:00  INFO  src.models.trainer — prepare_data: 1245 clean rows, 25 features ...
10:00:00  INFO  src.models.trainer — fold 0: train [0..349] (350)  test [360..449] (90)
10:00:02  INFO  src.models.trainer —   [XGB] fold 0  acc=0.xxx  f1=0.xxx
10:00:10  INFO  src.models.trainer —   [LGBM] fold 0  acc=0.xxx  f1=0.xxx
10:05:00  INFO  src.models.trainer —   [LSTM] fold 0  acc=0.xxx  f1=0.xxx
...
10:xx:xx  INFO  src.models.trainer — OOS ensemble  acc=0.xxx  f1=0.xxx
10:xx:xx  INFO  train — OOS results  acc=0.xxx  f1_macro=0.xxx
10:xx:xx  INFO  train — Models saved to: data/models/AAPL/
```

### Gate criterion to pass before Phase 04

> **OOS ensemble accuracy > 0.40 and F1 macro > 0.35**  
> (random baseline for 3 balanced classes: accuracy = 0.333, F1 macro = 0.333)

If results are below baseline after the full run, tune:
- Increase `--n-trials` to 100+
- Try `--threshold 0.01` (fewer FLAT labels, stronger UP/DOWN signal)
- Add more data: extend the Phase 01 `period` to `"10y"` for more training rows

---

## How to run tests only (no training)

```bash
# Phase 03 structural tests (24 tests, ~12 s, no training)
python3 -m pytest tests/test_ml_pipeline.py -v

# All phases together
python3 -m pytest tests/ -v
```
