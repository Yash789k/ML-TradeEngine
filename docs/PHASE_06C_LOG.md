# Phase 06C — Environment Characterisation

**Status:** Complete  
**Tests:** 27 new + 176 total (all passing)  
**New files:**
- `src/research/env_analyzer.py`
- `tests/test_env_analyzer.py`

---

## Objective

Phase 06C characterises the *algorithmic trading environment* to answer four
questions that are essential for the research paper:

| # | Analysis | Question answered |
|---|---|---|
| 1 | **Regime Breakdown** | Does each strategy earn its Sharpe in the right regime state? |
| 2 | **Cost Sensitivity** | At what commission level does edge evaporate? |
| 3 | **Signal Decay** | Does strategy performance degrade over time? |
| 4 | **Factor Attribution** | How much of the return is true alpha vs. beta exposure? |

---

## Architecture

```
src/research/env_analyzer.py          EnvAnalyzer class
  ├── regime_breakdown()              → data/research/env/regime_breakdown.parquet
  ├── cost_sensitivity()              → data/research/env/cost_sensitivity.parquet
  ├── signal_decay()                  → data/research/env/signal_decay_{series,summary}.parquet
  ├── factor_attribution()            → data/research/env/factor_attribution.parquet
  └── run_all()                       → runs all four sequentially

research.py   (CLI)
  └── env subcommand                  → --regime / --cost / --decay / --factors / --show-only
```

Analyses 1, 3, 4 operate exclusively on **saved artifacts** from the Phase 06A
zoo run (`equity_curve.parquet`) — no re-simulation.  
Analysis 2 re-runs strategies at multiple cost levels on a configurable ticker
subset (default: AAPL, NVDA, SPY, TSLA, GLD × all 8 strategies).

---

## Analysis 1 — Regime Breakdown

### Method
For each `(ticker, strategy)` pair:

1. Fit a Gaussian HMM on the ticker's `(log_return, realized_vol_21)` series
   (same approach as Phase 02 — 3 states, full covariance, 500 iterations).
2. Decode Viterbi path → `{0: bear, 1: ranging, 2: bull}`.
3. Load the strategy's saved `equity_curve.parquet` and compute daily returns.
4. Mask returns by each regime state.
5. Compute per-regime:
   - `n_days` / `pct_days` — regime frequency
   - `ann_return` — mean daily return × 252 (proxy CAGR for non-contiguous periods)
   - `sharpe` — annualised Sharpe over regime days

### Robustness
The HMM uses a **covariance fallback chain**: tries `full` first; if the
covariance matrix is near-singular (can happen on very flat or synthetic data)
it retries with `diag`. `fit_hmm()` in `src/features/regime.py` now accepts an
explicit `covariance_type` parameter so callers can control this directly.

### What to expect
A well-designed strategy should show:
- `AlphaTrendsStrategy` highest Sharpe in the `bull` regime (it explicitly gates
  on HMM bull state) — this validates the paper's thesis.
- `Mean_Reversion` higher Sharpe in `ranging` than `bull` (it enters on BB
  extremes, which happen more in sideways markets).
- `Turtle_Breakout` positive Sharpe only in trending regimes.

---

## Analysis 2 — Cost Sensitivity

### Method
For each `(ticker, strategy)` pair (on the cost-sensitivity subset):

1. Re-run `strategy.run()` at six commission levels:
   `[0, 5, 10, 20, 50, 100]` basis points per side.
2. Slippage is held at 0 to isolate the commission effect.
3. Record `sharpe`, `cagr`, `n_trades` at each level.

### Output
`cost_sensitivity.parquet` indexed by `(ticker, strategy, commission)`.

### What to expect
- High-frequency strategies (Mean_Reversion, Vol_Breakout with many trades) will
  show rapid Sharpe decay and may go negative by 20–50 bps.
- Low-frequency strategies (Momentum_12_1, Carry_Proxy with ~10–30 trades/year)
  will remain positive even at 100 bps because few round-trips occur.
- `AlphaTrendsStrategy` break-even commission is a key data point for the paper:
  it should survive to at least 20–30 bps, demonstrating institutional viability.

---

## Analysis 3 — Signal Decay (Rolling Sharpe)

### Method
For every `(ticker, strategy)` pair found in `data/research/`:

1. Load `equity_curve.parquet` → daily returns.
2. Compute a rolling 90-day (≈ one quarter) Sharpe:
   `rolling_sharpe_t = mean(r[t-90:t]) / std(r[t-90:t]) × √252`
3. Compute summary statistics:
   - `slope` — annualised OLS slope of rolling Sharpe over time (negative = decaying)
   - `mean_sharpe` — average rolling Sharpe across all windows
   - `end_sharpe` — rolling Sharpe at the last available window
   - `pct_positive` — fraction of windows with Sharpe > 0
   - `is_decaying` — `True` if slope < 0 AND end_sharpe < mean_sharpe

### Output
Two parquet files:
- `signal_decay_series.parquet` — full time series `(ticker, strategy, date) → rolling_sharpe`
- `signal_decay_summary.parquet` — one row per `(ticker, strategy)`

### What to expect
Strategies with structural alpha (AlphaTrends, Momentum) should show stable or
improving rolling Sharpe.  Pure mean-reversion strategies may show decay in
strong trending periods (2020–2021 bull run, 2023 AI rally).

---

## Analysis 4 — Fama-French 3-Factor Attribution

### Method
For every `(ticker, strategy)` pair, run OLS:

```
r_strat - RF = α + β_mkt·(Mkt-RF) + β_smb·SMB + β_hml·HML + ε
```

where factors are downloaded from Kenneth French's data library via
`pandas_datareader.data.DataReader('F-F_Research_Data_Factors_daily', 'famafrench')`.

**Metrics per row:**

| Column | Description |
|--------|---|
| `alpha_ann` | Intercept × 252 — annualised "true alpha" after controlling for factor exposure |
| `alpha_tstat` | t-statistic on the intercept (H₀: α = 0) |
| `beta_mkt` | Market beta |
| `beta_smb` | Size factor loading (positive = small-cap tilt) |
| `beta_hml` | Value factor loading (positive = value tilt) |
| `r2` | OLS R² — how much variance is explained by the three factors |

**Fallback:** If the Kenneth French server is unavailable, proxy factors are
constructed from cached OHLCV:  
`Mkt-RF ≈ SPY`, `SMB ≈ IWM − SPY`, `HML ≈ XLF − XLK`.

### What to expect for the paper
- A strategy with **high `alpha_ann`** and **`|alpha_tstat| > 2`** has
  statistically significant alpha unexplained by the three factors.  This is the
  central claim of the paper for `AlphaTrendsStrategy`.
- Low R² (< 0.2) means the strategy's returns are not primarily driven by
  market/size/value exposure — supporting the "regime-gating" narrative.
- High `beta_mkt` with low alpha means the strategy is just leveraged beta.

---

## CLI Reference

```bash
# Run all 4 analyses on the default 5-ticker universe
python3 research.py env

# Run on the full 22-ticker research universe
python3 research.py env --research

# Run individual analyses
python3 research.py env --regime           # regime breakdown only
python3 research.py env --decay            # signal decay only
python3 research.py env --factors          # FF3 attribution only

# Cost sensitivity on a custom subset
python3 research.py env --cost --cost-tickers AAPL NVDA SPY TSLA GLD MSFT

# Print saved results without re-running
python3 research.py env --show-only
```

---

## Test Coverage (`tests/test_env_analyzer.py`)

| Class | Tests |
|---|---|
| `TestHelpers` | `_regime_sharpe` (basic, zero-variance), `_regime_ann_return`, `_compute_ticker_regime` (shape, short data) |
| `TestRegimeBreakdown` | returns DataFrame, expected columns, regime label set, pct_days ≈ 1, saved parquet, missing strategy dir graceful skip |
| `TestCostSensitivity` | Sharpe monotonically decreases with commission, saved parquet, `bps` column |
| `TestSignalDecay` | returns summary DataFrame, expected columns, `pct_positive ∈ [0,1]`, both parquets saved, short series skipped |
| `TestFactorAttribution` | runs without error, expected columns, `r2 ∈ [0,1]`, saved parquet |
| `TestPrintMethods` | all four `print_*` methods don't raise |

**Total:** 27 new tests, 176 in full suite — all pass.

---

## Design Decisions

**Why re-run simulations for cost sensitivity instead of adjusting saved PnL?**  
Adjusting a saved equity curve for different costs requires reconstructing the
compounding path trade-by-trade. Re-running the simulator at each cost level is
cleaner, more accurate, and strategies typically complete in < 1 second each.
For the default 5-ticker subset × 8 strategies × 6 cost levels = 240 runs, this
takes under 60 seconds.

**Why a 90-day rolling window for signal decay?**  
90 days ≈ one quarter — the standard institutional evaluation period.  It is
short enough to catch regime shifts (2020 COVID crash, 2022 rate-hike regime)
while being long enough that the Sharpe estimate has sufficient degrees of freedom
(~90 observations).

**Why Fama-French 3 factors and not 5?**  
The 3-factor model (market, size, value) is the academic standard for strategy
validation papers.  The 5-factor model adds profitability (RMW) and investment
(CMA), which are less relevant for short-term momentum/regime strategies and
would reduce statistical power due to multicollinearity.  Extending to 5 factors
is a natural robustness check for the final paper draft.

---

## Artifacts saved

```
data/research/env/
  regime_breakdown.parquet        (ticker, strategy, regime) → metrics
  cost_sensitivity.parquet        (ticker, strategy, commission) → sharpe, cagr, n_trades, bps
  signal_decay_series.parquet     (ticker, strategy, date) → rolling_sharpe
  signal_decay_summary.parquet    (ticker, strategy) → slope, mean_sharpe, end_sharpe, …
  factor_attribution.parquet      (ticker, strategy) → alpha_ann, alpha_tstat, betas, r2
```
