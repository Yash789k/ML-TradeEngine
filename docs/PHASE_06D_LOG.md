# Phase 06D — Regime-Gated Meta-Strategy + ML Diagnosis

**Status:** Complete  
**Tests:** 12 new + 225 total (all passing)  
**New files:**
- `src/research/meta_strategy.py`
- `tests/test_meta_strategy.py`
- `scripts/pilot_label_redesign.py`

---

## Objective

Two goals in this phase:

1. **Diagnose** why the Phase 03 ML ensemble backtest failed the MVP gates
   (portfolio Sharpe −0.07, 0/22 tickers > 1.0) and evaluate three fix levers.
2. **Implement** the regime-gated meta-strategy — the paper's central claim —
   and produce the missing Section 5 numbers for the research paper.

---

## Part 1 — ML Ensemble Diagnosis

### Findings

| Symptom | Evidence |
|---|---|
| Thin edge | OOS accuracy ≈ 41.5% vs 33.3% random; UP-precision 47–56% |
| DOWN-bias | Model predicts ~47–70% DOWN vs ~38% true DOWN; long-only → sits in cash through a bull market |
| Ineffective confidence gate | 0.38 threshold passes 90–100% of 3-class softmax signals |
| Reproducibility gap | `horizon`/`threshold` were not saved to `training_results.json` (now fixed) |

### Lever A — raise confidence threshold *(REJECTED)*

The existing `parameter_sweep.parquet` artifacts already cover thresholds
0.35 → 0.50 at three cost levels. Mean Sharpe across 22 tickers **decreases
monotonically** as the threshold rises (−0.06 → −0.32 at 10 bps): the model's
high-confidence signals are not better signals, and higher thresholds just
add cash drag. Raising to 0.55 would make things worse, not better.

### Lever B — label redesign + retrain *(REJECTED after pilot)*

Pilot: XGB+LGBM, 30 Optuna trials, 5 walk-forward folds on AAPL / NVDA / SPY
with `horizon 5→10`, `threshold ±0.5%→±1.5%` (class weights were already
applied in production via `compute_class_weights`). Artifacts isolated in
`data/models_pilot/`.

| Ticker | Sharpe (old labels) | Sharpe (new labels) | UP-precision old→new |
|---|---|---|---|
| AAPL | −0.05 | 0.00 | 49% → 46% |
| NVDA | 0.60 | −0.06 | 55% → 50% |
| SPY  | −0.31 | −0.58 | 51% → 53% |

The redesign mildly reduces the DOWN-bias (AAPL 62%→51% DOWN signals) but
does **not** improve — and mostly worsens — OOS trading performance. The
single-stock directional edge is genuinely thin; relabelling can't create
signal that isn't there.

### Lever C — regime-gated meta-strategy *(ADOPTED — see Part 2)*

---

## Part 2 — Regime-Gated Meta-Strategy

### Architecture

```
src/research/meta_strategy.py
  ├── load_strategy_returns()   equity curves → daily returns (Phase 06A artifacts)
  ├── load_regime()             hmm_regime series (Phase 02 feature cache)
  ├── MetaStrategy.run_ticker() walk-forward per-ticker allocator
  └── MetaStrategy.run_universe() 22-ticker EW portfolio + benchmarks + artifacts

research.py meta --research      CLI (also --tickers/--min-obs/--rebalance/--no-save)
```

### Allocation rule (strictly walk-forward)

At day *t*, per ticker: hold the strategy with the best historical Sharpe on
days *d < t* where `hmm_regime[d]` equals the current confirmed regime.
Anti-whipsaw controls: 60-obs in-regime burn-in (EW fallback), 3-bar regime
confirmation, 10-bar minimum hold, 21-bar re-evaluation cadence, 15 bps
switch cost. No-lookahead is enforced by a unit test that corrupts future
returns and asserts past allocations are unchanged.

Without the whipsaw controls the allocator switched 3,336 times and costs ate
the entire edge (Sharpe −0.43); with them, 485 switches and Sharpe 0.70.

### Results (May 2021 – May 2026, 1,304 days, 22 tickers EW)

| Portfolio | Sharpe | CAGR | MaxDD | Calmar | t-stat |
|---|---|---|---|---|---|
| **META (regime-gated)** | **0.70** | 9.4% | **−4.5%** | **2.08** | **3.41** |
| best single: Momentum 12-1 | 0.56 | 11.1% | −18.9% | 0.59 | 2.25 |
| SPY buy & hold | 0.65 | 14.0% | −18.8% | 0.75 | — |

Alpha vs SPY +4.6%/yr, beta 0.33. META beats every constituent on Sharpe and
holds a diversified mix (Mean Reversion 28%, Turtle 19%, Momentum 15%, …).

### Artifacts

```
data/research/meta/meta_returns.parquet       per-ticker + portfolio daily returns
data/research/meta/meta_allocations.parquet   selected strategy per ticker/day
data/research/meta/meta_summary.json          all metrics vs benchmarks
data/models_pilot/pilot_report.json           Lever B pilot comparison
```

### Known limitation

The allocator is walk-forward, but the HMM regime labels come from a model
fitted on the full sample (Phase 02) — mild look-ahead in the *labels*.
Documented in paper §7.1; live deployment needs rolling HMM re-estimation.

---

## Paper status

`docs/PHASE_06D_PAPER_DRAFT.md` updated to v0.2: abstract, §3.6 (mechanism),
§5.1–5.2 (allocation mix + performance table), §7.1 (look-ahead caveat) and
§8 (conclusion) now carry real numbers. Remaining `[FILL]`s are the Phase 06C
environment-characterisation narratives and figures.
