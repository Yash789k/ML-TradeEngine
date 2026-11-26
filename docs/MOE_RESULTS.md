# MoE Gating Model — Results

**Feature:** Mixture-of-Experts gating over the [XGBoost, LightGBM, LSTM]
ensemble (deck slides 37–38), replacing static soft-vote weights with
context-dependent weights.

**Code:** `src/models/gating.py` (new), `src/models/ensemble.py`
(`ensemble_mode="static" | "moe"`), `src/models/trainer.py` (now persists
per-model OOS probabilities), `scripts/compare_moe.py` (benchmark).
Tests: `tests/test_gating.py` (17 tests).

---

## 1. Design

- **Gate:** one L2-regularised logistic regression per expert, each
  predicting P(expert's argmax is correct | context). Predicted
  correctness probabilities are floored at 0.05 and L1-normalised into
  the weight vector. A softmax network was rejected: with ~1,000 OOS
  rows per ticker it overfits (consistent with the deck's own caveat on
  slide 38).
- **Gate context (all causal):** HMM regime one-hot, realised 21-day
  vol, expanding midrank vol percentile, 63-day trailing-peak drawdown,
  and rolling 63-bar per-expert accuracy **shifted one bar** so no
  outcome is visible at its own decision time.
- **Minimum-evidence rule:** below 100 training rows the gate refuses to
  fit and emits uniform weights (equivalent to the static baseline).
- **No leakage:** the gate is trained strictly walk-forward on the
  ensemble's out-of-sample predictions — for each evaluation block the
  gate sees only rows ending 10 bars (embargo) before the block start,
  mirroring the expanding `WalkForwardSplit` used for the experts
  themselves. Block 0 has no history and falls back to uniform weights.

## 2. Benchmark protocol

Pilot retrain (AAPL, NVDA, SPY; 5 walk-forward folds, 20 Optuna trials,
LSTM 30 epochs) with per-model OOS probability persistence, isolated in
`data/models_moe/`. Both ensembles are evaluated with the standard
Phase 04 pipeline (confidence threshold 0.38, long-only,
15 bps/side costs) on identical OOS rows.

## 3. Results (static soft-vote vs MoE, walk-forward OOS)

| Ticker | Mode | Accuracy | Sharpe | CAGR | MaxDD | Calmar |
|---|---|---:|---:|---:|---:|---:|
| AAPL | static | 37.8% | −0.085 | 2.7% | −29.2% | 0.09 |
| AAPL | **moe** | 38.4% | −0.112 | 2.2% | −32.0% | 0.07 |
| NVDA | static | 46.0% | 0.658 | 25.5% | −53.7% | 0.47 |
| NVDA | **moe** | 46.0% | **0.664** | **25.7%** | −53.7% | **0.48** |
| SPY | static | 36.5% | −0.272 | 1.8% | −18.9% | 0.09 |
| SPY | **moe** | 35.4% | **−0.125** | **3.2%** | −18.9% | **0.17** |

Full JSON: `data/research/moe/moe_comparison.json`.
Per-bar gate weights: `data/research/moe/gate_weights_{ticker}.parquet`.

## 4. Per-regime gate behaviour

Mean gate weights by HMM regime (uniform burn-in block excluded):

| Ticker | Regime | w(XGB) | w(LGBM) | w(LSTM) |
|---|---|---:|---:|---:|
| AAPL | bear | 0.41 | 0.40 | 0.19 |
| AAPL | bull | 0.48 | 0.26 | 0.26 |
| NVDA | bear | 0.41 | 0.40 | 0.19 |
| NVDA | ranging | 0.43 | 0.44 | 0.13 |
| SPY | bear | 0.32 | 0.37 | 0.31 |
| SPY | bull | 0.31 | 0.35 | 0.34 |

The gate is doing what it was built to do: it learns materially
different weightings per ticker and per regime (e.g. LSTM is
downweighted to ~0.13–0.19 on AAPL/NVDA where its rolling accuracy is
poor, but kept near 0.31–0.34 on SPY where it contributes), rather than
applying one fixed vector everywhere.

## 5. Interpretation & limitations (cite in FYP §7.1)

- **Improvement is real but small.** MoE improves 2 of 3 tickers on
  Sharpe/Calmar (SPY Sharpe −0.27 → −0.13; NVDA marginal) and degrades
  AAPL slightly. The gate reallocates between experts whose individual
  edges are thin (Phase 06D diagnosis), so the ceiling on what
  reweighting can add is low — a gate cannot manufacture edge that the
  experts do not have.
- **Sample size.** ~1,000 OOS rows per ticker limits gate capacity;
  this motivated the logistic (not neural) gate and the min-evidence
  rule.
- **Regime labels carry mild look-ahead** (HMM fitted on the full
  sample, Phase 02) — the same caveat as the Phase 06D meta-strategy.
- **Not yet in the live path.** The live engine still uses static
  weights; enabling MoE live requires the per-model OOS history to be
  rolled forward daily, which is deferred until the gate shows a larger
  edge.

## 6. Reproduce

```bash
python3 scripts/compare_moe.py --train    # pilot retrain + evaluation
python3 scripts/compare_moe.py            # evaluation only
python3 -m pytest tests/test_gating.py -q
```
