# Dynamic Policy Layer — Results

**Feature:** Runtime-adaptive operating policy (deck slide 39) replacing
the hard-coded constants `_CONFIDENCE_THRESHOLD = 0.38` and
`_ATR_MULTIPLIER = 2.0`.

**Code:** `src/risk/policy_engine.py` (new), integrated into
`src/backtest/signals.py` (dynamic per-bar threshold),
`src/risk/engine.py` (per-bar ATR stop multiplier, size cap, entry
eligibility), `src/risk/stops.py` (parameterised multiplier), and the
live path `src/live/signal_generator.py` + `src/live/engine.py`
(enable with `python3 live.py run --policy`).
Tests: `tests/test_policy_engine.py` (13 tests, incl. a causality test).

---

## 1. Rules implemented

All inputs are causal (expanding midrank vol percentile; backward-only
regime stability), so the same policy frame drives backtest and live.

| Parameter | Static | Dynamic rule |
|---|---|---|
| Confidence threshold | 0.38 | +0.10 while the HMM state changed within the last 5 bars (transitioning regime → lower-quality signals) |
| ATR stop multiplier | 2.0× | linear ramp 2.0× → 3.0× as the causal vol percentile rises 0.80 → 1.00 |
| Position size cap | 100% | 50% when vol percentile > 0.90 or price > 10% below its 63-day peak |
| Trade eligibility | always | new entries blocked when vol percentile > 0.95 or regime unknown; open positions are managed by stops, never force-closed |

## 2. Benchmark protocol

Full Phase 04→05 pipeline (walk-forward OOS predictions → signals →
RiskEngine with Kelly filter, stops, circuit breaker, 15 bps/side) run
twice per ticker on the 22-asset universe with identical inputs; the
only difference is the policy layer. A third ablation variant
(`dyn_nobump`) disables the confidence tightening but keeps the
ATR/size/eligibility rules. Script: `scripts/compare_policy.py`;
artifacts: `data/research/policy/policy_comparison.{parquet,json}`.

## 3. Aggregate results (22 tickers)

| Metric | Static | Dynamic (full) | Dynamic (no conf. bump) |
|---|---:|---:|---:|
| Mean Sharpe | **−0.72** | −0.95 | −0.94 |
| Mean MaxDD | −24.3% | **−17.4%** | −17.6% |
| Mean Calmar | 0.013 | 0.014 | 0.009 |
| Tickers with better Sharpe vs static | — | 9/22 | 10/22 |
| Tickers with smaller MaxDD vs static | — | **20/22** | 20/22 |
| Mean trades per ticker | 40 | 53 | 52 |

Selected per-ticker deltas (full table in the parquet):

| Ticker | Sharpe static → dynamic | MaxDD static → dynamic |
|---|---:|---:|
| NVDA | −1.61 → **−0.77** | −39.3% → **−23.7%** |
| XLI | −1.64 → **−0.67** | −40.7% → **−17.9%** |
| TSLA | −0.60 → −1.17 | −34.5% → **−16.8%** |
| IWM | −1.51 → **−0.88** | −25.1% → **−15.3%** |
| XLF | +0.21 → −1.61 | −23.8% → −22.6% |
| JPM | +0.07 → −1.08 | −31.6% → −32.4% |

## 4. Interpretation (cite in FYP §7.1)

- **The policy layer is a drawdown-control mechanism, not an alpha
  source.** It cut the mean maximum drawdown by ~28% (−24.3% → −17.4%)
  and improved drawdown on 20 of 22 tickers — exactly what the
  size-cap/eligibility/wider-stop rules are designed to do. It did NOT
  improve mean Sharpe: on tickers where the ML signal happened to be
  profitable in high-vol periods (XLF, JPM), blocking those entries
  removed the gains.
- **The confidence-tightening rule adds nothing.** The no-bump ablation
  matches or slightly beats the full rule set — consistent with the
  Phase 06D finding that raising the confidence threshold does not help
  this ensemble (the probability spread carries little signal).
- **Underlying edge is the binding constraint.** All aggregate Sharpes
  are negative because the single-ticker ML strategy itself is weak
  (Phase 06D). The policy layer should be judged on risk shaping, and
  by that measure it works; pairing it with the regime-gated
  meta-strategy (which has positive edge) is the natural next step.
- **Recommended default:** enable the policy layer for live/paper
  trading (`--policy`) with `unstable_bump=0.0`, i.e. keep the ATR
  widening, stress size cap, and eligibility halt; skip the threshold
  tightening.

## 5. Reproduce

```bash
python3 scripts/compare_policy.py
python3 -m pytest tests/test_policy_engine.py -q
python3 live.py run --dry-run --policy      # live path with dynamic policy
```
