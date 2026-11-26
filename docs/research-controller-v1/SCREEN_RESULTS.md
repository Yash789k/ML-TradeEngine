# First controller screen — 30 October 2026

**Proposal alignment:** the proposal was supplied after this preliminary screen. Its main experiment is a contextual convex mixture gate on next-open, cost-and-volatility labels. This screen evaluates probability combination on the saved next-close, fixed-1% label contract; it does not complete that main experiment. See `PROPOSAL_ALIGNMENT_AND_GAPS.md` for the proposed continuation. All confidence intervals below are marginal bootstrap intervals; Holm adjusts the p-values, not those intervals.

Both new forecast controllers improved on the old equal linear/tree mixture in the consumed 2025 sample. **Neither established superiority to the rolling class-frequency prior.** This is useful feasibility evidence for further work, not a profitable-trading or deployment result.

The pre-registered experiment, outputs and immutable report are in `data/research-controller-v1/20261030-screen-v1/`. [Original generated report](../../data/research-controller-v1/20261030-screen-v1/REPORT.md), [protocol](../../data/research-controller-v1/20261030-screen-v1/protocol.json), and [machine-readable results](../../data/research-controller-v1/20261030-screen-v1/summary.json) provide the full record.

## What was actually run

The screen verified all 22 sealed original prediction files and their raw/adjusted market inputs, then reused the 2024–2025 out-of-fold forecasts. The 2024 observations supplied controller history; 2025 supplied two exploratory half-year evaluation blocks. There were **250 evaluation dates and 5,500 dependent symbol-date observations**. No new market download or base-model fit was performed.

- **Delayed Hedge:** starts with equal prior/linear/tree weights; updates once per fully matured date with the cross-sectional mean log loss and fixed learning rate 0.1. Each label's `t+6` endpoint must be strictly before the decision. It cannot consume the same date twice.
- **Logistic forecast stacking:** standardized log probabilities of the same three experts, L2 logistic regression with fixed `C=0.1`. Two fits only: mature 2024 rows at the first 2025 decision, then mature 2024/first-half-2025 rows at the first July decision.
- **Controls:** equal linear/tree probabilities, the existing rolling fitted prior, and an equal three-expert mixture. The three-expert mixture separates adding the prior from learning adaptive weights.

These are probability-combination methods. They do not yet select strategy allocations, optimize a utility function, or control risk/execution.

| Model | 2025 log loss ↓ | Accuracy |
|---|---:|---:|
| Equal linear/tree | 1.098255 | 39.96% |
| Equal prior/linear/tree | 1.072557 | 41.31% |
| Rolling prior | 1.063110 | 42.65% |
| Delayed Hedge | 1.062424 | 42.13% |
| Logistic forecast stack | 1.059851 | 44.24% |

The primary endpoint was log loss, fixed before evaluation. The stack's higher descriptive accuracy does not change that endpoint or supply a post-hoc promotion rule.

## What the comparisons support

The registered primary inference used 2,000 paired date-block bootstrap replicates, block length 10, seed 20261030, with each half-year kept separate. All assets stayed together within each date. Holm adjustment covered six controller-versus-control comparisons. Block-6 and block-21 sensitivity results are retained in `summary.json`.

| Comparison | Δ log loss | 95% interval | Holm p |
|---|---:|---|---:|
| Hedge versus equal linear/tree | −0.035831 | [−0.049623, −0.022049] | 0.0030 |
| Hedge versus prior | −0.000686 | [−0.006426, +0.004130] | 1.0000 |
| Hedge versus equal three-expert mixture | −0.010133 | [−0.017064, −0.002971] | 0.0160 |
| Stack versus equal linear/tree | −0.038403 | [−0.057736, −0.017753] | 0.0030 |
| Stack versus prior | −0.003258 | [−0.017410, +0.012063] | 1.0000 |
| Stack versus equal three-expert mixture | −0.012706 | [−0.028036, +0.003676] | 0.3313 |

Hedge supports a small historical improvement over the same-expert fixed mixture in this exploratory analysis. The stack's corresponding interval includes zero. Both prior comparisons remain inconclusive. Historical reuse, fixed-universe overlap and current-vintage adjustments limit generalization even where an interval excludes zero. Neither controller has been compared on matched-capital portfolio outcomes in this screen.

## Verification and continuation

The initial 11 behavior cases passed before registration: strict label delay, future-label mutation for Hedge, date-averaged updates, deterministic/idempotent feedback, invalid probabilities, checksum tampering, bootstrap/Holm oracles, and both stack partitions with training-only scaling. Evidence is `tests.json`; the research run used two meta-model fits and took 4.963 seconds.

The registered source and original result manifest remain unchanged. A separate qualification supplement, `data/research-controller-v1/qualification-20261030/reconstruction.json`, records exact package versions and reconstructs the omitted scaler variance/scale from eligible training rows. Applying the already-saved coefficients reproduces both sets of stack predictions within **1.12e−16** maximum absolute difference, with no new model fits.

Six additional independent review cases passed in 5.65 seconds. They cover full-stack future-label and future-probability mutation through the July refit, actual fit maturity/training hashes, unique mature-date Hedge feedback, all three baseline arithmetic identities, and immutable input/source/output verification. JUnit, exact test IDs, logs and environment versions are in `data/research-controller-v1/review-v1/`. Those tests fitted synthetic classifiers only; they added no real-data selection trial or registered research fit.

To verify the original package and its exact registered source without retraining:

```sh
data/research-v2/clean-env/bin/python -B - <<'PY'
import hashlib, json
from pathlib import Path
p = Path('data/research-controller-v1/20261030-screen-v1')
manifest = json.loads((p/'manifest.json').read_text())
protocol = json.loads((p/'protocol.json').read_text())
for collection in [manifest['files'], protocol['inputs'], protocol['sources']]:
    for path, expected in collection.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
print('Result, input and registered-source hashes match')
PY
```

To run the isolated original behavior checks without touching the research evidence:

```sh
data/research-v2/clean-env/bin/python -B -m pytest tests/controller_v1 -q -p no:cacheprovider
```

`screen register` and `screen run` deliberately refuse to overwrite an existing registration or attempted run. Any new research comparison needs a new module/output/protocol version and budget. The next useful experiment is a registered matched-execution portfolio comparison with identical intent mapping, starting capital, costs and risk controls, including the prior and fixed mixtures. Do not retune to this screen or use its inspected periods as a new untouched holdout.
