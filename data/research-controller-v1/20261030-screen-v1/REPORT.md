# Controller screen: exploratory results

This is a predictive feasibility screen on previously inspected 2025 history, using immutable rolling forecasts. It is not a fresh holdout or a trading result.

Evaluation: 250 dates, 5500 dependent symbol-date observations, 22 assets. No base models were retrained; the stack used two registered fits.

| Model | Log loss ↓ | Accuracy | Brier ↓ |
|---|---:|---:|---:|
| fixed_mixture | 1.098255 | 39.96% | 0.667362 |
| hedge | 1.062424 | 42.13% | 0.644098 |
| prior | 1.063110 | 42.65% | 0.644095 |
| stack | 1.059851 | 44.24% | 0.641578 |
| uniform3 | 1.072557 | 41.31% | 0.651216 |

Primary paired date-block inference (block 10; 2,000 replicates; Holm family of six). Negative effects favor the controller.

| Contrast | Δ log loss | 95% interval | Holm p |
|---|---:|---|---:|
| hedge-vs-fixed_mixture | -0.035831 | [-0.049623, -0.022049] | 0.0030 |
| hedge-vs-prior | -0.000686 | [-0.006426, +0.004130] | 1.0000 |
| hedge-vs-uniform3 | -0.010133 | [-0.017064, -0.002971] | 0.0160 |
| stack-vs-fixed_mixture | -0.038403 | [-0.057736, -0.017753] | 0.0030 |
| stack-vs-prior | -0.003258 | [-0.017410, +0.012063] | 1.0000 |
| stack-vs-uniform3 | -0.012706 | [-0.028036, +0.003676] | 0.3313 |

Hedge updates once per matured date using the mean loss across all assets. Endpoints must be strictly before the decision, and each date is consumed once. The logistic stack fits only mature OOF rows in 2024 at the first 2025 decision, then mature 2024/first-half-2025 rows at the first July decision.

The source panel predates this screen and was already inspected. The original current-vintage adjustment, universe overlap and survivorship limitations remain. These results do not certify portfolio value, costs, controller integration, profitability, or promotion. The next stage needs a new registered portfolio test with identical capital, execution and risk constraints.

Completed in 4.963 seconds. Exact inputs/source hashes, split rules, budget and all six comparisons are in protocol.json; detailed metrics and block-6/21 sensitivity are in summary.json.
