# Findings behind the prompt

Read-only review on 28 September 2026, plus creation of this prompt package.
No new model training, test-suite execution, provider refresh, scheduler change
or source modification was performed in this side conversation.

## Reference quality versus current output

`/Users/a/Downloads/ASLcontextSign_report.pdf` has 30 PDF pages and approximately
8,335 extracted words, including contents and references. It presents a system
architecture, feature definitions/equations, 12 numbered tables, five numbered
figures, detailed class reports, confusion analysis, feature ablations, training
data efficiency, latency, confidence sensitivity and cross-signer transfer.
It cites 16 references. Representative PDF pages inspected visually: 10, 17,
21 and 22; text and section structure were reviewed throughout.

`output/pdf/ML_TradeEngine_Research.pdf` has seven pages and approximately 2,526
extracted words. It contains numerical tables, one equity-curve figure and five
reference entries. It is a concise methods/readiness report with aggregate
results. It does **not** lack underlying data: saved probabilities, folds and
ledgers support considerably more analysis than appears in the manuscript.

The quality gap is empirical and explanatory breadth, not simply length or
positive versus negative findings. The next study needs class/instrument detail,
error mechanisms, calibration, controlled ablations, generalization, learning
curves, operational measurements and stronger scholarly context.

The ASL paper is a useful coverage reference, not an infallible scientific
template. For example, PDF page 22 reports a completed cross-signer experiment,
while pages 25 and 27 still describe that dataset/experiment as unused/future
work. Its later experiments also use different sample totals. The new manuscript
must identify such version differences and remove contradictory status claims.
Do not copy the reference's claims of novelty, authorship or numerical success.

## Repository evidence checked

| Finding | Direct source | Consequence for the prompt |
|---|---|---|
| Verified release contains 22 instruments through 25 Sep 2026; requested history begins 29 Sep 2016 | `data/releases/2026-09-28-adaptive/market/manifest.json` | Reuse provenance; report actual common windows and exclusions |
| Full forecasts exist, not only summary scores | `experiment/*_predictions.parquet`: 686 rows × 18 columns for each instrument | Derive confusion, class/instrument, calibration and failure diagnostics now |
| Shared account data exists | `experiment/*_account.parquet`, `*_events.json`, `returns.parquet` | Derive costs, exposures, holdings, drawdown and loss attribution |
| Audit uses 1,056 classifications across 48 label-complete dates, and 54 account sessions | `experiment/classification.csv`, `portfolio_metrics.csv` | Preserve denominators and dependence; do not count 1,056 independent time observations |
| Hybrid accuracy 37.59%; prior 40.625%; net hybrid audit return -0.56% versus +2.03% benchmark | Same metric files | No current basis for superiority; richer analysis must not manufacture it |
| Stored verification reports 310 offline passes and 17 unrun provider integrations | `verification.json`, `pytest.ini` | Treat as saved evidence, not newly run tests or proof of unattended operation |
| Second archived run matched 66 checked files | `data/releases/2026-09-28-reproduction/verification.json` | Preserve/reuse exact reproduction; add clean-environment evidence when needed |
| Only a 25 Sep commissioning preview receipt is present | `data/paper/adaptive-v1/days/2026-09-25/result.json` | No prospective observations have yet been demonstrated |
| Daily schedule still calls `bootstrap_models.py` and `live.py run --dry-run` | `.github/workflows/daily_signals.yml` | Converge scheduling on the current daily trader and durable state |
| Current `trader.py` path runs a local adjusted-return proxy simulation | `trader.py`, `src/live/adaptive.py` | Do not equate it with actual broker shares/fills |
| Editing shared source can invalidate the current release | `src/live/adaptive.py::verify_release` and release `runtime_source` hashes | Isolate new research from the frozen pilot runtime |
| Empirical abstract and discussion numbers are embedded directly in strings | `scripts/build_adaptive_paper.py` | Generate/check prose against statistics, not only tables |
| Calibration temperature and expert gate use the same trailing block | `src/models/adaptive.py::fit_at` | Test a separated/cross-fitted alternative for claims of independent gate improvement |
| Existing tests include causal mutation, accounting, retries and mocked daily recovery | `tests/test_adaptive_release.py`, `test_portfolio_execution.py`, `test_causality_regressions.py` | Extend specific gaps; do not discard working tests or chase a larger total |

The two referenced tasks were read through `read_thread`:
`Make trading research trustworthy` and `Audit quant trading pipeline`.
They document the original defects, later repairs, failed strategy gates and
the explicit distinction between simulated and prospective/broker results.
Earlier task claims were checked against available repository evidence where
used above; they are not authority to execute old requests in this side task.

## Additional checks the next execution must resolve

- Bind release approval to the exact tested source/config and required behaviors;
  accepting a passing JUnit report alone does not establish this binding.
- Prove research/daily parity for model refit anchors and calendar rebalancing,
  not only account event handling. Different anchors can change forecasts.
- Record expected sessions and actual failures; derive live pilot progress from
  valid receipts rather than the release's static zero-session declaration.
- Test the full scheduler/runner/storage chain. A committed workflow, a manual
  success or a local source inspection is not evidence of a successful remote job.
- Keep exploratory diagnostics on consumed history separate from new registered
  claims. The July–September audit has already been inspected.
- Wait for all outcome endpoints before calling classification evaluation final;
  a pilot decision cutoff and its final label-maturity date differ.

The prompt is designed to produce a complete empirical investigation even if
the most defensible finding remains that the tested hybrid has no trading edge.
