# Required evidence, tests and collection schedule

Companion specification for EXECUTION_PROMPT.md. These are requirements for the
next execution, not claims that the experiments or schedules already exist.

## A. Deliverables and dependency order

1. **Inventory and freeze:** data/source manifests, evidence matrix, inspected-period
   log, versioned protocol and compute budget. Preserve the current release.
2. **Extract existing evidence:** detailed prediction, portfolio and error diagnostics
   from verified saved outputs. Label these historical analyses exploratory.
3. **Close implementation gaps:** one operational policy path, state persistence,
   provenance-bound tests, fault handling and a verified schedule.
4. **Run registered experiments:** chronological comparisons, controlled ablations,
   generalization and robustness, with all trials recorded.
5. **Produce the complete historical manuscript:** linked tables, figures, sources,
   appendices and reproducibility package; mark prospective sections interim.
6. **Collect prospective observations:** immutable decisions, outcomes and operational
   receipts; follow the frozen protocol while research proceeds separately.
7. **Finalize the time-dependent study:** wait for label maturity, lock the dataset,
   reproduce results and update claims. Do not claim profitable deployment by default.

Each step has a named output and measurable completion condition in STATUS.md.
Do not delay daily collection until every retrospective experiment is finished,
but do not deploy untested changes to the frozen collector to save time.

## B. Data and provenance requirements

| Dataset | Required records | Admission and evidence |
|---|---|---|
| Prices/volume | Raw OHLCV, adjusted series, symbol, session, provider, retrieval time, available-at convention | Completed sessions, unique ordered keys, finite values, exact coverage; never pad missing bars |
| Corporate actions | Split/dividend events and adjustment convention where available | Explicit current-vintage limitations; adjusted proxy units do not certify broker accounting |
| Universe | Constituents, selection date/rule, inception/delisting coverage, ETF overlap | Freeze membership before evaluation; document survivorship and unavailable historical membership |
| Benchmarks | Cash convention, costed investable baseline, SPY reference, optional risk-free series | Same periods/capital/costs for primary comparisons; distinguish descriptive frictionless references |
| Features/labels | Schema, transformations, warm-up, information cutoff, target interval, maturity time | Versioned; learned transforms fit inside eligible training partitions only |
| Forecasts | Instrument/date/fold/model/seed, full probabilities, predicted class, training cutoff, eventual outcome | Keep all predictions, including abstentions and mistakes; never label unobserved targets as FLAT |
| Experiments | Hypothesis, parent trial, configuration, seed, budget, code/data hashes, status and runtime | Include failed/stopped trials; cache/reuse only when all provenance agrees |
| Portfolio | Target, decision time, order/fill IDs, cash, reservations, positions, realized/unrealized P&L, fees | Complete account ledger, including no-trade days, rejected orders and terminal open positions |
| Operations | Expected/actual run time, host/runtime, exit status, stages, failures, retries, input/output hashes | Durable receipts; distinguish scheduled, manually invoked, replayed and prospective runs |
| Optional inputs | Macro vintage/release time, news content/version/availability, crypto archive coverage | Eligible historical data or explicit exclusion; no zero-filled substitutes to suggest implementation |

Immediately reusable sources: `data/releases/2026-09-28-adaptive/market/`,
`experiment/*_predictions.parquet`, `*_folds.json`, `*_account.parquet`,
`*_events.json`, `returns.parquet`, `classification.csv` and `portfolio_metrics.csv`.
Earlier repaired experiments are in `data/repairs/`; inspect their manifests
before comparing. Legacy `data/research/` outputs are not automatically valid.

The existing adaptive experiment has 686 decision rows per instrument and full
three-class probabilities. Its recent audit has 54 account sessions but only
48 label-complete dates. These support more diagnostics now, without fabricating
new observations. Independent prospective evidence remains time-dependent.

## C. Experiment matrix

| ID / question | Design | Required outputs and completion condition |
|---|---|---|
| R1 Data audit | All assets, folds, calendars and exclusions | Coverage/missingness table, class counts, dataset flow diagram and usable sample totals that reconcile |
| R2 Forecasting value | Same folds/labels for prior, always-UP, simple model and eligible classifiers | Accuracy, balanced accuracy, per-class precision/recall/F1/support, log loss, Brier and paired differences with uncertainty |
| R3 Calibration/selection | Raw versus calibrated probabilities; same decision samples | Reliability diagrams, bin support, calibration uncertainty, confidence/coverage curves; no holdout-driven threshold selection |
| R4 Hybrid benefit | Trend, component ML, fixed mixture, adaptive gate, gate-off; same portfolio engine | Incremental prediction and net-P&L effects, gate utilization, expert disagreement, costs/exposure attribution; justify any unavailable variant |
| R5 Features | Fixed feature families removed/added under the same folds/budget | Paired ablations with uncertainty; refit each variant appropriately; no claims from importance rankings alone |
| R6 Adaptation/data efficiency | Predeclared history lengths/refit cadences/training fractions on development folds | Train/validation learning curves, model-age/shift effects, training cost and selection rationale; do not tune on final/pilot outcomes |
| R7 Generalization | Time blocks plus an eligible instrument/sector transfer design | Per-period/per-instrument/regime results and support; distinguish single-stock models from genuinely transferable pooled models |
| R8 Economic value | Costed cash/equal-weight/trend and candidate portfolios, shared constraints | Net/gross returns, CAGR, Sharpe convention, Sortino, drawdown/duration, turnover, exposure, costs, fills and net closed-trade outcomes |
| R9 Robustness | Frozen decisions or frozen adaptive policy, as appropriate, under registered perturbations | Cost, delay, missing-bar, gap, liquidity/fill-assumption and parameter-neighborhood sensitivities; specify which decisions change |
| R10 Failure mechanisms | Predeclared representative and worst-case slices | Confusion examples, confident mistakes, loss concentration, missed trends/reversals, gate misfires and leakage controls; avoid cherry-picked narratives |
| R11 Efficiency/operation | Real pipeline, declared hardware/runtime and repeated measurements | Stage/end-to-end latency, memory, model size, request counts, retry/failure rates, cold/warm conditions and sample sizes |
| R12 Prospective replication | Registered algorithm, daily observations revealed in time | Decision ledger, independent shadow account, expected-session coverage, eventual labels and uncertainty; no backfilled pilot evidence |

Use the same information set for claims of architectural improvement. For
example, compare a fixed model with a rolling model using aligned targets,
features, dates and execution before attributing a difference to retraining.
Comparing the old screenshot with the new audit percentages does not do that.

Simple models are mandatory controls. XGB/LGBM/LSTM/MoE and supplementary data
are included only with a stated research question and valid implementation.
Registry entries must explain exclusion, failure or deferral; do not silently
drop a weak model. Use a staged budget instead of a full cross-product search:
one small development smoke run, a frozen bounded search, then finalist runs
with registered seeds and outer folds. Retain elapsed time and stopping reasons.

Statistical specification: choose a primary predictive endpoint and primary net
economic endpoint before selection; use paired, date-block inference that keeps
cross-sectional dependence and overlapping labels visible. Define block/HAC
choices and sensitivity checks in advance. Count the actual tested hypotheses
and explain family-wise/FDR controls. Exploratory slice discovery is not a new
confirmatory test. State effect-size thresholds, approximate power and minimum
evidence requirements. There is no universal accuracy threshold that proves alpha.

## D. Manuscript coverage contract

The manuscript needs substantive explanations, not tables pasted between short
status paragraphs. Describe the reason for each experiment and interpret its
effect size, uncertainty, practical consequence and limitation.

Required figure subjects, combined into panels where it improves readability:

1. System architecture and decision/label/fill availability timeline.
2. Dataset coverage, class distribution and retained/excluded observations.
3. Confusion matrices and the dominant error transitions.
4. Calibration and confidence-versus-coverage behavior, with sample counts.
5. Baseline/candidate differences with uncertainty across folds.
6. Equity, drawdown, exposure and cost contribution across matched periods.
7. Feature/component ablation effects.
8. Training-data/history learning curves and adaptation behavior.
9. Instrument/period/regime generalization heatmaps with support.
10. Cost/execution sensitivity and operational latency/reliability.
11. Prospective coverage and outcome panels once observations exist.

Required tables cover dataset accounting, feature definitions/availability,
model configurations and budgets, baselines, complete class reports, instrument
and fold summaries, portfolio performance, ablations, sensitivity, operational
tests and the claim-to-evidence map. Put long per-instrument tables in an appendix
and include machine-readable versions. Cite relevant primary research, not only
API documentation; distinguish comparable reproduced baselines from context-only
literature. Derive novelty only after reviewing the literature.

Generate every number in the abstract, prose, captions, tables and dashboard
from one versioned statistics source or enforce an automated equality check.
Preserve full precision in source artifacts and document display rounding.
PDF QA includes every page, all labels/units/legends, consistent sample counts,
cross-references, bibliography, table boundaries and no orphaned/filler pages.
Do not copy an assertion that all code is human-authored or original merely
because it appears in the reference paper.

## E. Test and acceptance matrix

| Area | Essential behavior checks | Evidence required |
|---|---|---|
| Accounting | Hand-calculated entry/exit, entry+exit costs, reductions/partial fills, flat prices, gaps, terminal holdings; shorts only in supported contract | Exact expected cash/positions/P&L; reconciliation within documented tolerance |
| Capital/risk | Pending-cash reservation, no spending pending sells, target deltas, concentration/group/turnover controls, drawdown and market drift | Invalid increases rejected; reductions/halts persist; no duplicate exposure |
| Causality | Append/change/remove future rows, mature labels only, purged boundaries, training-only scaling/selection/calibration/gates | Earlier admissible features/predictions/decisions invariant; explicit cutoff assertions |
| Data | Missing/duplicate/out-of-order bars, holidays/DST/early closes, partial bars, stale provider responses, revisions and splits | Batch fails before orders; no fabricated bars or silent stale fallback |
| Inference | Exact feature order/schema, NaN/Inf rejection, valid probabilities, full sequence warm-up, serialization and model age | Exact or justified numeric equivalence; no uniform prediction fallback |
| Policy parity | Same as-of stream through research and actual daily policy, including first decision, refits, weekly holds and holiday weeks | Matching models/targets/ledger states across the full contract, not ledger-only tests |
| Recovery | Kill process before/after plan, individual account commit, receipt and model write; retry after restart | Recover or halt explicitly; no double fills or inconsistent dual-account receipts |
| Concurrency/storage | Concurrent CLI/UI/scheduler invocation, stale writer, disk-full, truncated state, missing artifacts, backup restore | Durable locking/CAS/rollback evidence; recovery does not erase original receipts |
| Scheduling | Real current command, actual trigger, holiday skip, early close/DST, sleep/offline host, delayed/dropped job, retry deadline | Saved schedule plus actual trigger receipt; watchdog detects absent as well as failed runs |
| Refit lifecycle | At least two registered refit boundaries in a controlled replay; exact dependency/model provenance, failed retrain rollback | No future labels, same cadence as research, last qualified runtime or visible halt |
| Release gates | Changed source/config, unrelated old JUnit, empty/partial suite, missing critical tests, stale receipt, altered model | Gate refuses unsupported approval; evidence is tied to the tested version |
| Research statistics | Known synthetic null/positive/negative controls, benchmark mismatch, fees/sign errors, zero-variance edge cases | Recomputed metrics match oracles; inferential assumptions and undefined metrics are explicit |
| End-to-end product | Clean bootstrap → fresh reads → features → sealed model → signals → targets → ledger → persisted receipt → UI/report | Complete real-data local smoke run plus sequential fault-tested replay; manual/scheduled modes identified |
| Paper consistency | Changed underlying metric/count/split, removed figure source, stale abstract, broken reference | Build updates all dependent outputs or fails; every claim resolves to evidence |

Run the offline suite first, then appropriate public-provider integration tests
in a separate explicit job. Inspect `pytest.ini`: default invocations exclude
integration tests. Update obsolete tests to the declared provider capability;
do not suppress a critical failure to preserve a green count. Optional excluded
providers have a documented status and do not silently appear in model features.

Keep a test manifest with command, environment, source/data/config hashes,
test IDs and pass/fail/skip reasons. Record performance measurements separately
from correctness tests. Replay tests use isolated state, never the active pilot DB.
Reproduce the finalist experiment from its frozen source/snapshots in a clean
environment; document numeric tolerances and hardware-dependent differences.

## F. Executable collection schedule to implement

This section specifies the future schedule; no schedule was activated while
preparing the prompt. Discover existing jobs before adding one. Use one owner
for daily execution, with a persistent state directory and tested recovery.

| Trigger | Work | Required receipt / behavior |
|---|---|---|
| Before pilot | Verify frozen algorithm, dependency locks, source/data manifests, account capital and scheduler | Registered trial ID, runnable command, durable-state/backup path, actual trigger test |
| Every XNYS session close +30 min | Fetch completed bars; validate; infer/refit if due; process prior pending fills and current targets; reconcile; persist | Full daily receipt including all symbols, models, targets, marks, failures and runtime |
| Close +45 and +60 min, only on eligible transient failure | Bounded provider/job retries under same run/decision ID | No duplicate fills; stop on deterministic integrity/risk failures |
| Close +90 min watchdog | Check an expected receipt exists, is current, valid and reconciled | Detect missing jobs; notify only actionable failures/required input through supported user-facing channel |
| Next session pre-open | Verify prior close processing and unresolved failures; optional provider cross-check | No new strategy decisions or changed timing; halt/escalate unresolved inconsistency |
| Registered rebalance session | Recompute weekly targets through shared policy | Declare holiday-Friday handling and test both research/live; no mid-pilot rule change |
| Every 21 observed sessions from the registered refit anchor | Versioned model update using matured labels and fixed hyperparameters | Same anchor/cadence as research; drift monitoring cannot silently retune the pilot |
| Weekly, after data processing | Archive receipts, reconcile coverage, update operational summary, verify backup | No winner reselection or weekly tuning on pilot returns |
| Sessions 5, 21, 42 and 63 | Predeclared operational review | Coverage, failures, latency, risk breaches and protocol deviations; no optional stopping for profit |
| Each label endpoint | Attach matured outcomes to immutable earlier predictions | Keep provisional/unobserved labels distinct from finalized metrics |
| Pilot end plus final label maturity | Freeze evidence, compute registered outcomes and rebuild manuscript | Separate operational completion from statistical/economic qualification |

Use `America/New_York` exchange sessions and display local times in
`Asia/Hong_Kong`. Normal close+30 is 04:30 HKT next day during US daylight time,
05:30 HKT next day during standard time; early closes have earlier triggers.
Do not encode this as one fixed Hong Kong time. The scheduler must skip holidays
and verify exchange-session availability. A periodic wakeup may check the next
due event; the market-calendar guard controls execution.

For a first decision on **29 September 2026**, the current XNYS calendar gives:

| Milestone | US session | Close+30 in Hong Kong |
|---|---|---|
| 1 | 29 Sep 2026 | 30 Sep, 04:30 |
| 5 | 5 Oct 2026 | 6 Oct, 04:30 |
| 21 | 27 Oct 2026 | 28 Oct, 04:30 |
| 42 | 25 Nov 2026 | 26 Nov, 05:30 |
| 63 | 28 Dec 2026 | 29 Dec, 05:30 |
| Final t+6 label endpoint | 6 Jan 2027 | 7 Jan, 05:30 after receipt |

These dates assume all expected sessions complete; recompute for the actual
start and later exchange notices. The next-close execution means the first
decision is not an immediate fill. The pilot's last decision also needs six
further sessions for its five-session outcome. Do not promise statistical
significance from 63 observations. A missed expected session is recorded as
an outage/deviation and handled under the registered recovery rule; never
silently fabricate/backfill it or slide the end date until performance improves.

GitHub-hosted schedules can be delayed or dropped. A cache is not authoritative
account storage. If using Actions, prove durable state/release availability,
concurrency and restore behavior; the existing legacy workflow is insufficient.
If using a local scheduler, document the dependency on host uptime and test
sleep/restart behavior. Use the platform's supported scheduler tools when
applicable; report actual job identifiers and next events only after verification.
If an always-on host or paid storage is needed, prepare the exact requirement
before asking for that separate provisioning decision.

Primary operational references, checked for this brief:
- [NYSE calendars and trading hours](https://www.nyse.com/trade/hours-calendars)
- [NYSE official 2026 calendar](https://www.nyse.com/publicdocs/nyse/ICE_NYSE_2026_Yearly_Trading_Calendar.pdf)
- [GitHub workflow troubleshooting and schedule delays](https://docs.github.com/en/actions/how-tos/troubleshoot-workflows)
- [Bailey and López de Prado, Deflated Sharpe Ratio](https://doi.org/10.2139/ssrn.2460551)

## G. Final acceptance

The handoff must contain a runnable product, versioned evidence package,
substantial editable/PDF manuscript, actual test results, actual schedule
verification, claim map and continuation status. Every requirement is complete,
explicitly inapplicable with justification, or pending with a precise blocker
and next event. Required pending experiments prevent a claim that the full
study is complete. A null result does not block writing a thorough historical
paper, but it does block unsupported claims of predictive or economic superiority.
