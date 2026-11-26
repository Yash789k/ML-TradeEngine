# Adaptive release readiness - 28 September 2026

**The local research and simulated-execution product is implemented. Combined
profitable-trader readiness is NOT APPROVED.** The evidence does not permit
changing the latter decision merely to complete a checklist.

## Current evidence

- Fresh public data: all 22 instruments through 25 September 2026, the latest
  completed US session at retrieval. Separate raw/adjusted snapshots preserve
  old caches and results.
- New training: 726 purged rolling fits, sealed models refit for current use,
  no future label endpoints, fixed logistic/tree mixture and a past-data gate.
- Recent prediction audit: hybrid 37.59%, always-UP 35.89%, trailing prior
  40.63%. The hybrid also loses to the prior on log loss. Its improvement over
  always-UP has a 95% date-block interval of -2.27 to +5.30 percentage points.
- Recent net trading audit: selected trend -0.57%, ML -0.49%, hybrid -0.56%,
  comparable equal-weight benchmark +2.03%. All candidate adjusted p-values
  are 1.0. This is not an established trading edge.
- Shared ledgers: 8,927 simulated fills; maximum reconciliation residual
  $2.19e-11. The accounts stayed cash constrained.
- Verification: 310 offline tests passed; the final focused execution and
  release tests also passed. The 17 legacy provider integration tests were
  not run. Actual fresh public downloads succeeded for all 22 instruments.
- Reproduction: a second full run from archived source and snapshots matched
  every checked prediction, account history, metric, fold and event record
  exactly. The sealed release pins runtime code and core dependency versions.
- Prospective pilot: zero observed sessions, reserved from 29 September for
  63 completed sessions. No historical fill is counted as prospective.

The screenshot's older percentages are disappointing relative to its own
baselines. They should not be compared directly with the new percentages:
labels now start at the executable next close, periods differ, and the summary
aggregation differs. There is no useful universal target such as 80% accuracy.
Out-of-time baseline improvement, probability quality, net portfolio returns
and uncertainty are the relevant tests.

## Delivered product

`python trader.py daily` runs public-data intake, schema/freshness checks,
versioned inference and scheduled refitting, target construction, account
processing and durable daily receipts. `python -m streamlit run app.py` shows
readiness, account state, the comparisons, current probabilities and manuscript.

The deployment account is eligible for cash. A separate research shadow follows
the candidate selected before the recent audit: the trend rule. The hybrid is
retained for documented research, not silently substituted for the selected
candidate. All accounts are local simulations in adjusted total-return proxy
units. They are not broker accounts or executable share inventories.

The runtime checks artifact/source hashes. Each observed session has a durable
plan, so a crash after ledger commit can resume without duplicate fills. The
process rejects stale/missing observations, changed same-session plans and
missed pilot sessions. An advisory lock prevents two CLI runs from sharing a
workspace concurrently. The execution handler processes hold sessions without
unintended daily rebalancing and still enforces drawdown reduction.

## Decisions and remaining evidence

| Decision | Status | Evidence still needed |
|---|---|---|
| Tested local simulation | Approved within this scope | Continue operational observation |
| Data/models for this release | Approved | Recheck freshness each run |
| Historical strategy edge | Not approved | A new predeclared candidate and independent evidence |
| Prospective operation | Not approved | Complete 63-session pilot without backfilled decisions |
| Consistent profitable trader | Not approved | Net performance with credible uncertainty across independent periods |
| Research manuscript | Completed negative-result technical study | Human authorship review and venue-specific peer review |
| External broker trading | Not approved | Explicit authorization, actual timing and corporate-action adapter, reconciliation |

The pilot is an operational sample, not a promise that 63 sessions will prove
profitability. Changing the candidate or tuning on the new audit requires a
new experiment. Existing network-provider tests and remote scheduled workflows
are not represented as tested merely because the local suite passes.

## Evidence locations

- `data/releases/2026-09-28-adaptive/readiness.json`: machine-readable decisions.
- `verification.json` and `test-results.xml` in that directory: exact test counts
  and reproducibility status attached by the release-sealing command.
- `experiment/`: probabilities, fold cutoffs, classification and portfolio
  results, cost stress, targets, SQLite accounts and full event records.
- `data/releases/2026-09-28-reproduction/verification.json`: exact second-run
  comparisons using archived source and data.
- `data/paper/adaptive-v1/latest.json`: most recent actual daily receipt.
- `docs/ADAPTIVE_PROTOCOL_2026-09-28.md`: protocol recorded before fetching the
  July-September extension.
- `docs/ADAPTIVE_RESEARCH_PAPER.md` and
  `output/pdf/ML_TradeEngine_Research.pdf`: complete manuscript and figure.

No broker orders, external alerts or unattended scheduler were enabled.
