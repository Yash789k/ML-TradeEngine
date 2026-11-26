# Acceptance against the execution brief

This document separates completed local verification from observations that cannot
exist yet. Test counts are job-specific; overlapping jobs are not added together.
The authoritative source-bound runtime receipt is the r2 release's
`test-evidence.json`; full-repository and UI job manifests are under
`data/research-v2/tests/`.

| Required area | Executed evidence | Scope and remaining limit |
|---|---|---|
| Repository, prior evidence, instructions | Baseline source archive/diff; 722-file release inventory; both PDFs and both task histories reviewed; root AGENTS.md incorporates proposed instructions | Existing work preserved. Historical claims are independently rechecked, not treated as new experiments |
| Registry and exposure | Frozen protocol, evaluation_exposure.json, R1–R12, controlled design and validation budget | Every historical period is consumed/exploratory; no relabeled holdout |
| Accounting | `test_hand_calculated_nav`, `test_fees_and_cash_reconcile`, `test_long_entry_partial_fill_restart_hold_reduce_exit`, diagnostics flat-price cost oracle, round-trip episode reconciliation | Long-only local proxy units; shorts explicitly rejected by current account |
| Capital and risk | Pending reservation, pending sell, concentration/group, turnover, gap, rejected-order, drawdown tests in `test_portfolio_execution.py` and r2 replay | Real broker margin, auctions and market impact outside declared contract |
| Causality | Future-prefix mutation, endpoint purge, supervised preprocessing, unavailable-label gate checks; actual two-refit r2 policy replay; controlled checkpoint boundary audit | Historical raw snapshots are current vintage, not historical arrival-time archives |
| Data and calendar | Stale/missing/revised inputs, duplicate/order admission, DST, early close, holiday-Friday, partial snapshot tests; separate 17-case public-provider integration job | Historical constituent/action tape unavailable. New collector saves raw provider responses/action payloads. Optional macro/news/crypto remain excluded |
| Inference | Schema/order/finite probability, exact source/model bytes, serialization, model age and warm-up checks | No uniform probability fallback; invalid batches halt |
| Policy parity | `test_policy_refits_holiday_future_invariance`; 45-session revealed-prefix replay with two refits; r2 unchanged daily/CLI/UI/schedule policy | Revised weekly and anchor conventions are versioned, not retroactively assigned to old historical returns |
| Recovery | Six transaction boundaries plus partial/renamed model process-kill experiments; partial-refit input freeze; common halt gate; terminal receipt/latest repair | Controlled SIGKILL and storage faults are not physical power-loss experiments |
| Concurrency and storage | OS lock contention, disk-full atomic write, truncated/missing database tables, actual original-path backup restoration and continuation, immutable receipt checks | Restoration requires registered original paths. An arbitrary relocated copy is only an archive |
| Scheduling | Actual launchd process/parent PID, successful r2 public-data commissioning, timer re-entry and exit0, recorded next event; simulated late/offline/missing runs, transient-only retry deadlines and watchdog | Logged-in awake local host required. No physical sleep/power-cycle or sustained unattended session-close run yet. Remote GitHub workflow inspected: disabled_inactivity; no remote mutation/run claimed |
| Refit lifecycle | Two controlled boundaries, source/dependency/model provenance, frozen snapshot before partial fits, replay equivalence | Actual pilot boundary observations must occur at their registered dates |
| Release gates | Altered source/config/model, unrelated JUnit, missing critical tests, forged/unrelated/duplicate/premature outcomes, missing underlying bytes and corrupted journals refused | A passing unrelated test file cannot qualify this release |
| Research statistics | Positive/null/negative block oracles, multiplicity hand calculation, benchmark alignment, fee signs and undefined cash ratios; independent scalar reconciliation | Conditional blocks and approximate power cannot guarantee a future effect |
| End-to-end product | Clean runtime verify; real launchd data→features→model→signals→targets→two persistent ledgers→receipt; separate Streamlit button integration with isolated state; browser rendering | All execution is local simulation. Deployment cash; frozen trend research shadow. External orders disabled |
| Controlled study | 848 registered training trials / 1,696 fits plus four pilot fits; 17 predictive and eight economic contrasts; ablations, calibration/gate separation, histories, cadence, six-target transfer | Larger legacy models explicitly excluded because their transformations/splits are not qualified under the bounded comparison; no inferiority claim |
| Clean reproduction | All 848 trials rerun from archived source under locked core dependencies; 352,800 forecast-view records exact; 40 portfolios / 2,520 account sessions match ledgers, targets, events and economic inference | Records include repeated views: only 5,544 unique base instrument-date observations across 252 outer dates |
| Paper consistency | Statistics-source hashes, generated prose/tables/figures, claim map; metric mutation, missing evidence/figure, partial experiment and broken-reference tests; every PDF page rendered/reviewed | Historical manuscript can be complete while prospective results remain interim |
| Prospective analysis and freeze | Read-only validated-report command; same-cost EW replay, recorded model prior, six terminal sessions; lock/read-only/path containment tests; refuses premature freeze | Final inference only after complete frozen observations, no repeated interim significance tests or automatic economic promotion |

## Time-dependent completion

The fixed pilot contains 63 expected decisions from 29 September through
28 December 2026, then six sessions for final fills/marks and outcome maturity.
Reviews occur at sessions 5, 21, 42 and 63. A missed session is a recorded outage;
no backfill or performance-driven date extension is permitted. The final label
endpoint is 6 January 2027; its close+30 receipt is due 7 January 05:30 Hong Kong.
These dates use the installed XNYS calendar and require attention to future
exchange notices. Even complete collection does not automatically establish power,
profitability, broker readiness or a publishable claim of superiority.

The current host must be kept awake and logged in during collection windows.
An always-on host would require separately provisioning and qualifying the same
sealed runtime and durable state; no such infrastructure has been claimed.

Historical point-in-time constituent/event-vintage data and real execution
microstructure are different information requirements. Waiting for this pilot
will not repair those missing historical archives. They are explicitly excluded
from current claims.
