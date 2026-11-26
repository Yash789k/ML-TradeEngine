**11 October 2026: active collection moved to the registered three-session continuation.** See [research-week1-r3/STATUS.md](../research-week1-r3/STATUS.md). Two prior decisions and the missed-session halt are preserved; the old workspace is no longer scheduled.

# Five-session replacement — 1 October 2026

The user authorized preparation for tonight after the original pilot halted.
This is a separately registered trial. The failed 29 September trial and its
missing-session halt remain evidence; no decisions are backfilled or transferred.
Registration occurred during the US 1 October session, before its close, without
using current-session results to change the strategy, models or selection rules.

**Installed and verified at 22:11 HKT on 1 October:** the new launchd owner has
triggered successfully, its heartbeat is fresh, and sealed status reports no halt,
no missing sessions and 0/5 observations before the first due event. The next
collection is **2 October at 04:30 HKT**. The live dashboard is
http://127.0.0.1:8504, served by `app_week1_r2.py`.

## Active configuration

- Trial: `PILOT-TREND-WEEK1-R2-20261001`.
- Release: `data/releases/research-v2-runtime-20261001-week1-r2`.
- Manifest SHA-256: `12eefa88386aaa2929866554e51b0b818c113489ab12ddc4813a1591cc34906d`.
- Workspace: `data/paper/research-week1-r2`.
- Protocol and registry: `data/research-week1-r2/protocol.json`, `registry.json`.
- Sole collector: launchd `local.ml-trade-engine.collector-v2`.
- Observer: `trading-pilot-evidence-watch`, 21:30 HKT Monday–Friday and
  05:30 HKT Tuesday–Saturday. No hourly Codex checks.
- Deployment: cash. Trend research shadow: $10,000 simulated capital.
- The frozen 25 September models and original refit anchor are carried unchanged.
- External broker orders remain disabled. Five dates are descriptive only.

## Registered schedule — Hong Kong time

| Event | Time |
|---|---|
| Decision 1, US 1 October | 2 October, 04:30 |
| Decision 2, US 2 October | 3 October, 04:30 |
| Decision 3, US 5 October | 6 October, 04:30 |
| Decision 4, US 6 October | 7 October, 04:30 |
| Decision 5, US 7 October | 8 October, 04:30 |
| Five-session watchdog / interim review | 8 October, 05:30 |
| Final outcome and terminal account collection | 16 October, 04:30 |
| Final watchdog | 16 October, 05:30 |

The exchange calendar determines the actual events in
`data/research-week1-r2/schedule.json`. Collection runs at close+30, eligible
retries at +45/+60, and the watchdog at +90. No new targets are permitted in the
six tail sessions: US 8, 9, 12, 13, 14 and 15 October. These settle pending
simulated fills, mark accounts and mature existing labels. None of the five
forecast outcomes is mature at the interim review.

## Host availability

The user will leave the laptop open tonight. A separate, temporary launchd sleep
assertion `local.ml-trade-engine.awake-20261001` runs through **2 October, 06:00
HKT**. It prevents idle sleep; it is not a collector, and does not place orders.
It cannot overcome a closed lid, shutdown, logout, depleted battery or lost
network. Keep the Mac plugged in, open, logged in and online. The assertion expires
after tonight; later collection windows still require host availability.

## Verification and continuation

Preparation, test, provider-smoke, installation and scheduler verification receipts
are saved under `data/research-week1-r2`. Qualification and commissioning are
engineering checks, not prospective trading days. Progress is derived from valid
receipts in the new workspace; scheduled dates are not observations.

The 29 source-bound runtime/schedule tests passed, including retry/fault handling,
accounting, causal policy replay and the exact replacement calendar. Seven report
tests passed for incomplete labels, missing decisions, halts and maturity gates.
An isolated live-provider commissioning run fetched all 22 instruments through
30 September and reconciled both $10,000 accounts with zero residual. It made no
model refit or external order and contributes zero prospective observations.
All 10 runtime source files and 44 model files match the parent release. The
stopped failed workspace's 201 files were verified unchanged after installation.
The new dashboard rendered with no errors and displayed 0/5, no missed sessions,
and the new review dates. Its existing paper download is labeled with the earlier
schedule pending the next manuscript update.

Evidence: `qualification.xml`, `report-qualified-tests.xml`,
`provider-smoke-verification.json`, `failed-trial-preservation.json`,
`installation.json`, `scheduler-verification.json`,
`dashboard-render-verification.json`, and `overnight-awake-v2.json`.
One preparation attempt failed before evaluation due to a missing test staging
directory. Its source/artifacts and failure note are retained under `validation/`
and `preparation-attempt-1/`; the successful isolated qualification used
`validation-attempt-2/`. No failed result or old seal was overwritten.

Read-only status:

```sh
data/research-v2/clean-env/bin/python -B data/releases/research-v2-runtime-20261001-week1-r2/scripts/collector_v2.py status --release data/releases/research-v2-runtime-20261001-week1-r2 --workspace data/paper/research-week1-r2
```

New dated reports, always outside the live workspace and release:

```sh
# Interim snapshot; no prospective completion claim
data/research-v2/clean-env/bin/python -B -m research_week1_r2.report --output data/research-week1-r2/reports/NEW-DATED-RUN
# Only five valid decisions, reconciled accounts and no unresolved halt
data/research-v2/clean-env/bin/python -B -m research_week1_r2.report --week-close --output data/research-week1-r2/reports/NEW-WEEK-CLOSE-RUN
# Only five decisions, five mature outcomes, six terminal receipts and no halt
data/research-v2/clean-env/bin/python -B -m research_week1_r2.report --freeze --output data/research-week1-r2/reports/NEW-FINAL-RUN
```

Historical edge remains unapproved. The existing manuscript is available with
the previous schedule; a prospective update must disclose this replacement and
the failed earlier trial. No external publication is authorized by this restart.
