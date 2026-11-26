# Version 2 local collection contract

**Current schedule amendment:** use [research-week1/OPERATIONS.md](../research-week1/OPERATIONS.md)
and its linked status. The active pilot now has five sessions and a separate
workspace/release. The dates and active paths below describe the superseded
63-session configuration; its runtime failure/recovery contract is retained.

The collector uses an isolated runtime package and a newly sealed release. It does
not import broker, email, credential or legacy live-execution modules. Deployment
holds $10,000 cash; a separate $10,000 account observes the fixed historical trend
candidate. Quantities are chained adjusted total-return proxy units, not broker
shares. Model probabilities remain research observations, not an approved alpha.

## Policy, timing and evidence

The pilot starts on 29 September 2026 and has 63 fixed expected XNYS sessions,
ending 28 December 2026. The startup models are the carried 25 September fits;
refreshes occur at anchor offsets +21 and +42 (28 October and 27 November).
The next +63 refresh is outside the fixed decision period. This is an explicit
v2 cadence, and no equality with the older 2024-anchored historical experiment is
claimed. Both v2 replay and the daily path call the same feature/model/target,
rebalance, account and persistence functions. A 45-session, two-instrument actual
model replay crossed two refit boundaries and the Thursday-before-Friday-holiday
rebalance, matching probability vectors, targets and both account states exactly.

Rebalancing occurs on the first decision and the final XNYS session of the ISO
week. Model inference, pending fills, marking and risk checks happen each session.
A decision can first fill at the next close. The five-session label is
`Close[t+6]/Close[t+1]-1`; provisional labels are absent, not FLAT. After the final
decision, six further outcome reads continue through 6 January 2027. Final data
maturity is therefore 7 January at 05:30 Hong Kong time, after the endpoint receipt.

Every daily snapshot preserves the public provider JSON, raw bars, corporate
action event payloads and adjusted bars with checksums and retrieval timestamps.
The collector fails on stale, missing, duplicate, nonfinite, misordered or partial
bars. Raw partial responses remain available for investigation. It never fills
missing observations. Current-vintage historical adjustments are a limitation;
proxy return chaining does not certify executable share or dividend accounting.

## Daily schedule

A single local macOS launch agent wakes every 60 seconds; XNYS sessions and their
actual close times govern due events. During DST the normal close+30 run is 04:30
HKT next day; standard time is 05:30. Early closes are earlier. Thanksgiving is
skipped. The local definition is `scripts/scheduler/local.ml-trade-engine.collector-v2.plist`.

| Event | Behavior |
|---|---|
| Close +30 minutes | Public reads, strict validation, due model refresh, immutable plan, pending fills, cash and shadow ledger, reconciliation, receipt |
| Close +45 and +60 | Retry only an eligible transient provider failure or interrupted job, same decision identity; no duplicate fills |
| Close +90 | Require a valid reconciled receipt; missing runs create actionable durable events and halt further decisions |
| Next session open −30 | Check preceding expected receipt; no new decisions |
| Last session of week, watchdog | Copy receipts/state; SQLite backup API; restored file hashes and integrity check |
| Observations 5/21/42/63 | Fixed operational review; no profit-dependent stopping or winner selection |
| Final six outcome sessions | Close +30 public reads, prior pending fills and hold-only marks, transient retries +45/+60 and +90 watchdog; no new alpha targets |

The valid-session count is computed from unique reconciled receipts bound to the
release, trial, universe, original plan and model/snapshot provenance. Commissioning
and replays never contribute. Status additionally compares persisted account state
with the latest completed receipt. A startup model, stale old JUnit file, changed
source/config, missing critical tests or altered artifacts cannot authorize a new
seal. The sealing command runs the required suite and checks source hashes both
before and after that run; source movement during testing rejects the seal.

## Failure and recovery

One advisory process lock covers CLI, UI and scheduler writes. Atomic JSON writes
fsync contents and directory entries. Plans precede account commits; each SQLite
account event commits atomically and retries use the same close identity. An
interruption after only one account commits is recovered from the immutable plan;
completed results are reused only after validation. A retry after receipt persistence
also finishes pending outcome attachment and latest-status publication.

A missed expected session is recorded as an outage. The fixed pilot end date does
not slide and no missed decisions or fills are manufactured. Deterministic integrity
failures and absent watchdog receipts persist `halt.json`. Do not delete it merely
to restore a green display: investigate state, record the protocol deviation and
register any changed pilot/recovery rule. Interrupted writes and bad/truncated state
fail visibly. Backups are first verified in a separate staging directory. Operational restores
must return to the exact original canonical workspace path because immutable plans
contain absolute snapshot/model paths; preserve the displaced original directory
as a separate archive. Stop the scheduler before using the restore command; rollback of software uses the prior untouched release and
its own state, never a v2 account with a v1 policy.

The 27 registered tests cover two real refits, future-row mutation, first/weekly
policy parity, six crash points, account idempotency, competing OS processes,
disk-full atomic replacement, source data mutation/missing sessions, DST/early
close/holiday times, transient-only retries, missed jobs, outcome-tail retries,
receipt tampering, ledger corruption, and backup/restore verification. These are
engineering results; they imply nothing about profitable returns.

## Availability limits and completion

This is a user launch agent on a laptop. It requires a logged-in, powered, awake
host and network access. It does not prevent sleep, wake the laptop, provide
redundant hosting or guarantee unattended market-hour availability. Scheduler
heartbeat gaps are durably recorded on resume, and overdue expected events are
marked missed instead of backfilled. The controlled replay tests simulate these
outages; no claim of a physical sleep/restart experiment or remote CI run is made.

The old GitHub workflow's `schedule` trigger was removed locally to eliminate a
second requested owner. Its manual legacy validation remains. No remote workflow
change was published. A read-only GitHub API check found workflow `286364394`
(`Daily Signal Engine`) in `disabled_inactivity` state; the saved response is
`data/research-v2/operations/github-workflows-readonly.json`. This is an inspection
of remote scheduler state, not a remotely tested run or a remote mutation. No external orders,
paid host, outbound message or public deployment was performed.

Operational review dates are 5 October, 27 October, 25 November and 28 December.
Their HKT close+30 times are respectively 6 October 04:30, 28 October 04:30,
26 November 05:30 and 29 December 05:30. The evidence cutoff cannot be complete
until future receipts and mature outcomes exist. Even 63 complete observations
would not automatically establish statistical power, economic edge or broker readiness.

## Verified installation and commands

The active owner is `gui/501/local.ml-trade-engine.collector-v2`, running the
sealed `research-v2-runtime-20260928-r2` command with bytecode writing disabled.
Its actual RunAtLoad process had parent PID 1 and wrote a trigger receipt at
2026-09-28T08:50:56.358915+00:00. It fetched all 22 instruments, validated
the 25 September completed session, prepared signals and targets, committed both
commissioning accounts and reconciled them to zero error. The OS subsequently
reported a second timer launch and previous exit code zero. These are actual
local commissioning events; they are not prospective or remote runs.

The cold batch took 33.274 seconds, including
30.108 seconds for retrieval and validation, with
22 public requests and no provider errors. Cold sample
size is one. Across 20 fixed warm 22-symbol batches, median total time was
1.798 seconds and empirical P95 was
4.159 seconds; P99 is unreported. Five
fixed SPY refits had median 0.424 seconds. These measurements
share one laptop with other research work and are descriptive engineering timing.

The actual 22-symbol commissioning account was also restored to its exact original
path from its verified backup, retaining the original under
`data/paper/research-v2/commissioning-before-restore-r2`; its receipt hash and both
ledger snapshots were unchanged. Eight separate actual SIGKILL experiments (before
and after plans, individual account commits, receipt persistence and partial/complete
model writes) recovered the same targets and account state without duplicate events.
Five altered-source/config/model/JUnit/critical-test cases were refused by the seal.
These local process tests do not claim physical power-loss or sleep testing.

Authoritative artifacts are `data/research-v2/operations/scheduler-verification-r2.json`,
`commissioning-restore-r2.json`, `latency-r2.json`, and `process-recovery-r2.json` in
that operations directory. Source-bound qualification is
`data/releases/research-v2-runtime-20260928-r2/test-evidence.json` (27 cases,
39.732 seconds). Active status reports zero prospective decisions, zero mature
outcomes, zero terminal closes and no halt. First expected decision remains
30 September at 04:30 HKT; readiness is incomplete until actual observations mature.

Run the frozen command from any working directory:

```sh
/Users/a/miniconda3/bin/python -B /Users/a/Downloads/ml-trade-engine/data/releases/research-v2-runtime-20260928-r2/scripts/collector_v2.py status --release /Users/a/Downloads/ml-trade-engine/data/releases/research-v2-runtime-20260928-r2 --workspace /Users/a/Downloads/ml-trade-engine/data/paper/research-v2
```

Use the same command with `verify` to validate the seal or `backup` to create a
separate verified backup. `daily` uses the exact same policy as the schedule and
must be inside the registered session window. `commission` uses isolated preview
state. Do not use preview or historical results to increment pilot progress.

Inspect the owner without changing its state:

```sh
launchctl print gui/501/local.ml-trade-engine.collector-v2
```

To deliberately stop this local schedule, unload only this owner:

```sh
launchctl bootout gui/501 /Users/a/Library/LaunchAgents/local.ml-trade-engine.collector-v2.plist
```

To resume the same unchanged sealed command, use `launchctl bootstrap gui/501`
with that plist. A missed pilot session remains an outage; resuming does not grant
permission to invent past observations. Keep both source-bound test records and
all original receipts when investigating or restoring state.


## r2 evidence corrections and restore contract

Independent review identified gaps in first-seal receipt/outcome validation and
backup semantics before any prospective observations. The r1 release stays sealed;
its commissioning-only state is preserved as `data/paper/research-v2-r1-commissioning`.
The corrected release has suffix `-r2` and uses the canonical registered pilot ID
`PILOT-TREND-V2-20260929`. Active state remains `data/paper/research-v2`.

R2 validates original decision fields, every persisted account mark, underlying
parquet/provider/model bytes, and unique mature outcomes linked to the exact
decision, trial, release, source snapshot, six-session endpoint and recomputed
return. It freezes the input snapshot before fitting, so interrupted model writes
cannot accidentally mix revised provider inputs. Status reads SQLite entirely in
read-only mode and reconstructs an in-memory account; missing event tables and
truncated databases remain untouched and are reported invalid.

The tail has six distinct `outcome_tail` receipts, excluded from the 63 decision
count. These fill existing pending orders and mark holdings without new alpha
targets. End-of-decision-period and end-of-label-period equity are separate
endpoints. If a halt prevents terminal accounting, valid earlier predictions can
still receive labels; the outcome-read receipt records the terminal failure and
readiness remains incomplete. A label-only salvage is not accounting recovery.

Backups include all root JSON control state (including halt), ordinary and terminal
receipts, input snapshots, outcomes, outcome-read attempts, milestones and outages.
The `restore` CLI requires `--backup-path` and `--archive`. Stop the launch agent,
restore to exactly the path recorded as `origin_workspace` in the backup receipt,
verify `status` and preserved archive, then resume the same sealed launch definition.
Copying a backup to a different path creates an archive, not a usable relocated
account. The restore function stages and checks the backup before moving current
state to the chosen new sibling archive. It never rewrites immutable receipt text.
