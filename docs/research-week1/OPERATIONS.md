# Current collection and observation contract

**Superseded, 1 October 2026:** collection and observation now follow
[research-week1-r2/OPERATIONS.md](../research-week1-r2/OPERATIONS.md).
This document is the preserved contract for the halted parent trial. Do not
restart its owner, clear its halt, or monitor its now-retired heartbeat.

Read [STATUS.md](STATUS.md) for exact dates, paths and commands. The five-session
amendment replaces only the prior pilot duration and reporting scope. It preserves
the r2 runtime source, carried models, policy, costs, capital and risk rules.
The old 63-session artifacts remain historical evidence and must not be edited
or counted as a completed trial.

The sole collection owner is `local.ml-trade-engine.collector-v2`, installed in
`/Users/a/Library/LaunchAgents/local.ml-trade-engine.collector-v2.plist`.
Its current arguments target the `research-v2-runtime-20260928-week1` sealed release
and `data/paper/research-week1` workspace. The prior schedule is preserved in
`data/research-week1/superseded-schedule.plist`. Do not run the old installer.
There must be no second schedule or manual daily/tick/commission caller.

The CLI runs every 60 seconds under launchd and applies XNYS calendar guards.
Close+30 creates a decision receipt; +45/+60 retry eligible transient failures;
+90 checks for a valid receipt. Pre-open checks, weekly backups, accounting,
failure classification, write locking and restart recovery retain the r2 contract.
The fifth watchdog writes `milestones/5.json`. No 21/42/63 review or refit falls
within the amended window. No new targets are produced after US 5 October.

The six following outcome sessions retain pending-fill processing and hold-only
account marks. Their receipts are distinct from decisions. Final completion needs
all five decision receipts, five mature outcomes, six terminal receipts, reconciled
ledgers and no unresolved halt. A missing event does not move the fixed end date.

The `trading-pilot-evidence-watch` automation is an observer and reporter, changed
on 30 September 2026 from hourly to two checks per US trading weekday. During the
registered pilot, the open check is 21:30 HKT Monday–Friday (09:30 New York), and
the post-close check is 05:30 HKT Tuesday–Saturday (17:30 New York). The latter
runs after close+30 collection and the bounded retries, at the close+90 watchdog
window. There are ten scheduled observer starts per week instead of 168.
These Hong Kong times cover the current September–October pilot; any separately
registered extension across a US daylight-saving change requires a schedule
update. The launchd collector itself remains exchange-calendar aware.

It reads this active workspace, retains notification state under
`docs/research-week1`, and reports new actionable failures or the week-close/final
milestone in the existing Codex task. It may create versioned read-only analytical
reports outside the live workspace. It must not collect, tune, backfill, clear
halts, place orders or publish externally. Commissioning/replay counts stay separate.

## Boundary-check procedure

1. Read the current status and existing `observer-state.json`. Use the XNYS
   calendar and registered schedule to identify the relevant session: the HKT
   morning check belongs to the preceding US date. Skip exchange holidays and
   checks outside the open/post-close window; never collect missed observations.
2. Run the sealed read-only status command in STATUS.md. Check the active launchd
   owner, heartbeat freshness, unresolved halt and any new or changed event,
   receipt, outage or milestone records. At the open, confirm the pre-open check
   and prior close processing. After the close, confirm the due collection,
   retries and watchdog. If the watchdog is currently running at the boundary,
   allow one wait of at most 60 seconds and then inspect again; do not poll hourly.
3. For new evidence, verify provenance/hash integrity, session coverage and
   uniqueness, input freshness, account reconciliation and label maturity.
   Collection already performs the required data and accounting validation.
   Reuse source-bound qualification results while source/configuration is
   unchanged; do not rerun the full test suite or historical experiments every
   check. Additional tests must address a changed component or a new failure
   and use isolated state, never the live account. Record their purpose/results.
4. Retain observer-only notification state here. Notify only a new actionable
   failure, missing session, stale heartbeat, integrity issue, required action,
   valid five-session milestone or final maturity. Do not repeat the same halt
   under a new timestamp or treat commissioning/replay as prospective evidence.
   The 29 September missed-session halt is already reported; no repeated alert
   is needed while it is unchanged. The superseded workspace is not monitored.
5. At five valid decisions with no unresolved halt, use the new dated
   `research_week1.report --week-close` command in STATUS.md and prepare the
   manuscript operational update. Pending labels do not delay an interim paper.
   Use `--freeze` only with all five decisions, five mature outcomes, six terminal
   receipts, reconciled accounts and no unresolved halt. Report descriptive
   results and exact paths, without profitability or statistical-power approval.
   Pause this observer after reporting the complete final package.

The minute-level launchd timer is local Python and consumes no Codex model tokens.
Only the Codex observer cadence changed; this does not clear the existing halt
or re-register the missed pilot. A replacement needs a new prospective protocol
and reliable host availability; preserve the failed trial and its original dates.

An obsolete heartbeat in `data/paper/research-v2` is expected because that workspace
has been superseded. Do not treat it as an outage of the active pilot. After final
evidence has been reported, the observer pauses. After its final scheduled event,
the collector has no further decision/data-download events; its timer may still
publish idle status until explicitly unloaded.

Inspect the actual owner:

```sh
launchctl print gui/501/local.ml-trade-engine.collector-v2
```

Logs and durable records are under `data/paper/research-week1`: `scheduler/`,
`heartbeat.json`, `days/`, `attempts/`, `events/`, `outages/`, `milestones/`,
`outcomes/`, `terminal/`, `outcome-runs/`, `backups/`, and the two account databases.
Not-yet-created records are absent, not successful zero-error sessions.

Recovery retains the r2 exact-original-workspace-path requirement. Stop the single
owner before restoring a verified backup, preserve displaced state, and use the
active workspace's original path. Do not mix the old release with the new account.
The full unchanged failure and restoration rules are documented in
`docs/research-v2/OPERATIONS.md`, whose dates and release paths are superseded here.
