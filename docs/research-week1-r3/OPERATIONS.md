> **14 October 2026 update:** r4 is now the sole active continuation. See
> [active status](/Users/a/Downloads/ml-trade-engine/docs/research-week1-r4/STATUS.md)
> and [operations](/Users/a/Downloads/ml-trade-engine/docs/research-week1-r4/OPERATIONS.md).
> R3 is preserved with zero decisions, two missed sessions and its halt; its
> remaining 14 October window was superseded before close. Do not monitor its
> stopped heartbeat as an active failure. The two valid r2 decisions remain archived.

# Observer and collection procedure

Read STATUS.md. The sole active collector is launchd
`local.ml-trade-engine.collector-v2`, targeting the r3 release and workspace.
All r2, week1 and research-v2 accounts are preserved, superseded workspaces.
Their stopped heartbeats and old halts are not new alerts.

The observer checks eligible XNYS market opens (21:30 HKT Monday–Friday) and
post-close watchdogs (05:30 HKT Tuesday–Saturday). Skip holidays and off-window
wakes; preserve the last actual check time when recording a skipped invocation.
The existing recurrence is a boundary schedule, not an hourly AI poll.

At each eligible check, use the sealed status command in STATUS.md; inspect
new durable receipts, input/snapshot hashes, heartbeat (stale after 300 seconds),
halt, account reconciliation, watchdogs and milestones. Confirm the loaded
launchd owner targets r3 and the finite sleep guard is active before expiry.
Retain observer-only state in `observer-state.json` here. Treat recovered gaps
between required events as non-actionable when no session was missed.

Launchd alone downloads data, commits decisions and validates accounts. The
observer must not run daily/tick/commission, backfill, clear halts, refit, retune,
enable orders or publish. A new failure requires a disclosed recovery decision,
not an automatic reset. Reuse source-bound qualification; additional targeted
tests require changed code or a new failure. No repeated historical/provider runs.

Progress is two verified archived decisions plus unique valid new decisions,
with a maximum new target of three. Commissioning and replays count as zero.
The archived segment remains a failed operational trial and its portfolio return
must never be spliced into the fresh account's return series.

Stay quiet unless there is a new actionable failure, required action, valid
three-session continuation milestone (five observations across both segments),
or final maturity. Deduplicate consequences of any already-reported halt.
Allow one wait of at most 60 seconds if a watchdog write is in progress.

At three valid new decisions with consistent new accounts and no new halt,
run the dated `research_week1_r3.report --week-close` command and prepare the
manuscript update for review. Pending labels do not delay the interim report.
Use `--freeze` only with all three new decisions, all five mature outcomes,
six terminal receipts and consistent new accounts without a new halt. The
preserved old halt is disclosed, never cleared or called successful.
Pause this observer after reporting the completed final package. Never approve
profitability or statistical power from five observations.

Inspect installation and scheduler verification under `data/research-week1-r3`.
The guard prevents idle sleep until 23 October 06:00 HKT; the user must keep the
Mac powered, open, logged in and online. Keep display sleep allowed. Physical
wake, lid closure, network recovery and paid remote hosting are not guaranteed.
