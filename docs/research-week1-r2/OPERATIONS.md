**11 October 2026: active collection moved to the registered three-session continuation.** See [research-week1-r3/STATUS.md](../research-week1-r3/STATUS.md). Two prior decisions and the missed-session halt are preserved; the old workspace is no longer scheduled.

# Replacement pilot: collection and boundary checks

Read [STATUS.md](STATUS.md) for exact paths and dates. The active release is
`data/releases/research-v2-runtime-20261001-week1-r2`; the sole active account
workspace is `data/paper/research-week1-r2`. Both `data/paper/research-week1` and
`data/paper/research-v2` are superseded and preserved. Their stale heartbeats and
old halts must not trigger alerts for this trial.

The sole collection owner is launchd `local.ml-trade-engine.collector-v2`. Its
60-second local Python timer consumes no Codex tokens and executes only due XNYS
events. The separate overnight caffeinate job only prevents sleep. The observer
must never invoke daily/tick/commission, backfill, clear a halt, refit, retune,
enable external orders, or publish. Keep frozen release source and models intact.

## Boundary-check procedure

1. At 21:30 HKT Monday–Friday, check the current US session's open. At 05:30 HKT
   Tuesday–Saturday, check the preceding US session's post-close watchdog. These
   times cover this October pilot before the next US daylight-saving transition.
   Validate the session against XNYS; skip holidays and off-window invocations.
2. Read `observer-state.json` here and run the sealed read-only status command in
   STATUS.md. Verify launchd targets the new release/workspace and check heartbeat
   freshness, halt state, ledger consistency and newly changed durable records.
   On the first open before any decision, no prior receipt is expected. Later
   opens check the prior session's pre-open receipt. Closing checks verify daily,
   eligible retry and watchdog records. If the watchdog is in progress, allow
   one wait of at most 60 seconds before checking again.
3. Verify new receipt provenance, hashes, unique sessions, input freshness,
   reconciliation and label maturity. Reuse source-bound qualification while
   code/config are unchanged. Additional targeted tests require changed code or
   a newly diagnosed failure and must use isolated state. No routine full suite,
   historical reruns or provider downloads by the observer.
4. Persist observer-only state here. Stay quiet on unchanged/non-actionable state;
   alert only new actionable failures, newly missing expected sessions, stale
   heartbeat, integrity issues, required user action, valid five-session milestone
   or final maturity. Deduplicate by incident, including consequences of an
   already-reported halt. The failed parent trial has already been reported.
   Do not silently restart this replacement if it fails.
5. On five valid decisions with reconciled ledgers and no halt, run the dated
   `research_week1_r2.report --week-close` command in STATUS.md and prepare a
   manuscript operational update for review. Labels remain pending at week close.
   Only use `--freeze` with all five decisions, five mature outcomes, six terminal
   receipts, reconciled accounts and no halt. Report descriptive results and exact
   paths, never profitability or statistical-power approval. Pause the observer
   after reporting the complete final package; preserve collector and evidence.

Commissioning and replay are engineering evidence and count as zero prospective
decisions. Tonight's isolated provider commissioning uses a separate directory
under `data/research-week1-r2`; it must never be merged into the live account.

The temporary stay-awake assertion expires at 2 October 06:00 HKT. Later sessions
require an available host. The OS cannot guarantee wake from a closed lid, logout,
shutdown or depleted battery. Notifications require the Codex app to be available.

Raw records: `heartbeat.json`, `halt.json` if halted, `scheduler/`, `days/`,
`attempts/`, `events/`, `outages/`, `milestones/`, `outcomes/`, `terminal/`,
`outcome-runs/`, `backups/`, plus the account databases. Missing records are absent
evidence. The unchanged r2 backup/recovery contract in `docs/research-v2/OPERATIONS.md`
requires restoration to the exact original workspace path after stopping the owner.
