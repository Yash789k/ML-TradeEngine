> **Superseded on 27 October 2026:** collection now follows [research-outcomes-r7/OPERATIONS.md](../research-outcomes-r7/OPERATIONS.md), under the user's request to restart outcome dates from tonight. R6 is preserved and no longer writes. Ignore its stopped heartbeat and cancelled future tail dates. The procedure below is retained historical evidence; do not invoke its shutdown/report gates against the new outcome-only owner.

# Archived R6 collector and observer procedure

Read STATUS.md and PROVIDER_INCIDENT.md. The sole collector is `local.ml-trade-engine.collector-v2`, targeting the sealed r6 release and `data/paper/research-week1-r6`. Earlier workspaces are preserved and superseded. Ignore their stopped heartbeats, retained halts and cancelled future tail dates as active alerts.

The observer runs twice per eligible XNYS session, approximately five minutes after market open and five minutes after the close+90 watchdog. During this pilot these are 21:35 HKT Monday–Friday and 05:35 Tuesday–Saturday. Use the sealed calendar: eligible windows are open through open+15 minutes and close+90 through close+105 minutes. Skip off-window/non-session wakes, preserving the last actual inspection timestamp. No hourly AI checks.

Run the sealed read-only status command in STATUS.md. Inspect new durable receipts, snapshot hashes, heartbeat (stale after 300 seconds), halt, reconciliation, watchdogs, milestones, loaded owner and finite sleep guard. Keep observer-only hashes and notification IDs in observer-state.json. Deduplicate consequences of an already reported halt. A recovered gap outside required events is non-actionable when no required receipt is missing. Allow one wait of at most 60 seconds if the watchdog is writing.

Launchd alone collects and validates. Do not run daily/tick/commission, clear halts, backfill, refit, retune, enable orders or publish. The new provider retry rule rejects an inconsistent latest OHLC bar and permits only the existing two bounded retries; persistent invalid data still prevents a decision. Reuse the source-bound qualification; add targeted tests only for a new failure or changed code.

Count four archived decisions (r2: US 1–2 October; r4: US 14–15 October) plus unique valid r6 decisions (one registered: US 22 October). Commissioning, replay, diagnostic downloads and terminal marks are not decision days. Preserve r5's rejected US 21 October response and zero-decision halt, r3's zero-decision failure, r2/r4 missed sessions and the original week1 failure. Never splice separate accounts' returns.

Notify only new actionable failures, required user action, the valid one-new-decision milestone or final maturity. Stay quiet on unchanged state. At one valid r6 decision, all four verified prior decisions, reconciled accounts and no active halt, generate a dated `research_week1_r6.report_20261023 --week-close` report and prepare an operational manuscript update for review. Labels may remain pending. Freeze only with all five mature outcomes and six valid new terminal receipts, reconciled accounts and no halt. Report descriptive findings and all failed trials, never statistical power or profitability approval. After verifying the completed final package, follow the authorized shutdown procedure below and pause the observer.

The six tail sessions are US 23, 26, 27, 28, 29 and 30 October. Final collection is 31 October 04:30 HKT; watchdog 05:30, observer approximately 05:35. The sleep guard expires 31 October 06:00 HKT. Keep the Mac powered, lid open, logged in and online. The guard prevents idle sleep; it cannot override lid closure, logout, shutdown or lost connectivity.

## Authorized shutdown after complete collection

On 24 October 2026 HKT the user requested: “If the collection is complete, stop the daily collections and scheduled tasks.” This authorizes shutdown once the full evidence gate passes. The five decision dates alone do not meet that gate; at this request the active continuation had 0/6 terminal receipts and no mature active outcome.

After a new `research_week1_r6.report_20261023 --freeze` report succeeds, verify its manifest, all five mature decision outcomes, six valid new terminal receipts, reconciled accounts and absence of an active halt. Then:

1. Confirm `local.ml-trade-engine.collector-v2` still points to the r6 release and workspace. If ownership changed, stop and report the discrepancy instead of unloading another trial.
2. Preserve the current collector and `local.ml-trade-engine.awake-week1-r6` LaunchAgent plists in a new dated directory under `data/research-week1-r6/shutdown/`, with their original paths and hashes. Unload those two services using `launchctl bootout gui/$(id -u) PATH`. Once unloaded, move their plists out of `~/Library/LaunchAgents` into that preservation directory so login cannot restart them. Do not delete evidence or modify the frozen release, configuration, accounts or halt files.
3. Verify both services are absent from launchd, the two plists are absent from LaunchAgents, and the preserved copies match their original hashes. Save a dated shutdown receipt with the final report path/hash, service results, preserved plist paths and the user's authorization. Keep the read-only dashboard available.
4. Pause `trading-pilot-evidence-watch` through the automation tool, preserving its fields. Report the final evidence package and the verified shutdown. Do not claim shutdown if unloading, preservation or pause is unconfirmed.

If final completeness or integrity checks fail, keep the outcome collection/observer active and report only actionable failures. This request does not authorize manufacturing missing outcomes, clearing halts or treating an incomplete study as complete. No additional confirmation is needed when the completion conditions are met.
