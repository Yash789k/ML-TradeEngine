> **5 November 2026 closeout:** The registered schedule below has ended incomplete (2/6 reads; 4/5 outcomes). The obsolete AI observer was deleted, and no future dates are registered. Collector and guard remain loaded; the completion-dependent shutdown gate has not passed. See [current status](STATUS.md) and [closeout proof](closeout-20261105/closeout.json). These historical procedures do not authorize backfill, new dates, or a final-completion claim.

# Outcome-only restart: collection, observation and shutdown

Read STATUS.md. Registration precedes the first new session's close. The sole active collection owner is `local.ml-trade-engine.collector-v2`, targeting `data/releases/outcomes-r7-20261027/collector.py` and `data/paper/research-outcomes-r7`. Old r6 collection is superseded; preserve its 23/26 October missed reads, all original receipts/accounts and prior failed trials. Do not alert on old stopped heartbeats, old halts or cancelled future events.

## Observer

Check only a registered XNYS session's open..open+15 minutes or close+90..close+105 minutes. The observer schedule uses America/New_York (09:35 and 17:35 weekdays) to follow November DST. First check the sealed `window` command with the invocation timestamp; a null result means skip, record only the skipped invocation and preserve the last actual inspection time. Do not repeat interim milestones already reported. No hourly AI checks.

```sh
data/research-v2/clean-env/bin/python -B data/releases/outcomes-r7-20261027/collector.py window --at ISO-TIMESTAMP
data/research-v2/clean-env/bin/python -B data/releases/outcomes-r7-20261027/collector.py status
```

Within a valid window, inspect verified `reads/*/result.json`, the referenced snapshots, attempts/failures, watchdog events, halt, heartbeat (stale after 300 seconds), sole loaded collection owner and sleep guard. Status revalidates archived account reconciliation at the last actual mark, pinned source, all original forecasts and derived outcome prices. Check `data/research-outcomes-r7/r6-preservation.json` if an integrity discrepancy is suspected. Keep observer-only hashes, event IDs and notification records in `docs/research-outcomes-r7/observer-state.json`. Deduplicate alerts for one incident. Notify only new actionable failure, required user action or final amended completion. Ordinary progress remains quiet.

Launchd alone invokes tick. Do not run tick manually, issue new targets, change original forecast endpoints, refit, retune, write old ledgers, clear halts, fabricate on-time receipts, enable broker orders or publish. Reuse the source-bound qualifications; test only changed behavior or a new failure. The last required read is US 3 November, collected 4 November 05:30 HKT, with watchdog 06:30 and observer 06:35. The guard expires at 07:00 HKT.

The new collector can download later market history and measure an original prediction at its unchanged entry/endpoint. It records when those prices were actually obtained. This is delayed outcome measurement, not a reconstructed execution or a new prospective decision. There is no account-writing code path. Preserved account returns remain separate and stop at their last actual mark. Original operational completion remains incomplete even if all six new reads succeed.

## Reports and authorized completion shutdown

The human requested stopping collection and scheduled tasks when complete, then explicitly rescheduled outcome collection from tonight on 27 October. The registered amendment completes the outcome dataset when all six new read receipts and all five fixed-date outcomes validate, archived accounts remain unchanged/reconciled at their last actual marks and no new integrity halt exists. It does not relax or approve the old portfolio gate.

Generate a new dated report only after that gate passes:

```sh
data/research-v2/clean-env/bin/python -B data/releases/outcomes-r7-20261027/collector.py report --final-outcomes --output data/research-outcomes-r7/reports/NEW-DATED-FINAL
```

Verify `manifest.json` file hashes, report status `final_amended_outcome_dataset`, six receipts, five dates/110 instrument-date rows, all input receipt/snapshot hashes, all archived account checks and no active halt. Include every earlier failed trial and missed session from the pinned parent protocol, the original 23/26 October missed outcome reads, and current-vintage adjustment limitations. Never stitch account returns, claim profitability or mark original operational completeness approved.

After successful final verification:

1. Confirm the collector's arguments still point to the r7 release/workspace; never stop a different active trial.
2. Preserve bytes and hashes of `~/Library/LaunchAgents/local.ml-trade-engine.collector-v2.plist` and `local.ml-trade-engine.awake-outcomes-r7.plist` in a fresh dated `data/research-outcomes-r7/shutdown/` directory.
3. Unload each using `launchctl bootout gui/$(id -u) PATH`, then move the preserved plists outside LaunchAgents so login cannot restart them. Leave the dashboard loaded. Do not modify accounts, halts or evidence.
4. Verify neither collection/guard service is loaded, both original plists are absent, and preserved hashes match. Save a shutdown receipt with final report path/hash and the user's authorization.
5. Pause the existing `trading-pilot-evidence-watch` automation using its tool, preserving other fields. Report the final outcome package and verified shutdown. No extra confirmation is needed.

If the new gate fails, do not claim completion, silently add dates or rewrite the protocol. Preserve the actual failure and report what requires action. Necessary remaining registered downloads remain active unless separately instructed.
