# Registered period ended incomplete — 5 November 2026 HKT

The read-only closeout verified **2/6 scheduled outcome reads** (US 27 and 28 October) and **4/5 original decision-date outcomes**, comprising 88 dependent instrument-date rows. The five original predictions remain preserved. The US 22 October prediction still has no verified outcome in the saved r7 data. **No further collection or observation dates are registered.**

- US 29 October: an initial attempt recorded transient provider connection/read timeouts for GOOGL, GLD and DIA; no valid receipt followed.
- US 30 October and 2–3 November: no valid receipts or attempt records. Their missing-start causes are not established.
- The four missing-session watchdogs recorded required action. There is no active integrity halt. The final amended dataset gate fails, and the original paper-account tail remains incomplete.

[Incomplete report](../../data/research-outcomes-r7/reports/incomplete-closeout-20261105/report.json) and [verification proof](closeout-20261105/report-verification.json) preserve the actual observations without a final freeze. Report/output and input-receipt hashes passed, and sealed status revalidated snapshots and archived account reconciliation at the last actual marks. Independent verification found all **1,741 archived files unchanged**, including preserved failed trials and halts. Account returns remain separate by segment; no profitability approval is given.

The obsolete `trading-pilot-evidence-watch` AI automation was **deleted**, with its old configuration preserved in [closeout evidence](closeout-20261105/closeout.json). The collector and sleep-guard services remain loaded because the completion-dependent shutdown condition did not pass; their loaded state is not a future collection schedule. The dashboard remains available. No new collection, backfill, account change, model fit or broker order was performed. External orders remain disabled.

This was a one-time administrative closeout after all registered windows expired, not an eligible market-boundary observation. Recorded late/off-window observer deliveries had been skipped under the boundary-only rule, which delayed notification; those skips are not claimed as inspections. Any further measurement needs a separate registered recovery with the missing sessions retained. Do not use `--final-outcomes` for this incomplete dataset.

The schedule and activation status below are historical.

---

# Outcome collection restarted — 27 October 2026 HKT

The user requested new outcome collection dates starting tonight. Five original prediction dates remain preserved: US 1, 2, 14, 15 and 22 October. The new outcome-only collection is registered for **US 27, 28, 29, 30 October and 2, 3 November**. It creates no predictions, strategy targets, fills, accounts or model fits. External broker orders remain disabled.

The original R6 outcome reads on **US 23 and 26 October were missed**. Its remaining scheduled tail is superseded before today's collection window. Original account accounting remains incomplete at its last actual mark; subsequent historical price downloads cannot recreate missed fills. Original prediction entry/endpoint dates remain fixed, including the final prediction's US 23 October entry and 30 October endpoint.

The new registered completion condition is six valid scheduled price snapshots and all five original-date outcomes, with verified archived evidence and no active integrity halt. This completes the amended outcome dataset, while the original portfolio/operational completion gate remains incomplete. The prior interim report's three mature dates and all earlier results remain preserved. A restarted-dataset count of zero does not erase that evidence.

## Schedule — Hong Kong time

| US session | Collection | Watchdog | AI review |
|---|---|---|---|
| 27 October | **28 October 04:30** | 28 October 05:30 | 28 October 05:35 |
| 28 October | 29 October 04:30 | 29 October 05:30 | 29 October 05:35 |
| 29 October | 30 October 04:30 | 30 October 05:30 | 30 October 05:35 |
| 30 October | 31 October 04:30 | 31 October 05:30 | 31 October 05:35 |
| 2 November | 3 November 05:30 | 3 November 06:30 | 3 November 06:35 |
| 3 November | **4 November 05:30** | 4 November 06:30 | 4 November 06:35 |

The market opens tonight, 27 October, at 21:30 HKT. The observer checks at open+5 minutes and close+95 minutes, using America/New_York scheduling to follow the US daylight-saving transition. These are 21:35/05:35 HKT before the transition and 22:35/06:35 afterward. There are no hourly AI checks. The local collector's 60-second calendar timer uses Python and consumes no AI tokens.

Collection allows an initial start at close+30..40 minutes, then two bounded retry windows at +45..55 and +60..65. Invalid inputs remain rejected. A missing new session stays missing; a later download cannot count as its on-time receipt. Subsequent valid reads may still measure fixed historical outcomes. An integrity/deterministic provider failure stops further downloads until separately addressed.

**Keep the Mac plugged in, lid open, logged in and online overnight through 4 November's final review.** The finite sleep guard expires at 07:00 HKT that day. It prevents idle sleep, but cannot overcome lid closure, shutdown, logout or network loss.

## Runtime, qualification and evidence

- Sealed release: `data/releases/outcomes-r7-20261027`.
- New workspace: `data/paper/research-outcomes-r7`.
- Registration: `data/research-outcomes-r7/protocol.json` (registered before evaluation, provider preflight and activation).
- Sole collector: `local.ml-trade-engine.collector-v2`.
- Sleep guard: `local.ml-trade-engine.awake-outcomes-r7`.
- Read-only dashboard: `local.ml-trade-engine.dashboard-outcomes-r7`, http://127.0.0.1:8504/.
- 26 targeted cases passed: DST/calendar boundaries, fixed-horizon labels, future-price invariance, bounded retries, restart idempotency, missing windows, integrity rejection, account nonmutation, manual-tick rejection and six-session replay. Replays are qualification only.
- Live provider preflight validated all 22 instruments for the last completed US 26 October session. This manual diagnostic counts as zero scheduled reads and zero decisions.
- Installation/preservation: `data/research-outcomes-r7/installation.json`, `r6-preservation.json`, and `superseded/`.
- Five additional activation checks passed against the actual sealed release: source/config tamper rejection, premature final-report refusal and preservation of all six archived account files during reporting. **31 targeted checks passed in total.**
- Activation verified on 27 October: sole launchd collector, successful exit, heartbeat age 23 seconds, AC power, active caffeinate guard, all 1,741 files across r2–r6 workspaces unchanged, healthy dashboard with no AppTest exceptions/errors, and twelve correctly timed observer checks across the six sessions including DST. Proof: `data/research-outcomes-r7/activation-verification.json`.
- Initial read-only report: `data/research-outcomes-r7/reports/activation-20261027/`. New scheduled reads remain **0/6 before the first window**; preflight and replay are excluded. Use actual unique verified receipts for progress, never this static description.

```sh
data/research-v2/clean-env/bin/python -B data/releases/outcomes-r7-20261027/collector.py status
```

Only after six valid scheduled read receipts, five mature original-date outcomes, unchanged reconciled archived accounts and no active halt:

```sh
data/research-v2/clean-env/bin/python -B data/releases/outcomes-r7-20261027/collector.py report --final-outcomes --output data/research-outcomes-r7/reports/NEW-DATED-FINAL
```

The final report is a new amended outcome dataset. It does not freeze or approve the original incomplete account tail. Preserve separate segment returns, earlier failed trials and missed sessions. Five dependent observation dates support descriptive results, not statistical power or profitability approval. After verifying the final package, stop the collector and sleep guard and pause the observer, as the user previously requested. Keep the dashboard and evidence available.
