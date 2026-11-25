> **Superseded for future collection on 27 October 2026.** The user requested restarting outcome collection tonight. [Active outcome-only schedule](../research-outcomes-r7/STATUS.md): US 27–30 October and 2–3 November. R6's US 23/26 October outcome reads were missed; its remaining tail is cancelled before today's window. Five original prediction dates remain preserved. This workspace and its incomplete accounts are unchanged; stopped heartbeat is expected. Historical activation/interim status follows.

# Archived fifth-observation replacement — 22 October 2026

R6 successfully collected its registered **US 22 October** decision, completing at **23 October 04:31:09 HKT**. The 05:30:48 watchdog confirmed the valid receipt. It uses fresh $10,000 simulated accounts and the same frozen models, 22 symbols, costs, cash deployment and trend research shadow. External broker orders remain disabled.

**Current progress: 5/5 valid interrupted decision dates; 3/5 dates have mature outcomes; 0/6 active terminal receipts.** R2 observed US 1–2 October; r4 observed US 14–15 October. R5 produced zero observations after rejecting an inconsistent JPM provider bar on US 21 October. See [the incident and repair](PROVIDER_INCIDENT.md). R6 adds the US 22 October decision. All 22 instruments passed on the first attempt; no retry, refit or external order occurred. Both fresh accounts reconcile at $10,000, with shadow execution still pending. Every old trial, receipt, account and halt is preserved. This is an interrupted five-observation study across separate accounts, not five consecutive successful sessions.

## Completed interim milestone

The registered read-only report is `data/research-week1-r6/reports/week-close-20261023T0538HKT/`. It verifies the four archived receipts and the new r6 receipt, keeps returns separate and reports 66 labeled rows across three dates (US 1, 2 and 14 October). Outcomes for US 15 and 22 October remain pending. The review-only LaTeX insertion is `output/overleaf/week1-r6-update-20261023/operational-update.tex`; it has not been uploaded or published. No final freeze or profitability approval has occurred.

A reporting-only revision, `research_week1_r6.report_20261023`, fixes an inherited r5 tag for future r6 outcome rows. The original report source and current interim report are preserved; their current labeled rows are all archived r2/r4 rows and unaffected. Fourteen focused report checks passed. Use the revised module for future reports. The sealed collector and trading policy are unchanged.

## Failure and repair

R5 ran on time. The saved JPM opening price exceeded its daily high. Yahoo returned a corrected record by 22 October 12:34 HKT; the exact correction time is unknown. The old collector treated the rejection as permanent, so it never used the two scheduled retries. R6 adds a narrow transient classification for this specific latest-bar range error during fresh downloads. All data must still pass strict validation before a decision. No price is corrected locally; bad or historical/corrupt input is never accepted.

The repair was registered before tests and the new session. **43 source-bound runtime cases passed**, covering the original operational contract, the real invalid/corrected JPM responses, bounded recovery, persistent failure, unchanged failed snapshots and rejection before accounts are created. **13 report cases passed**. Models, policy, accounting and scheduling source remain unchanged; only the calendar validator and provider loader differ.

## Active release and evidence

- Trial: `PILOT-TREND-WEEK1-R6-20261022`.
- Release: `data/releases/research-v2-runtime-20261022-week1-r6`.
- Manifest SHA256: `24e4cdf4ec6eb6553bb8229457154d973957615263e092c373a5e33cbe41faba`.
- Workspace: `data/paper/research-week1-r6`.
- Registration, schedules, qualification and installation proof: `data/research-week1-r6/`.
- Sole collection owner: `gui/501/local.ml-trade-engine.collector-v2`.
- Sleep guard: `local.ml-trade-engine.awake-week1-r6`.
- Read-only dashboard: `local.ml-trade-engine.dashboard-week1-r6`, http://127.0.0.1:8504/.

Actual launchd commissioning started **22 October 12:39:04 HKT** and completed **12:39:34 HKT**, with all 22 symbols valid, both accounts reconciled at $10,000 and no model training. This isolated preview uses the last completed session and counts as **zero** prospective decisions. The live r6 state has no halt or missing session. Use the sealed status below for current progress; this section records activation evidence.

## Schedule — Hong Kong time

| Event | Time |
|---|---|
| AI market-open check | 22 October, approximately 21:35 |
| Fifth decision, US 22 October | **23 October, 04:30** |
| Eligible transient retries | 23 October, 04:45 and 05:00 |
| Collector watchdog / one-decision review | 23 October, 05:30 |
| AI post-close review | 23 October, approximately 05:35 |
| Six outcome-only sessions | US 23, 26, 27, 28, 29 and 30 October |
| Final outcome collection | **31 October, 04:30** |
| Final watchdog / observer | 31 October, 05:30 / approximately 05:35 |
| Sleep guard expires | 31 October, 06:00 |

Launchd checks the XNYS calendar every 60 seconds using local Python, which does not consume AI tokens. The AI observer retains only two boundary checks per eligible session. The six-session tail settles simulated fills, marks accounts and matures outcomes; it cannot create new strategy targets. All r5 future tail dates are superseded. Prior labels are derived into report artifacts from validated later snapshots, without rewriting archived accounts or splicing returns.

**Keep the Mac plugged in, lid open, logged in and online overnight.** At the post-close check the Mac was on AC power (70%, charging). The finite guard prevents idle sleep and allows display sleep. It cannot override physical lid closure, shutdown, logout, depleted power or a network outage. Upstream data can still fail validation; no successful future collection is guaranteed.

## Status and reporting commands

```sh
data/research-v2/clean-env/bin/python -B data/releases/research-v2-runtime-20261022-week1-r6/scripts/collector_v2.py status --release data/releases/research-v2-runtime-20261022-week1-r6 --workspace data/paper/research-week1-r6
```

Only after one valid new decision, all four archived decisions verified, reconciled new accounts and no active halt:

```sh
data/research-v2/clean-env/bin/python -B -m research_week1_r6.report_20261023 --week-close --output data/research-week1-r6/reports/NEW-DATED-WEEK-CLOSE
```

Only after all five mature outcomes and six valid new terminal receipts, with reconciled accounts and no active halt:

```sh
data/research-v2/clean-env/bin/python -B -m research_week1_r6.report_20261023 --freeze --output data/research-week1-r6/reports/NEW-DATED-FINAL
```

Use new output directories. Five interrupted observations support descriptive reporting only. Engineering recovery does not establish a trading edge or approve deployment/publication.

## Stop condition requested on 24 October 2026

The user authorized stopping the daily collector and scheduled checks when collection is complete. At the request, the five decision observations were complete but the active six-session outcome tail was still 0/6. Necessary outcome collection remains active. After a verified final evidence package, the observer will unload the collector and r6 sleep guard, preserve their plists outside LaunchAgents to prevent login restart, verify shutdown, and pause itself. The dashboard and all evidence will remain available. The detailed procedure is in OPERATIONS.md.
