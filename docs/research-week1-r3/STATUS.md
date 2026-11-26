> **14 October 2026 update:** r4 is now the sole active continuation. See
> [active status](/Users/a/Downloads/ml-trade-engine/docs/research-week1-r4/STATUS.md)
> and [operations](/Users/a/Downloads/ml-trade-engine/docs/research-week1-r4/OPERATIONS.md).
> R3 is preserved with zero decisions, two missed sessions and its halt; its
> remaining 14 October window was superseded before close. Do not monitor its
> stopped heartbeat as an active failure. The two valid r2 decisions remain archived.

# Active interrupted pilot — 11 October 2026

The user instructed “So make it work.” The two verified decisions from US
1 and 2 October are retained. The failed 5, 6 and 7 October sessions remain
failures. Three new decisions are registered for **US 12, 13 and 14 October**.
This is an interrupted five-observation study, not five uninterrupted successes.

The new segment starts with fresh $10,000 simulated accounts. No old positions,
pending orders or invented fills cross the gap. Models, instruments, costs,
cash deployment and trend shadow are unchanged; no production refit or retuning.
Report portfolio returns separately for each segment. Broker orders are disabled.

## Verified setup and evidence

- Active release: `data/releases/research-v2-runtime-20261011-week1-r3`.
- Manifest: `7bfc89e8f58a97d61a3ec800aa4819480f1bb66dd653f3fcce9493dc4d501946`.
- Active workspace: `data/paper/research-week1-r3`.
- Trial: `PILOT-TREND-WEEK1-R3-20261012`.
- Registration, qualification and installation: `data/research-week1-r3/`.
- Prior release and workspace: `research-v2-runtime-20261001-week1-r2` and
  `data/paper/research-week1-r2`; stopped, preserved with its halt and 490 files.
- 31 source-bound runtime tests passed, including three-decision/six-tail
  replay, configured milestone, faults, recovery and finite sleep-guard expiry.
- Eight report tests passed, including a label-maturity oracle. Actual launchd commissioning fetched all
  22 instruments; inspect `scheduler-verification.json` for the verified receipt.
- Commissioning counts as zero prospective observations. At installation,
  progress is **2 preserved + 0/3 new**, not a new completed trading day.

## Schedule, Hong Kong time

| Collection | Time |
|---|---|
| US 12 October decision | 13 October, 04:30 |
| US 13 October decision | 14 October, 04:30 |
| US 14 October decision | 15 October, 04:30 |
| Interrupted five-observation interim review | 15 October, 05:30 |
| Final tail/label collection | 23 October, 04:30 |
| Final watchdog | 23 October, 05:30 |
| Sleep guard expires | 23 October, 06:00 |

Each collection has bounded transient retries at 04:45 and 05:00 and a watchdog
at 05:30. Six US tail sessions (15, 16, 19, 20, 21, 22 October) only settle
pending simulated fills, mark accounts and mature outcomes; no new targets.
Old decision labels may be derived in separate report artifacts from the new
recorded snapshots after their endpoints. Old receipts and accounts stay intact.

Sole collection owner: `gui/501/local.ml-trade-engine.collector-v2`, a 60-second
local Python timer, with no AI-token use. Sleep guard:
`local.ml-trade-engine.awake-week1-r3`. Dashboard:
`local.ml-trade-engine.dashboard-week1-r3`, http://127.0.0.1:8504.

**Keep the Mac plugged in, lid open, logged in and online.** The finite
`caffeinate` assertion prevents idle sleep through the whole registered period,
including after relogin. It cannot overcome lid closure, shutdown, logout,
depleted battery or lost network. Physical sleep/wake has not been claimed tested.

## Commands

Read-only active status:

```sh
data/research-v2/clean-env/bin/python -B data/releases/research-v2-runtime-20261011-week1-r3/scripts/collector_v2.py status --release data/releases/research-v2-runtime-20261011-week1-r3 --workspace data/paper/research-week1-r3
```

New dated reports outside all workspaces/releases:

```sh
data/research-v2/clean-env/bin/python -B -m research_week1_r3.report --output data/research-week1-r3/reports/NEW-DATED-RUN
# Only 3 valid new decisions plus 2 verified archived decisions, consistent new accounts and no new halt:
data/research-v2/clean-env/bin/python -B -m research_week1_r3.report --week-close --output data/research-week1-r3/reports/NEW-WEEK-CLOSE-RUN
# Additionally all 5 decision outcomes and 6 valid new tail receipts:
data/research-v2/clean-env/bin/python -B -m research_week1_r3.report --freeze --output data/research-week1-r3/reports/NEW-FINAL-RUN
```

Historical edge remains unapproved. Engineering qualification and commissioning
do not establish prospective completion or profitability. The manuscript still
needs an operational update disclosing both failed trials and this interrupted
design; no external publication is authorized.
