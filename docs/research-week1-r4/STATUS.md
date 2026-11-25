> **Superseded for future collection on 21 October 2026.** The active fifth-observation continuation is documented in [research-week1-r5/STATUS.md](../research-week1-r5/STATUS.md). Preserve all earlier evidence and halts.

# Active interrupted pilot — 14 October 2026

The user requested “can we continue tonight.” This separately registered r4
continuation collects three decisions for **US 14, 15 and 16 October**.
Registration occurred during the first session, before its close; the model,
policy and costs were carried unchanged. Earlier inspected evidence is disclosed.

Two verified r2 decisions from US 1 and 2 October remain a separate failed
account segment. R2 missed 5, 6 and 7 October. R3 recorded zero decisions and
missed US 12 and 13 October; its planned 14 October decision was superseded
before its window. All earlier releases, receipts and halts are preserved.
Earlier week1 failures remain documented in their original workspace.
This is an interrupted five-observation study, not five uninterrupted successes.

R4 starts with fresh $10,000 simulated accounts. No positions, pending orders or
fills cross the gap. Cash deployment, trend shadow, 22 instruments, models and
costs remain fixed. No scheduled refit occurs within the three decisions.
Report segment returns separately. External broker orders remain disabled.

## Runtime and evidence

- Active release: `data/releases/research-v2-runtime-20261014-week1-r4`.
- Workspace: `data/paper/research-week1-r4`.
- Trial: `PILOT-TREND-WEEK1-R4-20261014`.
- Protocol and registration: `data/research-week1-r4/protocol.json` and `registry.json`.
- Source-bound qualification: `qualification.xml`, `qualification.log`, and
  `release-verification.json` under `data/research-week1-r4`.
- Installation and actual scheduled preview verification are recorded separately
  under `data/research-week1-r4`; never infer success from a planned schedule.
- Commissioning and historical replay count as zero prospective decisions.
- Progress must come from the sealed status command and unique valid receipts.

The only runtime change allows one commissioning preview before the first
registered session closes, including same-day installation. It uses the last
completed market session and separate accounts. It cannot create pilot evidence.

## Verified activation

At 14 October 21:41:46 HKT, the sealed status was unhalted with consistent accounts
and a fresh launchd heartbeat. One collector owner targets r4. The manifest is
`9f23d72fc0b892ad9af70efd2fbcdb71e2d3469f5d9f796ff40f76f52c3ec810`.
All 33 source-bound runtime checks and 11 report checks passed. The actual
launchd commissioning completed at 21:39:47 HKT using the last completed US
13 October session: all 22 instruments fetched, both preview accounts reconciled,
zero training and zero prospective decisions. See `scheduler-verification.json`.

The finite sleep guard and read-only dashboard are active. The dashboard render
had no exceptions and shows 2 archived + 0/3 new. The existing boundary observer
now targets r4; its frequency is unchanged. All 216 preserved r3 files and all
490 archived r2 files matched their inventories. Models are byte-identical.

The Mac was still on battery at verification. It must be plugged in and remain
open, logged in and online tonight; successful commissioning does not prove
future host availability or count as a pilot day.

## Schedule, Hong Kong time

| Collection | Time |
|---|---|
| US 14 October decision | 15 October, 04:30 |
| US 15 October decision | 16 October, 04:30 |
| US 16 October decision | 17 October, 04:30 |
| Interrupted five-observation interim review | 17 October, 05:30 |
| Final tail/label collection | 27 October, 04:30 |
| Final watchdog | 27 October, 05:30 |
| Sleep guard expires | 27 October, 06:00 |

Each decision has transient retries at 04:45 and 05:00, then a 05:30 watchdog.
The six hold-only tail sessions are US 19, 20, 21, 22, 23 and 26 October.
No new targets are permitted during the tail. Prior outcome labels are derived
only from recorded snapshots in separate report artifacts, never old accounts.

Sole collector: `gui/501/local.ml-trade-engine.collector-v2`, a local 60-second
Python timer without AI-token use. Sleep guard:
`local.ml-trade-engine.awake-week1-r4`. Dashboard:
`local.ml-trade-engine.dashboard-week1-r4`, http://127.0.0.1:8504.
The AI observer retains only market-open/post-close checks.

**Keep the Mac plugged in, lid open, logged in and online.** The idle-sleep guard
cannot override lid closure, logout, shutdown or power loss. Actual macOS power
logs showed sleep across the prior collection windows, including clamshell
sleep; the guard's presence alone does not prove host availability. Display
sleep is allowed. No unattended physical wake guarantee is claimed.

## Read-only status and report commands

```sh
data/research-v2/clean-env/bin/python -B data/releases/research-v2-runtime-20261014-week1-r4/scripts/collector_v2.py status --release data/releases/research-v2-runtime-20261014-week1-r4 --workspace data/paper/research-week1-r4
```

Reports always use a new dated directory outside releases and workspaces:

```sh
data/research-v2/clean-env/bin/python -B -m research_week1_r4.report --output data/research-week1-r4/reports/NEW-DATED-RUN
# Three valid new decisions plus two verified archived decisions; consistent accounts, no new halt:
data/research-v2/clean-env/bin/python -B -m research_week1_r4.report --week-close --output data/research-week1-r4/reports/NEW-WEEK-CLOSE-RUN
# Additionally all five mature outcomes and six new terminal receipts:
data/research-v2/clean-env/bin/python -B -m research_week1_r4.report --freeze --output data/research-week1-r4/reports/NEW-FINAL-RUN
```

Historical edge remains unapproved. Qualification and commissioning are
engineering evidence, not prospective completion or profitability. The next
manuscript update must disclose all failed trials and interrupted segments.
No external publication is authorized.
