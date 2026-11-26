> **Superseded on 22 October 2026:** r5 failed with zero valid decisions after rejecting inconsistent JPM OHLC data. Its halt and evidence are preserved. Active collection is now [r6](../research-week1-r6/STATUS.md); the material below records the earlier r5 setup.

# Active fifth-observation continuation — 21 October 2026

The user explicitly requested: “Then register and activate the fifth collection for tonight and make sure everything is ready to run.”

R5 registers **one new decision for US 21 October**, before that market session opens. It uses fresh $10,000 simulated accounts and the unchanged frozen models, costs, 22-instrument universe, cash deployment and trend research shadow. No refit, retuning, old-account reset or backfill occurs. External broker orders remain disabled.

Four verified archived observations are retained: r2 on US 1–2 October and r4 on US 14–15 October. Their accounts and returns remain separate. R2 missed 5–7 October; r3 recorded zero decisions and missed 12–13 October, with its final planned date superseded before its window; r4 missed 16 October. The original week1 failure is also preserved. This is five interrupted observations across three account segments, not five uninterrupted successes. The active r5 decision remains unobserved until a valid receipt exists.

## Active release and state

- Trial: `PILOT-TREND-WEEK1-R5-20261021`.
- Release: `data/releases/research-v2-runtime-20261021-week1-r5`.
- Manifest SHA256: `67b1bba3337831b9e66cdd4f684a28ec25a63e693dcd254bbc6299d673a03fc0`.
- Workspace: `data/paper/research-week1-r5`.
- Registration, schedule and evidence: `data/research-week1-r5/`.
- Sole collection owner: `gui/501/local.ml-trade-engine.collector-v2`, using a 60-second local Python timer; this does not consume AI tokens.
- Sleep guard: `local.ml-trade-engine.awake-week1-r5`.
- Read-only dashboard: `local.ml-trade-engine.dashboard-week1-r5`, http://127.0.0.1:8504/.

All prior releases and workspaces are superseded for future collection. R4's future outcome sessions from US 21 October onward are cancelled by this disclosed ownership change; future prices for its observed predictions will come from the new segment's recorded snapshots. No old account is altered or resumed.

## Verified activation

All 33 source-bound runtime checks passed, including the exact one-decision calendar, milestone and full six-session terminal replay. The runtime source and frozen model files are carried unchanged from r4. Thirteen report checks passed, including rejection of duplicate archived observations and preservation of separate segment returns.

Actual launchd commissioning started at **21 October 12:35:57 HKT** and completed at **12:36:15 HKT**, using the last completed US 20 October session. It is an isolated preview and counts as **zero** prospective decisions. The sealed status confirms **0/1 new**, no missing sessions, no halt and consistent state. Four archived observations remain verified. See `scheduler-verification.json` for the detailed activation proof and current-power limitation.

## Schedule — Hong Kong time

| Event | Time |
|---|---|
| Fifth decision collection, US 21 October | **22 October, 04:30** |
| Transient-failure retries | 22 October, 04:45 and 05:00 |
| Collector watchdog / one-decision milestone | 22 October, 05:30 |
| AI observer and completion review | 22 October, approximately 05:35 |
| Six outcome-only sessions | US 22, 23, 26, 27, 28 and 29 October |
| Final outcome collection | **30 October, 04:30** |
| Final watchdog / observer | 30 October, 05:30 / approximately 05:35 |
| Sleep guard expires | 30 October, 06:00 |

The collector derives events from the XNYS calendar. It will not create new targets or refit during the six-session tail. That tail settles pending simulated fills, marks the new accounts and observes the outcome endpoint. Prior labels are derived only in report artifacts from validated snapshots; archived accounts remain unchanged.

The existing AI observer retains two checks per trading session, at market open and after the post-close watchdog. Its target times are shifted five minutes later (21:35 and 05:35 during this pilot) to reduce skipped checks caused by early delivery. Non-session/out-of-window invocations are skipped; unchanged halts are not repeatedly notified. Collection does not depend on the AI observer running.

**Keep the Mac plugged in, lid open, logged in and online overnight.** It was on battery at activation. The finite guard prevents idle sleep and allows display sleep; it cannot override lid closure, logout, shutdown, loss of power or connectivity. No future host-availability guarantee is claimed.

## Read-only status and continuation

```sh
data/research-v2/clean-env/bin/python -B data/releases/research-v2-runtime-20261021-week1-r5/scripts/collector_v2.py status --release data/releases/research-v2-runtime-20261021-week1-r5 --workspace data/paper/research-week1-r5
```

After one valid new decision plus four verified archived observations, reconciled new accounts and no active halt:

```sh
data/research-v2/clean-env/bin/python -B -m research_week1_r5.report --week-close --output data/research-week1-r5/reports/NEW-DATED-WEEK-CLOSE
```

Only after all five outcomes mature and six new terminal receipts reconcile without an active halt:

```sh
data/research-v2/clean-env/bin/python -B -m research_week1_r5.report --freeze --output data/research-week1-r5/reports/NEW-DATED-FINAL
```

Use new output directories. Neither operational completion nor five descriptive observations establish profitability or statistical power. Historical-edge approval remains denied. No external publication is authorized.
