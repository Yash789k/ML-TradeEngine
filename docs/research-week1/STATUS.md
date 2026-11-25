# Five-session pilot — current status

**Superseded on 1 October 2026:** the user authorized a separately registered
replacement for tonight. Collection ownership has moved to
[research-week1-r2/STATUS.md](../research-week1-r2/STATUS.md). This trial remains
halted with 0 observations and two missed sessions; its release, workspace and
halt are preserved. The records and earlier schedule below describe the failed
trial and must not be used as the active schedule.

**30 September 2026 update:** the first registered session (US 29 September) was
missed while the host was unavailable. The collector is halted with **0/5** valid
decisions; the sealed status still reconciles both ledgers and reports external
orders disabled. The missed session cannot be backfilled or counted as observed.
Changing the observer cadence does not resume this trial. A replacement requires
a separately registered prospective start and reliable host availability.

28 September 2026. The user shortened the prospective pilot before any prospective
decisions were observed. One week means five XNYS trading sessions. The historical
study remains available for release; the shortened prospective study is descriptive.
No profitability, statistical power, sustained reliability or broker approval follows.

## Active configuration

- Trial: `PILOT-TREND-WEEK1-20260929`.
- Sealed release: `data/releases/research-v2-runtime-20260928-week1`.
- Release manifest SHA-256: `198b36b8abcfcabd5520fd9b50fccd3a3b0c0cf68c732e7269b850677d66df93`.
- Registered amendment: `data/research-week1/protocol.json`.
- Active accounts, snapshots and receipts: `data/paper/research-week1`.
- Collector: macOS launchd service `local.ml-trade-engine.collector-v2`.
- Installed schedule: `/Users/a/Library/LaunchAgents/local.ml-trade-engine.collector-v2.plist`.
- Observer: Codex task automation `trading-pilot-evidence-watch`, checking at
  21:30 HKT Monday–Friday and 05:30 HKT Tuesday–Saturday during this pilot.
- Dashboard: `app_week1.py`, served locally at http://127.0.0.1:8504 when running.

The prior 63-session release and workspace remain unchanged at
`data/releases/research-v2-runtime-20260928-r2` and `data/paper/research-v2`.
They are superseded, not completed. Their old heartbeat is no longer monitored.
Deployment remains cash; the trend shadow has $10,000 simulated capital.
All quantities are adjusted-return proxy units; external broker orders stay disabled.

## Schedule — all displayed times are Hong Kong

| Event | Date and time |
|---|---|
| Decision 1 (US 29 September) | 30 September, 04:30 |
| Decision 2 (US 30 September) | 1 October, 04:30 |
| Decision 3 (US 1 October) | 2 October, 04:30 |
| Decision 4 (US 2 October) | 3 October, 04:30 |
| Decision 5, last new decision (US 5 October) | 6 October, 04:30 |
| Week-close watchdog and operational review | 6 October, 05:30 |
| Final outcome / terminal account collection | 14 October, 04:30 |
| Final outcome watchdog | 14 October, 05:30 |

The collector checks due events every 60 seconds. Eligible transient failures retry
at 04:45 and 05:00; the watchdog checks at 05:30. On eligible sessions, pre-open
verification is at 21:00 HKT. The week-end backup follows the US Friday watchdog
on 3 October at 05:30 HKT. All events are derived from the exchange calendar;
`data/research-week1/schedule.json` contains the complete schedule.

The observer now checks only at the US open and the post-close watchdog window.
Its ten weekly checks replace 168 hourly checks. Data collection and inline
validation remain owned by the local launchd process, whose timer consumes no AI
tokens. The boundary procedure in [OPERATIONS.md](OPERATIONS.md) reuses existing
qualifications and limits extra tests to changed code or new failures. The next
observer check after this change is 30 September at 21:30 HKT. The registered
collection dates above remain preserved; they are not a claim that a halted
trial can still produce all five original decisions.

The saved automation and recurrence checks are recorded in
`data/research-week1/observer-cadence-20260930T060554Z/verification.json`.
Validation confirmed exactly two checks for each of 15 US weekdays, preserved
the other automation settings, and verified the collector schedule was unchanged.

The six tail sessions (US 6, 7, 8, 9, 12 and 13 October) issue no new targets.
They process outstanding simulated fills, mark accounts and attach mature labels.
The original forecast is `Close[t+6]/Close[t+1]-1`; changing the pilot duration
does not change that target. No five-session forecast is mature at the week-close
review. The operational paper can be released then with those outcomes pending.

The Mac must remain powered, awake, logged in and online. The collector is local,
not hosted by Overleaf or a cloud trading service. The observer also requires the
Codex app and host to be available. There is no new subscription or external order.

## Verification and next steps

The new release passed 29 source-bound tests: the 27 existing runtime cases plus
the exact five-session calendar and a complete isolated historical replay through
five decisions and six terminal receipts. The report passed seven checks for
week-close eligibility, final maturity and the five-date inference restriction.
These tests are engineering evidence, not prospective market observations.
See `data/research-week1/qualification.xml`, `report-qualified-tests.xml`,
`release-verification.json` and `installation.json`. The actual new launchd
commissioning run finished at 10:15:55 UTC, fetched 22 instruments and reconciled
both isolated accounts. `scheduler-verification.json` records that OS trigger,
timer re-entry, fresh heartbeat and status; prospective coverage remains **0/5**.
The old schedule has been replaced, with its original definition archived.

The current paper is in the existing [Overleaf project](https://www.overleaf.com/project/6aba2fa5d6050f063aaea85a),
with editable source under `output/overleaf/week1/` and PDF
`output/pdf/ML_TradeEngine_Research_Week1.pdf`. The active dashboard offers both.
Historical result tables remain unchanged; the paper explicitly records the
pre-observation protocol amendment and five-date descriptive scope.

1. Preserve the missed-session halt and failed trial. A replacement prospective
   trial needs a new registration and an available host; never silently move dates.
2. The boundary observer checks operational evidence. Only a valid five-session
   milestone permits the registered operational report and manuscript update.
   Report observed activity/failures even if results disappoint.
3. The paper may be shared as a historical study with a brief operational pilot;
   label the forecast outcomes pending. No destination or external publication is set.
4. After the six tail receipts and all five mature outcomes exist, freeze the
   descriptive evidence and prepare an outcome update. Do not retune or auto-promote.

Read-only status:

```sh
data/research-v2/clean-env/bin/python -B data/releases/research-v2-runtime-20260928-week1/scripts/collector_v2.py status --release data/releases/research-v2-runtime-20260928-week1 --workspace data/paper/research-week1
```

New dated reports (run from the repository root):

```sh
# Interim snapshot, allowed now
data/research-v2/clean-env/bin/python -B -m research_week1.report --output data/research-week1/reports/NEW-DATED-RUN
# After five valid decisions, before outcome maturity
data/research-v2/clean-env/bin/python -B -m research_week1.report --week-close --output data/research-week1/reports/NEW-WEEK-CLOSE-RUN
# Only after five decisions, five mature outcomes and six terminal receipts
data/research-v2/clean-env/bin/python -B -m research_week1.report --freeze --output data/research-week1/reports/NEW-FINAL-RUN
```

Each output must be a new directory outside the live workspace and sealed release.
The report validates provenance and reads under the collector's shared lock.
Five dates remain descriptive: no block-bootstrap p-values or economic approval.

## Independent gates

- Engineering: tested locally; actual week of availability remains unobserved.
- Historical edge: not approved; prior negative/mixed results unchanged.
- Prospective operations: halted; first registered session missed, 0/5 valid receipts.
- Prospective outcomes: pending five mature outcomes and six terminal receipts.
- Manuscript: historical study available; revised pilot schedule and interim outcomes explicit.
- Broker: disabled.
