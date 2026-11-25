> **Superseded on 22 October 2026:** r5 failed with zero valid decisions after rejecting inconsistent JPM OHLC data. Its halt and evidence are preserved. Active collection is now [r6](../research-week1-r6/STATUS.md); the material below records the earlier r5 setup.

# R5 collector and observer procedure

Read STATUS.md. The sole collector is `local.ml-trade-engine.collector-v2`, targeting the sealed r5 release and `data/paper/research-week1-r5`. All earlier workspaces are preserved and superseded. Ignore their stopped heartbeats, preserved halts and cancelled future dates as active alerts.

The observer runs twice per eligible XNYS session, approximately five minutes after market open and five minutes after the close+90 watchdog. During this pilot those are 21:35 HKT Monday–Friday and 05:35 Tuesday–Saturday. Use the sealed calendar; eligible windows are open through open+15 minutes, and close+90 through close+105 minutes. Skip holidays/off-window wakes, retaining the last actual inspection timestamp. The five-minute offset reduces early scheduler delivery; it is not an hourly poll.

At each eligible check, run the sealed read-only status command from STATUS.md. Inspect new durable receipts, input/snapshot hashes, heartbeat (stale after 300 seconds), halt, account reconciliation, watchdogs, milestones, the loaded collector owner and finite sleep guard. Retain observer-only hashes, notification IDs and state in `observer-state.json` here. Recovered gaps outside required events are non-actionable when no required collection is missing. Deduplicate consequences of an already-reported halt. Allow one wait of at most 60 seconds if the watchdog is still writing.

Launchd alone collects and validates data. The observer must not run daily/tick/commission, clear halts, backfill decisions, refit, retune, enable broker orders or publish. A new failure needs a disclosed recovery decision. Reuse source-bound qualifications; run targeted tests only for changed source or a new failure that needs investigation.

Count four verified archived decisions (r2: 1–2 October; r4: 14–15 October) plus unique valid r5 decisions (target one, US 21 October). Commissioning, replay, follow-up price downloads and terminal mark receipts are not additional decision days. Prior account returns must never be spliced into the fresh account. Preserve the original week1 failure, r2 misses 5–7 October, r3 misses 12–13 October/zero decisions, and r4 miss 16 October.

Notify only a new actionable failure, required user action, the valid one-new-decision milestone, or final maturity. Otherwise stay quiet.

At one valid r5 decision, consistent accounts, no active halt and four verified archived decisions, generate a new dated `research_week1_r5.report --week-close` report and prepare the manuscript operational update for review. Labels may remain pending. Use `--freeze` only after all five outcomes mature, six valid r5 terminal receipts exist, and the accounts reconcile without an active halt. Reports must keep segment returns separate, disclose all failures and avoid p-values, profitability approval or claims of continuous operational success. Pause this observer after reporting the completed final package.

The six tail sessions are US 22, 23, 26, 27, 28 and 29 October, ending 30 October at 04:30 HKT. They may settle simulated fills and mark accounts but never create new targets. The guard expires 30 October at 06:00 HKT. The Mac must stay powered, open, logged in and online; the guard does not guarantee physical wake or connectivity.
