> **25 November 2026 monthly progress review (`review_v1`, separate from collection):** a review console (`app_review.py`, port 8510; the sealed dashboard on 8504 is untouched), an evaluation package (`review_v1/`), saved results (`data/review-v1/`), seven test modules (`tests/test_review_v1_*.py`) and three documents ([progress report](../review-v1/PROGRESS_REPORT.md), [reproduction steps](../review-v1/REPRODUCE.md), [screen-recording brief](../review-v1/CODEX_VIDEO_PROMPT.md)) were added in new locations. Questions, periods, portfolios, costs and decision rules were registered in `data/review-v1/registration.json` (SHA-256 `abc146e5…`, one amendment) before any result was computed; the budget of five sandbox portfolios and two contrasts is spent. **Findings, all from saved statistics:** (1) the five registered decision receipts match their hashes and every recorded probability, trend flag, weight and order quantity reproduces from the recorded bars; (2) on 14,960 out-of-fold forecasts the calibrated mixture has log loss 1.0762 against 1.0641 for the class-frequency prior (difference +0.0121, 95% interval +0.0044 to +0.0194), and it is behind on the 88 recorded October rows as well; (3) a causal rolling two-state HMM on SPY passes its truncation, convergence and vintage checks and separates subsequent volatility, but as an exposure overlay neither registered Sharpe contrast is established (trend −0.32, Holm p 0.41; equal weight −0.16, Holm p 0.72); (4) on the previously inspected 2024 to September 2026 history the trend rule returned +35.0% (Sharpe 1.30) against +55.2% (Sharpe 1.40) for costed equal weight, not distinguishable by interval; (5) a simulated replay of the 41 sessions from 29 September to 24 November lost money in every invested portfolio and is too short for a Sharpe ratio; (6) sentiment is not an input to the pilot, and adding FinBERT columns for NVDA and SPY made log loss worse (+0.0462, interval +0.0103 to +0.0848). The replay and the dashboard's run button are labelled simulated and are not prospective evidence; recorded evidence remains 5 of 18 scheduled decision sessions. **Verification on 25 November:** `python -B -m review_v1.build all` completed with 15 of 15 manifest checks passed; `python -B -m pytest tests/test_review_v1_*.py -q` gave 114 passed, 0 skipped (the FinBERT comparison ran from the local model cache); `ruff check --select E9,F,B,PLE` on the new files reported no findings. The legacy and sealed-release suites were not re-run. **Not changed:** collector, launch agents, pilot workspaces, sealed releases, hash manifests, accounts, models; external orders remain disabled and no broker endpoint is called. **Continuation:** (a) build the proposal's label (next-open entry, cost- and volatility-scaled threshold) and rerun the classifier-against-prior comparison with `review_v1.classifier`'s statistics, registering it first; (b) move the collector to an always-on host and log the five portfolios in shadow for at least 63 sessions; (c) add an exposure-matched equal-weight benchmark; (d) test regime probabilities as gate inputs in the main learned-gate experiment. Any further portfolio variant needs a new registration.

> **25 November 2026 publication snapshot:** all local work through the 5 November closeout is committed on branch `fyp/ceci2-adaptive-ensemble` (seven commits on top of `f194283`) and pushed to `origin`; `main` is unchanged. Gitignored artifacts (`data/releases`, `data/paper`, `data/repairs`, model and parquet files) remain local only. Three items were left untracked because the repository is public: `docs/research-controller-v1/MEETING_SUMMARY.md`, `data/research-controller-v1/proposal-review-v1/` (contains a student ID) and the third-party `output/overleaf/.tools/tectonic` binary. An offline run from the repository root (`python -B -m pytest tests/ -q -p no:cacheprovider --ignore=tests/test_week1_r6_provider.py`) gave 506 passed, 11 failed, 18 deselected. The 11 failures and the ignored module are r3–r6 pilot tests that target their sealed-release layout and runtime (`scripts/power_guard.py`, release `runtime_v2`) rather than the repository root; each passed in its release's own `tests.xml` at sealing time and was not re-run there today. No collection, schedule, account, model or code change was made, and no prospective observation exists after the 5 November closeout.

> **5 November 2026 outcome closeout:** [R7 ended incomplete](../research-outcomes-r7/STATUS.md): 2/6 scheduled reads and 4/5 original-date outcomes verified; no future dates remain. US 29 October logged provider timeouts; US 30 October and 2–3 November have no attempt records. A [new incomplete report](../../data/research-outcomes-r7/reports/incomplete-closeout-20261105/report.json) preserves the results; all 1,741 archived files remain unchanged. The obsolete AI observer was deleted. Collector and sleep guard remain loaded because the completion-dependent shutdown gate failed; dashboard/evidence remain available and external orders remain disabled. No final dataset, operational completion or profitability approval is claimed. Earlier schedule entries below are historical.

> **30 October 2026 controller research (separate from collection):** [research preparation](../research-controller-v1/READ_ME_FIRST.md) aligns the supplied FYP proposal and reviewer feedback with current implementation. It includes a contextual-mixture formulation, isolated HMM ablations, proposed compute budgets, 25 method/control options and 12 recent-release entries (reviewed, not all implemented). An isolated registered 2025 probability-combination screen used 250 dates/5,500 dependent symbol-date observations and two meta-fits. Hedge and logistic stacking beat the fixed linear/tree mixture; neither established superiority to the trailing prior. [Results and exact verification commands](../research-controller-v1/SCREEN_RESULTS.md). Initial and independent tests cover label maturity, future-data invariance, baseline arithmetic and provenance; they do not qualify economic deployment. The proposal's next-open/cost-and-volatility label contract remains distinct from the screen's next-close/fixed-1% contract. [Alignment and next experimental design](../research-controller-v1/PROPOSAL_ALIGNMENT_AND_GAPS.md). The October rubric and additional tutor feedback are pending; no final monthly report or new production gate is claimed. The saved read-only r7 snapshot on 30 October records 2/6 outcome reads, 4/5 mature decision dates and missing US 29 October; next scheduled collection in that snapshot is 31 October 04:30 HKT. Collector, accounts and schedules were not changed by this research work. The separate PR question was withdrawn by the user.

> **27 October 2026 active outcome restart:** [research-outcomes-r7](../research-outcomes-r7/STATUS.md) now owns the six new outcome reads for US 27–30 October and 2–3 November, first collection 28 October 04:30 HKT and final collection 4 November 05:30 HKT. Five original predictions and their fixed horizons are preserved. R6's missed US 23/26 October outcome reads and unfinished account tail remain disclosed; later prices cannot recreate missed trades. No new targets, accounts, model fits or external orders. The earlier status entries below are historical.

> **23 October 2026 interim milestone:** 5 valid interrupted decision dates are now verified across r2/r4/r6. The r6 decision completed at 04:31:09 HKT; 3 dates have mature labels and 6 active outcome-only receipts remain pending. [Archived r6 status](../research-week1-r6/STATUS.md). The interim report and review-only LaTeX update are prepared; no profitability approval or final freeze.

> **Active collection as of 22 October 2026: [research-week1-r6](../research-week1-r6/STATUS.md).** R5 failed on an inconsistent JPM price response; four archived observations remain. The new qualified runtime permits bounded retries for this latest-provider-bar failure, preserves strict validation and all earlier evidence, and registers one future US 22 October decision. See [incident evidence](../research-week1-r6/PROVIDER_INCIDENT.md). External orders remain disabled.

> **Superseded for future collection on 21 October 2026.** The active fifth-observation continuation is documented in [research-week1-r5/STATUS.md](../research-week1-r5/STATUS.md). Preserve all earlier evidence and halts.

> **14 October 2026 update:** r4 is now the sole active continuation. See
> [active status](/Users/a/Downloads/ml-trade-engine/docs/research-week1-r4/STATUS.md)
> and [operations](/Users/a/Downloads/ml-trade-engine/docs/research-week1-r4/OPERATIONS.md).
> R3 is preserved with zero decisions, two missed sessions and its halt; its
> remaining 14 October window was superseded before close. Do not monitor its
> stopped heartbeat as an active failure. The two valid r2 decisions remain archived.

**11 October 2026: active collection moved to the registered three-session continuation.** See [research-week1-r3/STATUS.md](../research-week1-r3/STATUS.md). Two prior decisions and the missed-session halt are preserved; the old workspace is no longer scheduled.

# Research v2 execution status

**Active pilot, 1 October 2026:** the user authorized a distinct five-session
replacement after the first trial halted. The newly qualified collector is
running without a halt, with first collection due 2 October 04:30 HKT.
See [current status](../research-week1-r2/STATUS.md) for exact paths, evidence,
the overnight power guard and commands. Both prior pilots remain preserved.

**Superseded pilot schedule, 28 September 2026:** the user requested a one-week
pilot before any prospective observations. The current five-session release,
workspace, schedule and continuation commands are in
[research-week1/STATUS.md](../research-week1/STATUS.md). The 63-session material
below is preserved historical status, not the active collection commitment.

**30 September update:** the five-session replacement is halted after its first
missed session, with 0/5 valid decisions. Its Codex observer now checks twice per
US trading weekday instead of hourly; collection remains owned by launchd.
See the current [status](../research-week1/STATUS.md) and
[boundary-check procedure](../research-week1/OPERATIONS.md).

28 September 2026. The local paper-trading product and historical investigation
are implemented. The historical paper is complete for the available evidence;
the prospective study remains interim. External broker orders are disabled.

## What is runnable

The active immutable runtime is `data/releases/research-v2-runtime-20260928-r2`,
manifest SHA-256 `570372e203c6ef943deb37c2856a4ca7f0c3d34a168ac247d5b1cb2232916f43`.
The canonical trial is `PILOT-TREND-V2-20260929`. Deployment holds cash; a separate
trend shadow uses $10,000 of simulated capital and adjusted-return proxy units.
Accounts, snapshots, decisions, outcomes, backups and receipts persist in
`data/paper/research-v2`. The earlier r1 commissioning workspace is preserved at
`data/paper/research-v2-r1-commissioning`. Original v1 releases and user changes
remain preserved; all 722 inventoried sealed files rechecked unchanged.

The dashboard is `app_v2.py`, served at http://127.0.0.1:8504 while this local
server remains running. UI actions call the same frozen CLI as launchd.
The real-data UI commissioning test uses isolated state, never the pilot account.

From the repository root:

```bash
data/research-v2/clean-env/bin/python -B data/releases/research-v2-runtime-20260928-r2/scripts/collector_v2.py status --release data/releases/research-v2-runtime-20260928-r2 --workspace data/paper/research-v2
/Users/a/miniconda3/bin/python -B -m streamlit run app_v2.py --server.port 8504
```

Use `verify` instead of `status` to verify the seal. `commission` is an isolated
preview. Let the sole scheduler own actual decisions; `tick` only processes
eligible due events. Do not bypass halts or fabricate missed sessions.

## Verified schedule and remaining time

The active owner is `gui/501/local.ml-trade-engine.collector-v2`, installed in
`~/Library/LaunchAgents/local.ml-trade-engine.collector-v2.plist`. It checks every
60 seconds and uses XNYS exchange events. An actual r2 launch at 08:50:56 UTC,
parent PID 1, fetched all 22 instruments, produced signals/targets and reconciled
both commissioning ledgers. Timer re-entry and exit 0 were also verified.
The remote GitHub workflow was independently inspected and is
`disabled_inactivity`; no remote change or successful remote run is claimed.

- First decision: US 29 September; **30 September 2026, 04:30 HKT**.
- Retries: close+45/+60 only for transient failures. Watchdog: close+90;
  next-session pre-open verification, weekly backups and registered reviews.
- Reviews 5/21/42/63: US 5 October, 27 October, 25 November, 28 December.
- Model refits at anchor+21/+42: US 28 October and 27 November. The latter is an
  early-close session, due 28 November 02:30 HKT.
- Final decision: **29 December 2026, 05:30 HKT** (US 28 December).
- Final t+6 label and terminal-accounting receipt: **7 January 2027, 05:30 HKT**
  (US 6 January). Six tail sessions process pending fills/marks without new targets.

Current verified coverage is **0/63 decisions, 0 mature decision dates, 0/6 terminal
sessions**, with no missed session or halt. Commissioning/replays do not count.
The 28 September follow-up rechecked the sealed status, active launch agent and
fresh heartbeat, and saved a read-only operational baseline at
`data/research-v2/operations/observer-baseline-20260928T1005Z.json` with a
manifest-verified interim report under
`data/research-v2/prospective/20260928T1005Z-operational-baseline/`.
No prospective session was yet due; setup activity remains separately labeled.
Times vary with US daylight saving and early closes; inspect the calendar-derived
next event rather than using a fixed Hong Kong cron time. A missed expected
session is an outage and does not slide the fixed end date.

**The Mac must remain powered, awake, logged in and network-connected.** It has
not undergone a physical sleep/power-loss experiment, and sustained unattended
operation has not yet occurred. Simulated outages and actual process kills are
separate evidence. An hourly Codex observer (`trading-pilot-evidence-watch`) is
active for actionable failures and registered milestones; it does not trade or
change the pilot and stays quiet when nothing meaningful changes.

## Research and verification

The protocol, exposure log and registry were recorded before new evaluations.
All dates through 25 September are exploratory/consumed. R1–R10 are completed
within their registered scope; R11 is tested locally; R12 awaits real time.
The study completed **848 controlled trials/1,696 estimator fits**, plus four
runtime-pilot fits. There are 352,800 probability-view records but only 5,544
unique base instrument-date observations across 252 outer dates. Seventeen
predictive and eight economic contrasts have registered date-block/Holm controls.

A clean environment reran every trial with maximum probability difference 0.0
and reproduced 40 portfolios / 2,520 account sessions, including targets, events,
ledger states, economic metrics and inference. Dependencies, sources, snapshots,
failed attempts, split boundaries and checkpoint hashes are retained.

Calibration improved the raw mixture but did not establish superiority to the
trailing prior. In the consumed audit, hybrid accuracy is 37.59% versus 40.62%
for the prior, and log loss 1.0882 versus 1.0756. Fresh-capital audit returns are
-0.46% hybrid versus +2.55% costed equal weight. These differ legitimately from
the original continuous-account audit because starting holdings differ.
No tested feature/cadence/transfer change establishes adjusted superiority.

Verification includes 27 source-bound r2 release tests, eight actual SIGKILL
recoveries, five refused invalid-evidence cases, actual original-path backup
restoration, a separate 17-case public-provider integration job and an actual
Streamlit-to-CLI commissioning integration. The final full-repository job passed 370 tests with zero failures or skips and unchanged source. Results and exact
collected IDs are under `tests/final-stable/`; UI evidence is under
`tests/ui-r2-qualified/`. Earlier passing suites whose evidence qualification
failed are retained: one collector command collapsed test IDs, and one run
observed a manuscript-source edit. They are not used as final qualification.

## Four separate decisions

| Gate | Decision |
|---|---|
| Engineering | Qualified for tested local commissioning, replay, recovery and scheduled collection; sustained host availability remains unobserved |
| Historical edge | Not approved; consumed historical experiments do not establish deployable predictive/economic superiority |
| Prospective | Pending 63 actual decision receipts, mature outcomes and 6 terminal-accounting receipts; completion will not automatically approve economics |
| Manuscript | Substantial historical manuscript, editable source, PDF and claim map complete; prospective results explicitly interim |

Broker readiness is outside these approvals and remains disabled.
The full requirement-by-requirement disposition is in `ACCEPTANCE.md`.

## Evidence and continuation

- `MANUSCRIPT.md`, `manuscript.template.md`, `references.json/.bib` and
  `output/pdf/ML_TradeEngine_Research_v2.pdf`: manuscript and sources.
- `data/research-v2/statistics.json` and `claim-map.json`: common statistics,
  source hashes and empirical claim selectors.
- `DIAGNOSTICS.md`, `EXPERIMENTS.md`, `OPERATIONS.md`, `INDEPENDENT_REVIEW.md`:
  detailed methods, negative findings, review and recovery instructions.
- `data/research-v2/experiments/full_reproduction.json`: complete clean replay.
- `data/research-v2/operations/*-r2.json`: real schedule, restore, process recovery
  and timing evidence. `data/research-v2/pdf-final-qa/`: rendered-page QA.

Create a new dated output directory each time; the report is read-only, takes a
shared lock and refuses paths inside the pilot workspace or sealed release:

```bash
data/research-v2/clean-env/bin/python -B -m research_v2.prospective_report --output data/research-v2/prospective/NEW-DATED-RUN
```

Only after all registered decisions, mature outcomes and terminal sessions exist,
add `--freeze`. Premature finalization is refused. The final report reconstructs
the costed equal-weight account from identical observed snapshots, attaches
recorded-model priors, computes the registered dependent-sample comparisons and
archives evidence. Interim views are descriptive; inferential tests are withheld
until the final freeze. No automatic strategy promotion occurs.

```bash
/Users/a/miniconda3/bin/python -B -m research_v2.paper compile
/Users/a/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -m research_v2.paper pdf
/Users/a/miniconda3/bin/python -B -m research_v2.paper check
```

The compiler consumes the latest manifest-validated prospective report. Review
and render every updated PDF page before any later publication. Historical
point-in-time constituent/action tapes and real broker microstructure are absent;
time passing in this pilot will not repair those missing historical archives.

The separate November 2026 [CECI2 experiment](../../ceci2_demo/README.md) trains on the proposal's next-open labels. It has its own data, out-of-time forecasts and exposure-matched benchmarks. The archived release remains the source for claims about the earlier pilot.

The November 27 [traditional mixture and offline allocation study](../../ceci2_demo/moe_v2/README.md) adds a jointly trained mixture benchmark and a qualitative-data ablation. All three evaluation years were previously examined. The simulated economic improvement of the mixture remains exploratory. The learned allocation policy underperformed its exposure-matched SPY control. The study documents the replay command, fitted checkpoints, source papers and observations by date. The next independent comparison requires a registered future period with timestamped qualitative events.

The [2022 extension](../../data/ceci2-demo/periods-v1/README.md) reuses the original classic mixture and provides a fourth matched year. The [model-routing study](../../data/ceci2-demo/rl-router-v1/README.md) evaluates the identity of the expert chosen for each stock. The period comparison also includes new jointly trained networks and fixed versus contextual gates across three expert pools. The published Compare view reports annual, quarterly and matched multi-year portfolio results. Run `python -m ceci2_demo.periods_v1.export` and `python -m ceci2_demo.periods_v1.model_trace` to regenerate its records.

The [illustrated comparison](../../site/assets/ensemble-study.pdf) explains the archived experiments and their limitations. Rebuild its figures and report with `python output/pdf/ensemble-study-2026-11-27/build_report.py` after exporting the period records.

The [data-to-results guide](../../site/assets/pipeline-guide.json) connects the original proposal to the historical implementation and powers the website guide and the opening pages of the illustrated report. It distinguishes the forecast horizon from trading frequency and describes the zero-reference Sharpe calculation used in the website tables.

The guide now plots the annual text records and compares matched returns, drawdowns and Sharpe ratios. Its model combination charts read the archived Colab portfolio results. The Model votes view derives its next-open target from the fixed replay simulator.

The [data provenance guide](../../site/assets/forecast-sources.json) explains the data sources, the two hosted forecast inputs and the separate research-agent experiment. The report builder derives the company chart and the Apple example from the archived source receipts and the recorded decision.

The [model input and output map](../../site/assets/model-map.json) connects the forecasters, controllers and trading components to their observed inputs. The website and report use this same map and label the matched evaluation windows separately from training histories.

The current illustrated report is six pages. Its pipeline, data charts, model map, worked forecast and matched results appear in the main text. The site retains the full interactive breakdowns and links directly to the matching report pages.

The website groups its pages under Study, Models, Results and Data. The opening comparison now holds the 13 forecasts and trading rules fixed. The Model votes picker includes fixed and adaptive mixtures, jointly trained forecasts and single-model routing. Regenerate the saved examples with `python -m ceci2_demo.periods_v1.export_adapter_examples`.

The pipeline, model map and data pages now lead with diagrams and charts. The six-page PDF uses the same method names and shows the Sharpe comparison and the recorded Apple decision visually.

The report and data page separate quantitative and qualitative inputs. The Yahoo Finance figure uses weekly samples of cached closing prices and the archived adjustment ratios. The matched pipeline comparison uses the 719 shared sessions in 2023–2025.

The annual research-agent view reads recorded decisions and returns for both horizons. No OpenAI-proposed setting was adopted. The five-session random-review setting changed for 2025. The study refits the gate and does not retrain the hosted language forecasters.

The training page plots the recorded Colab objective for the two jointly trained fits and the held-out log loss for the fixed, contextual and single-model voting methods. The report labels these as distinct measurements.
