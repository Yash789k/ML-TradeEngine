# Repair record — 28 September 2026

Scope: the seven tasks in the attached engineering brief. Existing local edits
and historical data are preserved. A pre-repair diff is saved under
`docs/repair-evidence/`. No broker orders or external alerts are authorized.

## Conventions

Daily decisions use information through the completed close; fills occur at the
next observed close. Commission and slippage are separate configured rates,
charged as cash costs on absolute reference-price notional on both sides.
Cash earns zero; reported excess-return metrics use a documented fixed 5%
annual hurdle, compounded to 252 daily periods. Drawdown includes starting NAV.
Open positions remain marked at the terminal close and are excluded from closed
trade win rates. Short proceeds are cash with an offsetting signed liability;
borrow costs and broker margin are not modeled in the legacy single-asset engine.

## Task evidence and gates

1. Accounting: implemented and verified; exact ledger and fee examples pass.
2. Causality: implemented and regression-tested; legacy artifacts remain ineligible.
3. Operations: local bootstrap verified; provider/remote CI readiness is explicitly limited.
4. Research: 22-ticker corrected reruns completed; no candidate qualified.
5. Shared portfolio: implemented; five reconciled historical account replays completed.
6. Execution parity: historical and local-paper handlers agree, including restart;
   external broker adapter parity remains unverified and submission isolated.
7. Replay/research status: evidence and revised documentation completed;
   prospective evidence requires future observations and separate authorization.

## Historical outputs affected

All pre-repair backtest, zoo, meta, risk, MoE, policy, sentiment, paper, README,
and dashboard performance claims require new manifests and corrected evaluation.
They are retained as historical exploratory artifacts, not overwritten or promoted.
Mean per-ticker metrics and blends of separate return streams are not account results.

## Task 1 — verified accounting repair

**Confirmed:** actual exit marks were omitted from cash; shorts compounded the
previous NAV; entry costs were missing from closed P&L; initial losses could be
missed by drawdown; Sortino used negative-return standard deviation; benchmark
holes became zero returns; zero-trade reruns retained old logs.

**Changed:** signed cash/share accounting with a daily reconciliation ledger;
net round-trip P&L; fixed schema for empty logs; initial-NAV drawdowns;
target-relative downside deviation; excess-return alpha; strict benchmark
coverage; signed HAC statistics. The risk simulator uses exact notional costs.

**Verification:** `python -m pytest tests/test_accounting_regressions.py
 tests/test_backtest.py tests/test_risk.py -q` — 66 passed; exact long/short,
flat-price, fees, reversals, open P&L and zero-trade overwrite examples.
Evidence: `repair-evidence/task1-tests.txt`.

**Affected history:** all standard/zoo/meta outputs and downstream risk
calibration. No historical result files were replaced.

**Limits:** reference-close notional friction is a model, not an observed fill;
legacy short tests exclude borrow costs and broker margin. Shared execution
explicitly disallows shorts and leverage. Terminal holdings remain marked.

## Task 2 — verified causal transformations

**Confirmed:** full-history regime estimation/decoding and indexing error;
full-dataset feature selection/class weights; fold-test ensemble weighting;
unmatured gate labels; future medians/backfills; full-history risk payoff ratios.

**Changed:** fixed initial 252-observation HMM fit, forward-only filtering and
unknown warm-up; positive transition priors and training-only scale fitting;
fold-local supervised selection/weights; purged inner/outer splits; fixed
ensemble weights; full LSTM sequence context; horizon+one-bar gate delay;
fixed ex-ante risk payoff assumption; macro admission requires availability
and vintage evidence; session-aware news timestamps. Final-model historical
backfill is forbidden. New model bundles include cutoffs and artifact hashes.

**Verification:** six new mutation/prefix/maturity/macro/purge regressions;
105 tests passed across causal features, models, gates, policies and research.
Evidence: `repair-evidence/task2-tests.txt`. The later full offline suite passed
290 tests (before the final additional execution/operational cases).

**Affected history:** every legacy model bundle and HMM-dependent strategy;
old MoE, policy and sentiment reports remain exploratory.

**Limits:** adjusted-price snapshots can reflect vendor revisions; macro and
sentiment cannot qualify without historical availability evidence. HMM
parameters are frozen after the initial calibration; periodic adaptive HMM
refits are not implemented. Risk classifier probabilities are not calibrated
probabilities of stop-managed trade profit.

## Task 3 — operational repairs and explicit admission checks

**Confirmed:** blank SMTP port exception, absent inference features, fabricated
short LSTM sequences, stale dependent features, synthetic weekday bars,
unfinished candles, Kraken history over-request, silent missing-ticker failures.

**Changed:** calendar checks and completed-bar admission; cache fingerprints and
forced-refresh propagation; required feature/sequence/probability checks;
model manifests and bootstrap command; fixed SMTP default; batch errors fail
closed and CLI exits nonzero; local transactional state replaces implicit
broker execution. Remote alerts and persistence are intentionally disabled.
The workflow now installs dependencies, tests, bootstraps artifacts, and performs
signal dry runs. It has not been dispatched from this session.

**Verification:** 15 operational/account regressions initially passed; a clean
artifact sandbox trained all three classifiers, loaded the sealed bundle,
inferred a real sequence and generated 40 local fills across 60 events. Restart
state matched and maximum ledger residual was 3.64e-12 dollars. This synthetic
run establishes functionality, not predictive or trading performance.

**Affected history:** legacy model bundles are rejected rather than silently
accepted; cached production data is stale. Original production caches remain
untouched by repair experiments.

**Limits:** clean *artifact* bootstrap was tested in the installed Python 3.13
environment; a fresh GitHub runner still needs a real CI run. Public provider
integration tests are explicitly marked `integration` and not part of offline
verification. Read-only operational inventory is saved in
`repair-evidence/operational-status.json`.

## Task 4 — corrected comparisons completed; no qualifying strategy

**Confirmed:** saved results could not identify a defensible winner because
accounting, regimes, model selection and benchmark coverage were inconsistent.
The previously viewed historical periods cannot become an untouched holdout.

**Changed:** frozen common-session snapshots and archived source/dependencies;
training through 2023, selection in 2024, exploratory evaluation from 2025;
naive classification and trading baselines; signed HAC and block-bootstrap
active-return uncertainty; Bonferroni control over 330 strategy/ticker hypotheses;
fixed cost stress; explicit supplementary-data exclusions; model/output hashes.

**Verification:** `scripts/validated_research.py --output
 data/repairs/2026-09-28-causal-v2 --ml --trials 3 --epochs 10` completed 22
symbols and 748 metric rows, without experiment failures. Ensemble accuracy
averaged 34.31%, versus always-UP 41.94% and training-majority 40.04%.
Zero candidates met all declared eligibility gates. Full source/input/output
hash verification passed. See the generated repair report for evidence links.

**Affected history:** the original favorable meta and win-rate claims are
withdrawn as validation evidence. This is not a same-period accounting-only
sensitivity: data admission, model training and evaluation periods also changed.

**Limits:** bounded model budget; revised vendor history; retrospectively chosen
universe; previously examined evaluation dates. Sentiment and vintage-dependent
macro cannot be validated with current evidence. The MoE validation training
slice is ineligible for independent ranking. Future holdout remains unobserved.

## Task 5 — common target-position account

**Confirmed:** old live FLAT skipped exits; UP added an allocation; risk ignored
pending orders; risk/idempotency exceptions permitted continuation; brackets
and fill timing differed from research.

**Changed:** both `src/backtest/portfolio.py` and `src/live/engine.py` use
`ExecutionSession.on_close` and the same `Account`. Complete target weights
produce holding deltas, with cash reservations, entry/exit fees, position,
gross, turnover and group limits. Repeated IDs verify inputs. Partial fills,
cancellations, rejected gaps and restart are persisted. SQLite commits each
close event atomically and rejects competing stale writers. FLAT targets zero;
policy caps flow to weights. The contract is next observed close, no bracket
stops/profit targets, fractional long-only US equities/ETFs, no leverage.
Conflicting strategy targets net according to their assigned capital budgets.
Market-drift breaches block increases and are recorded for reduction.

**Verification:** account tests cover shared capital, entry, hold, reduction,
exit, same-event retry, gap rejection, partial fill, restart, invalid targets,
correlated group caps and concurrent writers. Sixty randomized events produced
identical historical and restarted local-paper states and event logs.

**Affected history:** old bracket execution and return-stream meta blends do
not represent this account. Those experimental APIs remain separately labeled.

**Limits:** no claim of Alpaca fill parity; external submission is isolated.
Pending sells are not used as spendable cash until filled. Market price changes
can cause temporary exposure drift; orders cannot add to a projected breach.
Legacy stop/Kelly experiments remain a separate exploratory risk simulator.

## Task 6 — local execution parity and recovery

**Confirmed:** FLAT/UP semantics, fill timing, brackets, pending-order checks and
exception handling differed across old backtest and broker paths.

**Changed:** the daily engine and portfolio replayer share one event handler;
FLAT targets zero, UP uses target deltas, policy caps reach the account, and
next-close fills use the same costs and constraints. No brackets or external
order transport are invoked. Event commits are atomic and stale writers fail.

**Verification:** direct LiveEngine-versus-replay integration, randomized restart
comparison, duplicate decisions/fills, partial fills, reductions, exits, gap
rejection and failed-event rollback. Seventeen operational/execution tests passed
in the focused final group; all are included in the 301-test offline pass.

**Affected history:** previous broker-paper behavior cannot validate a backtest;
legacy bracket order assumptions and return-stream blends are not reused as
shared-account fill evidence.

**Limits:** no Alpaca integration or future fill evidence. The local adapter is
ready for simulation; broker timing and account reconciliation need separate
verification before an authorized trial.

## Task 7 — local evidence and honest research status completed

**Confirmed:** prior broker-paper activity and publication claims lacked a
reconciled shared-account evidence trail.

**Changed:** sequential local replay through the actual daily event handler;
transactional decision/fill logs; immutable manifests and snapshots; exposure,
cash, drawdown and reconciliation checks; figure/table generation from saved
outputs; rewritten README/paper; historical-output warnings; prospective trial
and model-admission protocols. External submission remains isolated.

**Verification:** five $10,000 accounts over 2 January 2025–18 May 2026 generated
18,998 simulated fills. All reconciled within 3.10e-11 dollars. Highest ending
NAV among these comparisons was the fixed momentum scenario, $11,975.01;
none had positive adjusted excess-return evidence against SPY. The clean-artifact
synthetic bootstrap separately completed 40 fills and real three-model inference.
Final full offline suite: **301 passed, 17 network-provider tests deselected**.
`git diff --check`, Python compilation, and frozen-file hash validation passed.

**Affected history:** README/paper copies retained in `docs/archive/`; prior
build logs, original audit and original result directories remain available.
Only the final `causal-v2` and `shared-v2` directories support the new report.

**Limits:** no fresh-runner CI dispatch, live provider certification, broker
adapter reconciliation, full second training reproduction, prospective trading
record or durable-alpha claim. See `PROSPECTIVE_PAPER_PROTOCOL.md` for the next
release gates. No new scheduled automation was created.
