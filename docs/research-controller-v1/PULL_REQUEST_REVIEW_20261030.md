# GitHub prototype and pull-request review — 30 October 2026

**There are no pull requests to accept in the supplied repository at this review snapshot.** GitHub's open-PR endpoint, all-state PR endpoint and independent PR search each returned an empty collection. The repository has one branch, `main`, at `f194283bb6104e3feb980a67bc05a06459677909`. The local checkout has the same committed HEAD but substantial newer uncommitted work.

Reviewed repository: [Yash789k/ML-TradeEngine](https://github.com/Yash789k/ML-TradeEngine). Snapshot: approximately **30 October 2026, 17:38 HKT**. Exact response timestamps, URLs and source hashes are saved in `data/research-controller-v1/github-review-v1/`.

| Requested decision | Verified result | Recommendation |
|---|---|---|
| Open PRs to merge | 0 returned by REST and independent search | Nothing to accept or merge now |
| Closed/merged PR history | All-state PR list returned 0 | Do not infer missing PRs from local changes or commit messages |
| PR head SHAs, reviews, dependencies, conflicts | No PR objects exist in the supplied repository | Not applicable; no per-PR approval or merge order is claimed |
| Published prototype readiness | Concrete causal, accounting and operational defects remain in `main` | Hold production/research-validity claims; port and qualify repairs in small future PRs |
| Current local research/runtime | Newer modifications and versioned evidence are outside the published commit | Preserve them; review and package deliberately rather than replacing them with `main` |

The [open-PR listing](https://github.com/Yash789k/ML-TradeEngine/pulls) and [reviewed commit](https://github.com/Yash789k/ML-TradeEngine/commit/f194283bb6104e3feb980a67bc05a06459677909) are the relevant GitHub destinations. This review did not create PRs, submit comments, approve, merge, push, rerun workflows or change schedules.

## Published prototype findings

These findings apply to the exact published commit above. They must not be read as findings against the newer current working tree or its sealed local pilot.

### P1 — Exit accounting drops the price movement on the fill date

In [simulator.py line 121](https://github.com/Yash789k/ML-TradeEngine/blob/f194283bb6104e3feb980a67bc05a06459677909/src/backtest/simulator.py#L121), closing a position sets cash from the previous day's equity rather than actual shares and the exit price. Although `proceeds` and trade P&L use the current price, account equity discards the intervening return.

An isolated deterministic oracle on the exact published function used prices `[100, 100, 110]`, prior-close signals `[UP, FLAT, FLAT]`, $10,000 and zero costs. Buying at 100 then selling at 110 should finish at **$11,000**. The function finishes at **$10,000** while its exit trade reports **+$1,000**. This directly breaks trade/account reconciliation and invalidates derived performance metrics. Evidence: `prototype-oracles.json`.

**Needed before accepting a repair:** hand-calculated entry/exit and gap cases, entry-plus-exit costs, terminal holdings, cash/position reconciliation and consistency between trade P&L and equity. The local `src/backtest/simulator.py` is modified; preserve that repair rather than copying the published version over it.

### P1 — Regime features incorporate future data

[regime.py lines 155–157](https://github.com/Yash789k/ML-TradeEngine/blob/f194283bb6104e3feb980a67bc05a06459677909/src/features/regime.py#L155) fit the HMM to the whole supplied frame and decode that whole frame. [Line 98](https://github.com/Yash789k/ML-TradeEngine/blob/f194283bb6104e3feb980a67bc05a06459677909/src/features/regime.py#L98) uses sequence-wide Viterbi prediction. The feature pipeline calls this before model folds in [pipeline.py line 185](https://github.com/Yash789k/ML-TradeEngine/blob/f194283bb6104e3feb980a67bc05a06459677909/src/features/pipeline.py#L185). Thus a past feature can depend on observations that had not happened at its decision time.

**Needed before accepting a repair:** fit only an eligible past calibration prefix, use forward filtering, keep fitting warm-up rows explicitly unavailable, and demonstrate invariance to appended/changed future observations. Current local `src/features/regime.py` already takes that different approach; it is not the code on GitHub.

### P1 — Feature selection and weighting consume evaluation labels

[trainer.py lines 172–184](https://github.com/Yash789k/ML-TradeEngine/blob/f194283bb6104e3feb980a67bc05a06459677909/src/models/trainer.py#L172) run feature selection over the complete cleaned dataset before the walk-forward loop. The invoked selector fits on all passed labels in [pipeline.py line 269](https://github.com/Yash789k/ML-TradeEngine/blob/f194283bb6104e3feb980a67bc05a06459677909/src/features/pipeline.py#L269). Separately, [trainer.py line 246](https://github.com/Yash789k/ML-TradeEngine/blob/f194283bb6104e3feb980a67bc05a06459677909/src/models/trainer.py#L246) computes class weights from all labels, then supplies them to earlier folds.

For fold-level ensemble metrics, [lines 337–345](https://github.com/Yash789k/ML-TradeEngine/blob/f194283bb6104e3feb980a67bc05a06459677909/src/models/trainer.py#L337) derive weights from that fold's test F1 and score the weighted mixture on the same test fold. The later persisted prediction path uses a separate fixed-weight expression, so fold ensemble metrics and saved forecasts also have different semantics. Do not attribute the same-fold weighting defect specifically to those saved fixed-weight forecasts; they remain affected by the upstream feature-selection/class-weight issues.

**Needed before accepting a repair:** every learned transformation, class weight, ensemble weight and calibration choice must use eligible training/inner-validation data; exact prediction semantics must match the evaluated artifact. Add future-label mutation and boundary tests. These source files are modified locally and require a deliberate repair PR, not a blanket overwrite.

### P1 — Broker risk and duplicate-order checks fail open

The published live engine catches circuit-breaker failures at [line 212](https://github.com/Yash789k/ML-TradeEngine/blob/f194283bb6104e3feb980a67bc05a06459677909/src/live/engine.py#L212), duplicate-order query failures at [line 222](https://github.com/Yash789k/ML-TradeEngine/blob/f194283bb6104e3feb980a67bc05a06459677909/src/live/engine.py#L222), and portfolio-heat failures at [line 238](https://github.com/Yash789k/ML-TradeEngine/blob/f194283bb6104e3feb980a67bc05a06459677909/src/live/engine.py#L238), logs warnings, and continues to execution at line 242. Buying-power lookup failure also falls through in [line 326](https://github.com/Yash789k/ML-TradeEngine/blob/f194283bb6104e3feb980a67bc05a06459677909/src/live/engine.py#L326).

When broker calls fail, an increase can therefore proceed without verified risk state or idempotency. The heat calculation also measures filled positions rather than a persistent reservation of pending orders.

**Needed before accepting a repair:** fail visibly before any increase when required account/order/risk information is unavailable; retain deterministic client order IDs, pending cash/exposure reservations and restart-safe reconciliation. Keep external orders disabled while that contract is qualified. This inspection did not call a broker or submit orders.

### P2 — Missing HMM rows shift the decoded labels to the wrong dates

[regime.py line 99](https://github.com/Yash789k/ML-TradeEngine/blob/f194283bb6104e3feb980a67bc05a06459677909/src/features/regime.py#L99) maps the valid index against itself, producing positions `0..n_valid-1`, rather than mapping valid rows back to the original frame. With the first row missing and two valid rows, a deterministic decoder returning `[2, 1]` produces `[2, 1, -1]`; correct alignment is `[-1, 2, 1]`.

This was reproduced with the exact published function and a decoder stub, without model fitting. Fixing causal HMM estimation alone is insufficient unless index alignment is also tested. The newer local version maps actual finite-row positions.

## Remote checks and operational state

All **43** returned check runs attached to the current main SHA have conclusion `failure`. They are repeated **Generate Signals & Submit Orders** scheduled jobs, not 43 distinct code-test cases and not PR qualification checks. The combined commit-status endpoint returned no statuses. No successful remote test suite for this head was established by this review.

The [latest inspected job](https://github.com/Yash789k/ML-TradeEngine/actions/runs/33571092654/job/100064990569), from 1 September, passed dependency installation and failed during live-engine initialization. The log identifies [alerts.py line 61](https://github.com/Yash789k/ML-TradeEngine/blob/f194283bb6104e3feb980a67bc05a06459677909/src/live/alerts.py#L61): `int(os.environ.get("SMTP_PORT", "587"))` receives the empty string supplied by an unset workflow secret. This is an exact operational failure, independent of the scientific defects above. The current local file handles the blank-string case differently.

Workflow metadata reports `disabled_inactivity`. Its published [schedule](https://github.com/Yash789k/ML-TradeEngine/blob/f194283bb6104e3feb980a67bc05a06459677909/.github/workflows/daily_signals.yml#L4) uses one UTC weekday time instead of exchange sessions/DST/early-close events. The [dry-run setting](https://github.com/Yash789k/ML-TradeEngine/blob/f194283bb6104e3feb980a67bc05a06459677909/.github/workflows/daily_signals.yml#L15) defaults false and scheduled invocation lacks an explicit external-order authorization gate. These are reasons to keep this old workflow inactive while the separate local collection continues, rather than enabling it merely to obtain a green job.

The GitHub plugin supplied PR, branch, source, check and job-log evidence. Its generic fetch rejected workflow metadata and check-annotation endpoints; the workflow state was therefore read using `gh api` as a fallback. The annotation rejection was not interpreted as evidence of no annotations.

## Local overlap and sensible future review order

The inspected working tree has **44 tracked modified files**, plus untracked versioned research/runtime/test/evidence directories. GitHub main and local HEAD matching does **not** mean their current source trees match. In particular, `runtime_v2/`, `research_v2/`, the r2–r7 collection work and `research_controller_v1/` are not in the published commit. Their local evidence does not qualify the older GitHub source.

There is no actual PR merge order to approve today. If the local work is prepared for GitHub later, the following separation would make review practical:

1. **Safety and numerical repairs:** causal data/labels, account mathematics, model persistence and fail-closed execution behavior, with focused regression tests. Keep expensive generated artifacts and performance claims out of this review unit.
2. **Versioned runtime and collection evidence:** shared policy/ledger contract, integrity checks, durable receipts, recovery and scheduler configuration. Review against the repaired foundation; keep external orders disabled and avoid duplicate active schedules.
3. **Research and reporting:** explicit protocols, exposure logs, evidence-linked statistics/manuscript and honest readiness distinctions. Preserve prior negative/failed results.
4. **Controller experiment and interactive demonstration:** isolated controller screen, baselines, tests and display artifacts. This depends on reliable data/provenance; it must not silently replace the frozen pilot.

This is packaging advice, not approval of all local changes. Source overlap is substantial, and each proposed PR will need its own actual head SHA, complete diff, tests and conflict/dependency review. Do not merge an unseen PR simply because it has a plausible title or a passing workflow.

## Verification limits and preserved evidence

The review used exact Git objects at the remotely verified head for source inspection and executed two targeted deterministic numerical oracles. It did not run the entire old dependency-heavy test suite, reproduce the prototype's historical model results, inspect unpublished branches in other repositories, or validate any PR that does not exist in this repository snapshot.

Saved evidence includes `github-evidence.json`, `workflows.json`, `prototype-oracles.json` and eight exact published source files under `published-source/`. No existing working-tree source was edited during the review. No model/release manifest, live account, collector or scheduler was changed.
