# Controller architecture and evidence audit — 30 October 2026

This audit read the repository instructions, execution brief and evidence/test plan, inspected current source, and verified saved manifests. It did not modify the frozen collector, account databases, schedules, releases or prior research manifests.

## What “controller” means in this repository

There are distinct control layers. A forecasting combiner weights model probabilities; a strategy allocator converts forecasts and trend rules into desired holdings; a deterministic risk/execution layer constrains or rejects those holdings. The first new experiment in this directory tests only the forecasting layer. Forecast improvement cannot substitute for a portfolio experiment.

| Component | Current implementation and evidence | Implication |
|---|---|---|
| Rolling forecast experts | `src/models/adaptive.py:103` fits standardized logistic regression and histogram gradient boosting; `:126` averages probabilities and selects temperature; `:138` opens a binary gate against the fitted prior. | The qualified baseline has two modest tabular experts. Its calibration and original gate reuse the same trailing block. |
| Controlled gate experiments | `research_v2/experiments.py:96` separates fit/calibration/gate by actual label endpoints; `:149` fixes logistic/HGB settings; `:157` calibrates and compares reused/separate gates; `:170` emits seven probability views. | Independent gate validation already exists in historical research; it is different from a context-dependent mixture-of-experts router. |
| Checkpoint loading | `research_v2/models.py:24` checks the saved model hash and maps the original CLI class names. | This file is a loader, not the controller model implementation. |
| Legacy context gate | `src/models/gating.py:109` delays rolling expert correctness by the label horizon plus one; `:213` fits one regularized logistic correctness model per expert; `:257` normalizes nonnegative weights. | A context gate for XGB/LGBM/LSTM exists, but its old results are not admitted as comparable v2 evidence. |
| Legacy strategy allocator | `src/research/meta_strategy.py:191` selects past in-regime Sharpe rankings with hold/confirmation rules. | It ranks strategy returns; it is not equivalent to fitting a predictive probability combiner or executing matched capital accounts. |
| Operational policy | `runtime_v2/policy.py:86` makes trend-only target weights; `:95` constrains them; `:118` gives deployment cash and the shadow trend targets. | Model forecasts and the gate are recorded diagnostics. The operational shadow does not currently allocate by the learned gate. |
| Risk contract | `runtime_v2/risk.py:15` sets 80% gross, 10% asset, 40% target turnover, 35% growth-group exposure and 15 basis-point one-way costs; `:27` applies a 12% estimated annual-volatility ceiling using 63 observations and covariance shrinkage. | Risk controls belong in design, implementation and evaluation. The ceiling is an estimate-based sizing control, not a guaranteed realized-risk bound. |
| Persistent execution | `runtime_v2/account.py:20` defines limits including a 20% drawdown trigger and 3% cash-reservation buffer; `:288` applies the drawdown response; `:394` checks fills against remaining reservations; `:470` processes close events transactionally. | Controls can be demonstrated through intended/rejected weights, reservations and restart-safe receipts. A drawdown trigger cannot guarantee a maximum realized loss under gaps or execution delay. |

The legacy meta-strategy docstring (`src/research/meta_strategy.py:24`) still describes a full-series HMM. Current `src/features/regime.py:140` instead fits the first 252 eligible observations and emits only later forward-filtered states, with warm-up unknown. Do not treat that stale comment as a current implementation finding. Saved old features still require their own provenance.

## Evidence verified during this audit

- All **42** files in `data/research-v2/experiments/artifact_manifest_v2.json` matched their recorded hashes.
- All **five** source files pinned by the controlled experiment design matched current source.
- All **848** checkpoint pairs matched **1,696** saved model/prediction SHA-256 values. All **1,328** instrument-level partitions satisfied strict fit endpoint → calibration start → gate start → decision ordering.
- The original `artifact_manifest.json` has one expected mismatch: the later expanded `docs/research-v2/EXPERIMENTS.md`. The v2 manifest explicitly records the final document and preserves its earlier snapshot. Neither original manifest was rewritten.
- The saved full-reproduction receipt records 848 reproduced trials, maximum probability difference 0.0 and 40 reproduced account histories/2,520 account sessions. The audit verified this receipt's hash; it did not rerun those fits.

The controlled 2024–2025 base panel has **5,544 symbol-date pairs over 252 dates**, not 352,800 independent samples. The latter counts multiple forecast views, variants and seeds. Its fitted prior achieved log loss **1.063799** and accuracy **42.190%**, versus **1.077345 / 40.133%** for the calibrated mixture and **1.067900 / 41.937%** for the separate gate. Gate separation's interval includes zero; all eight registered economic contrasts have intervals including zero. These statements come from the manifest-verified `data/research-v2/experiments/summary.json` and `docs/research-v2/EXPERIMENTS.md:87`.

Legacy XGB/LGBM/LSTM/MoE results were excluded from that controlled experiment because they did not bind transforms, label endpoints, calibration roles and execution labels to the same partitions within its budget (`research_v2/experiments.py:253`; `docs/research-v2/EXPERIMENTS.md:44`). They were not shown inferior. Newly admitting them requires fresh versioned, comparable evaluation. Macro/news inputs also need historical availability evidence.

## Current prospective collection: receipt-derived snapshot

The read-only sealed command was captured in `data/research-controller-v1/audit/r7-status-20261030.json` with its exact check timestamp. It verified **2/6** outcome-read receipts (US 27 and 28 October), **4/5** mature original decision dates, and a missing US **29 October** read. The service was not integrity-halted. External orders remained disabled, and archived account ledgers remained consistent at their last actual marks.

The next registered collection was **31 October 2026, 04:30 HKT**, for US 30 October. The missed receipt is independently recorded in `data/paper/research-outcomes-r7/events/2026-10-29-watchdog.json:2`. Future observations cannot recreate that on-time receipt or the earlier unfinished account tails. The six-read finalization condition therefore remains unmet. No collection or schedule change was made during this work.

## Integration boundaries for continued research

Use a new versioned forecasting module and an explicit probability/schema/availability contract. Preserve an always-available prior and equal mixtures as controls; evaluate routing benefit against the same expert set. Use execution-aligned `t+1` entry / `t+6` endpoint labels, release each outcome only after its endpoint, and aggregate feedback by date before updating shared controller parameters. Fit scaling, stacking, regimes and calibration only within their eligible partitions.

The new `research_controller_v1/screen.py` follows that boundary and reuses sealed rolling forecasts. It deliberately has no runtime import or account integration. Any portfolio continuation must convert candidate outputs into identical intent rules and pass the same risk and execution contract with matched dates, costs and starting capital. No automatic deployment or live policy promotion follows this screen.
