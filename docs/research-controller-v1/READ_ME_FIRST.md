# October research preparation: Adaptive Ensemble Trading System

Prepared 30 October 2026, Asia/Hong_Kong. This package supports the forthcoming October progress report. The rubric and additional tutor feedback have not yet been supplied. It is not a final monthly report or a claim that all proposed models have been implemented.

## The project, after reading the proposal

The contribution is **a learned contextual gate over a fixed set of UP/FLAT/DOWN classifiers**, evaluated against individual experts and a static ensemble under the same risk policy and execution. HMM probabilities and mature expert-performance history are potential gate inputs. New text or foundation-model experts enlarge the pool only in separate, controlled comparisons. The slower LLM research agent is secondary; direct portfolio reinforcement learning is outside the committed core study.

This reading comes from the supplied 29-page proposal, particularly pages 3–5, 9–10, 16–17 and 24–26. The proposal's descriptions of the summer prototype are claims to check against implementation and receipts, not automatic proof that those components work as described.

## Recommended research order

1. **Finalize the actual gate.** Use a small regularized softmax gate that emits a convex weight vector over fixed expert probability vectors. Train it on genuinely out-of-time forecasts. A small nonlinear gate is a bounded challenger. Compare with fixed weights, a fitted static mixture, the prior and delayed Hedge. The present logistic forecasting stack is a useful control but is not this gate.
2. **Resolve labels and execution.** The proposal specifies next-open-to-open five-session returns and a threshold based on cost and past volatility. Existing verified research uses next-close-to-close returns with fixed ±1% thresholds. Either implement the proposal contract in a new version or explicitly document an amendment; never present these results as the same experiment.
3. **Isolate regime effects.** Remove HMM inputs from the gate while keeping the expert forecasts and policy identical. Removing HMM information from the whole pipeline is a different experiment. Use forward-filtered state probabilities, past-only fitting and support-aware regime summaries.
4. **Evaluate one newer expert.** Start with TabICLv2 or the pinned TabPFN-3.5 base classifier. Both map directly to the classification interface. A later frozen sequence-feature experiment can use TiRex plus a small classifier. Chronos-2, TimesFM-3 and Moirai 2.0 need a qualified class-probability adapter or classifier head. These priorities are judgements about project fit, not a claim that these models will improve trading.
5. **Test risk improvements separately.** Preserve hard cash, exposure, concentration and execution limits outside the gate. Study weight stability, uncertainty-based abstention and dynamic versus static risk rules one at a time. A constrained optimizer/CVaR allocator belongs in a later allocation comparison with forecasts held fixed.
6. **Explain the mechanism interactively.** Build toward a date-by-date replay showing available inputs, expert probabilities, gate weights, requested targets, binding risk limits, fills, cash and eventual mature outcomes. The delivered explorer covers the research comparison and preliminary results; complete trade replay remains later work.

The detailed gate formulation, ablation design and proposed resource limits are in [PROPOSAL_ALIGNMENT_AND_GAPS.md](PROPOSAL_ALIGNMENT_AND_GAPS.md). The newer models, exact release dates, source links, adaptation requirements and availability caveats are in [RECENT_MODELS_20261030.md](RECENT_MODELS_20261030.md).

## What is new and evidenced in this continuation

**Completed on 30 October:** read the proposal and supervisor feedback; checked the repository, sealed experiments and current receipts; reviewed 25 controller/control options and 12 recent-model/release entries; developed a proposal-aligned research design; built an interactive model/results explorer; implemented and ran one isolated preliminary probability-combination screen. The catalogues overlap in subject and are not 37 independently benchmarked models.

**Measured screen:** 250 previously inspected 2025 dates, 22 assets and 5,500 dependent symbol-date observations. Delayed Hedge and a two-fit logistic stack improved log loss over a fixed linear/tree mixture. Neither established an improvement over the trailing class-frequency prior. The primary endpoint and six comparisons were registered before this run; uncertainty used date blocks and Holm-adjusted p-values. These are preliminary prediction results, not net trading results. See [SCREEN_RESULTS.md](SCREEN_RESULTS.md) and its immutable source package.

**Engineering evidence for that screen:** initial checks and independent review covered delayed label eligibility, train-only scaling, complete-date updates, duplicate-feedback prevention, invalid probabilities, baseline arithmetic, checksum failures and future-data invariance through the July refit. The separate reconstruction supplement reproduced saved forecasts from frozen coefficients without additional real-data fitting. Passing these behaviors does not demonstrate a trading edge.

**Pre-existing foundation:** the broader repaired accounting, frozen collector, September research study and earlier models predate this continuation. They can be described as the base on which October work builds. They should not all be represented as features newly implemented in October without dated evidence. The local checkout is substantially ahead of its older published GitHub prototype; publication status and local implementation status are separate.

**Prospective limitation:** the saved read-only 30 October snapshot records five original interrupted decision dates, four mature decision dates, two of six outcome reads and a missing US 29 October read. The account tail is incomplete. This does not establish the uninterrupted October–April, static-versus-gated shadow programme described in the proposal. Later prices cannot recreate missed trades.

## What the October report can credibly say

The clearest narrative is that the team has strengthened the research design and tested early controller ideas, while discovering which assumptions still need alignment. Explain the lessons: model combination can beat a weak static mixture without beating a strong simple prior; operating receipts are different from backtest results; and a fair regime ablation must preserve other information paths.

Use the following status labels consistently:

- **Implemented and evaluated:** the specifically identified local modules and source-bound experiments, with dates and limitations.
- **Implemented but not qualified for this comparison:** legacy experts or prototype features that lack matching data, labels, folds, costs or execution.
- **Reviewed and proposed:** new TabICLv2/TabPFN/TiRex/forecasting-model additions, the next contextual gate design and later risk extensions.
- **Pending:** the faithful proposal experiment, completed prospective record, final report rubric and remaining tutor feedback.

The main reviewer feedback is addressed by specifying the gate's inputs, weight output, learning loss, chronology and candidate budget; defining the HMM intervention precisely; documenting all terms inside the report; and keeping the LLM research agent secondary. The formatting feedback is also supported by visual inspection: page 18's Section 3 paragraphs are materially tighter than page 9's methodology paragraphs. Correct paragraph styles when creating the next editable report, then render all pages. Student/project placeholders and unconfirmed workstream assignments also need resolution before submission.

## Useful files

- [Proposal alignment and missing work](PROPOSAL_ALIGNMENT_AND_GAPS.md): formulation, gaps, ablations and budget proposal.
- [Recent research](RECENT_MODELS_20261030.md): verified 2024–2026 model releases and how to adapt them.
- [Preliminary experimental results](SCREEN_RESULTS.md): measured outcomes and reproducible checks.
- [Repository evidence audit](REPOSITORY_AUDIT.md): inspected architecture, manifests and collector status.
- [Meeting summary](MEETING_SUMMARY.md): academic and administrative guidance from the transcript.
- [Classical controller literature](LITERATURE_CLASSICAL.md) and [advanced/risk literature](LITERATURE_ADVANCED.md): broader options, with sources and implementation limits.
- [Interactive research explorer](/Users/a/.cursor/projects/Users-a-Downloads-ml-trade-engine/canvases/controller-research-v1.canvas.tsx): proposal priorities, recent models, method comparison and measured curves. It is a static evidence snapshot with interactive navigation, not a live monitor.

The separate PR investigation was set aside at the user's request. Its archived notes are not part of the research recommendation or an action to merge code.
