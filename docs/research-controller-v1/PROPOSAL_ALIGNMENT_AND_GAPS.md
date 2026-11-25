# Proposal alignment, controller formulation and remaining gaps

**30 October 2026.** This is a design review and proposed bounded experiment, not a new registration, model run, monthly report or approval to deploy. It incorporates the supplied proposal and the review priorities: settle the learning formulation, isolate regime ablations, specify resources, make the report self-contained and keep LLM work subordinate. The final monthly-report rubric and any additional tutor feedback remain pending.

Source: `/Users/a/Downloads/fyp-proposal-sept9-formatted.pdf`, 29 pages, SHA-256 `3546f5ed2675b6a45ef41b524dfcb02449b269ea3aee7a737438bf6b816d741b`. The extracted page-delimited text is `data/research-controller-v1/proposal-review-v1/proposal.txt`. Relevant sections are the objectives on pages 3-6, design on pages 9-10, implementation/evaluation on pages 11-17, planning/resources on pages 18-20 and Appendix B on pages 24-29. The original PDF was not edited.

## The central correction to our first review

The proposal's committed controller is a **contextual mixture of classification experts**. It emits weights over experts, not stock weights. A fixed policy converts the resulting UP/FLAT/DOWN probabilities into positions, and deterministic controls enforce admissible trades. A directly learned allocation, action-utility head, decision-focused optimizer or PPO policy is a different scientific intervention. Those ideas remain useful extensions in `LITERATURE_ADVANCED.md`; they should not replace the proposal's routing question without an explicit scope amendment.

The local October screen is useful preliminary work but does not implement that full proposal. Its logistic stack directly maps nine expert log probabilities to a new three-class distribution. It is not constrained to be a weighted average of the original experts. Hedge does produce an expert mixture, but its weights depend on accumulated matured losses rather than HMM/market context. Neither uses the proposed rolling HMM in a controlled on/off comparison, and neither screen model has been tested as an allocator through the shared account.

## What the proposal actually commits to

| Topic | Proposal specification | Consequence for the next study |
|---|---|---|
| Research question | Learned regime/context-dependent expert weights versus individual classifiers and a static ensemble; overall and worst-regime risk-adjusted outcomes (pp. 3, 5, 16-17) | Freeze experts and policy while changing only routing |
| Forecast target | UP/FLAT/DOWN from the next open to the open five sessions later, threshold `max(round-trip cost, lambda × horizon-scaled trailing volatility)`; one-session sensitivity (pp. 5, 24) | Existing next-close, fixed-1% predictions are not interchangeable labels |
| Experts | Logistic, XGBoost, LightGBM and LSTM; text/LLM experts separately controlled (pp. 3, 9) | Current qualified logistic/HGB experts are a smaller feasibility pool; legacy saved scores do not close the comparison |
| Controller action | A probability-simplex weight vector over expert forecasts at each stock/date (pp. 9-10) | Implement an explicit weight-producing gate and test simplex/availability behavior |
| Context | Soft HMM posteriors, persistence, volatility, range, trend, drawdown, sentiment and matured expert performance (p. 9) | Admit only context with availability evidence; missing text must not be fabricated |
| Regimes | Rolling past-fitted HMM; state number chosen using validation; state correspondence across refits (p. 9) | Existing prefix-fit hard-state context is not the complete rolling-posterior design |
| Policy | Same probability-to-position mapping across controllers; dynamic threshold/stops/caps fixed after validation; separate static-policy comparison (pp. 10, 25-26) | Re-tuning each controller's policy would confound routing with allocation |
| Objective in Appendix B.4 | Growth minus downside and increasing-drawdown penalties for choosing policy settings (p. 26) | This is not yet a specified loss for learning the MoE gate |
| LLM research agent | Later phase, at most three proposals per review; equal data/model/trial budgets against scheduled retraining and search (pp. 6, 24-25) | Keep out of the primary routing experiment; all proposals, failures and activation delays count |
| Prospective operation | Intended October-April observation period (pp. 6, 16-19) | A schedule intention is not completed evidence; use actual unique receipts and disclose interruptions |

The opening-price target can be written precisely as `Open[t+6] / Open[t+1] - 1` for the five-session version; the one-session sensitivity is `Open[t+2] / Open[t+1] - 1`. Availability must use actual timestamps. A label observed at an opening may be available before that day's closing decision if its data has arrived; alternatively, keep a declared conservative prior-session rule. Do not silently switch between timestamp eligibility and calendar-date eligibility.

## Implemented locally, published remotely, and still proposed

| Evidence tier | Verified position on this date | What must not be inferred |
|---|---|---|
| Published GitHub `main` | Snapshot `f194283bb6104e3feb980a67bc05a06459677909`, matching local HEAD; the audit found no PRs and only `main`. The published prototype still differs from repaired local source. | A link to `main` does not expose all local repairs, research-v2 artifacts or the controller screen |
| Current local working tree | Separate qualified research/runtime, historical evidence, interrupted prospective continuations and the new screen exist locally. The audit found numerous uncommitted/untracked changes. | Local tests do not prove the GitHub workflow or published snapshot is qualified |
| Frozen operational path | Deployment cash and a trend shadow; classifier forecasts recorded as diagnostics. Current continuation is outcome collection, not activation of a learned allocation controller. | No claim that the new gate currently drives Alpaca, broker orders or live stock allocations |
| Completed new screen | Two logistic meta-fits, delayed Hedge and fixed controls on consumed historical forecasts; immutable results and independent causality checks exist | Not a contextual MoE, open-to-open label study, regime ablation or portfolio result |
| Proposed next work | Contextual simplex gate, exact regime ablations, matched execution experiment and later faithful target rebuild | No result or implementation claim until separate versioned evidence exists |

Remote evidence is in `data/research-controller-v1/github-review-v1/github-evidence.json` and the companion published-source audit. Local implementation/evidence details are in `REPOSITORY_AUDIT.md` and `SCREEN_RESULTS.md`. This document does not treat proposal statements about summer prototype performance or daily Alpaca operation as verified current facts.

## Concrete primary controller formulation

For stock `i`, decision `t` and fixed expert `m`, let `p_m(i,t,c)` be its out-of-time probability for class `c`. Let `z(i,t)` contain only the approved context available at the cutoff. Use a shared small linear softmax gate:

\[
g_\theta(i,t)=\operatorname{softmax}(Bz(i,t)+b),\qquad
q_\theta(i,t,c)=\sum_{m=1}^{M}g_{\theta,m}(i,t)p_m(i,t,c).
\]

This is a convex mixture of expert distributions even though fitting a context-dependent gate need not be a convex optimization problem. The proposal does not require a large neural gate. A shared pooled gate is a concrete starting choice; the date-level loss prevents 22 correlated stocks being described as 22 independent online rounds:

\[
\mathcal L(\theta)=\frac1{|\mathcal T|}\sum_{t\in\mathcal T}
\frac1{n_t}\sum_i-\log q_\theta(i,t,y_{i,t})
+\lambda_g\lVert B\rVert_F^2.
\]

Use complete eligible date groups or explicit missingness rules. Scaling is fitted on the gate-fitting partition. Expert forecasts used to train the gate must themselves have been produced out of time. Proposed initial numerical choices are probability clipping at `1e-12`, class order DOWN/FLAT/UP, `lambda_g` in `{0.001, 0.01}` under the mean-loss normalization above, and full-date-batch Adam with learning rate `0.01`, maximum 500 steps and inner-validation patience 25. Save the selected step count and reuse it for the full-history refit; no outer outcome controls stopping. These are design choices to freeze after a development runtime pilot, not empirically selected winners. Do not describe a normalized expert-correctness score or the existing unconstrained logistic stack as this exact objective.

**Initial context:** trailing realized volatility, normalized daily range, trend, asset drawdown, and a fixed trailing window of each expert's matured log loss and accuracy. In the HMM arm add filtered state probabilities and declared persistence/entropy summaries. Sentiment, macro vintages and regime-memory trading results enter only in later registered additions with eligible evidence. Forecast probabilities already enter the mixture; adding them as gate features is a separate choice, not an implicit extra information source.

**Training versus evaluation:** learn the gate with proper probability loss, then assess its decisions through the identical fixed policy. Overall and worst-regime net Sharpe remain the proposal's main economic outcomes. The two objectives need not agree; report that disagreement rather than retuning the gate to the observed portfolio headline. A learned utility head or differentiable portfolio loss would be a separate later comparison.

**Controls:** each individual expert, uniform weights over the same expert pool, past-fitted constant mixture weights, delayed Hedge, the existing unconstrained logistic stack, and a rolling class prior. Cash, costed equal weight and trend enter the economic comparison. Select any “best single expert” using eligible training/validation data only. A hindsight per-regime expert is an oracle diagnostic, never an attainable baseline or a bound on all mixtures.

**Weight constraints:** nonnegativity and unit sum are mandatory. If minimum weights are imposed, use an exact construction such as `epsilon + (1-M*epsilon)*softmax(logits)`, with registered `epsilon < 1/M`; flooring before normalization does not preserve the requested floor. Gate smoothing or limits on changes in expert weights are optional distinct ablations. They are not substitutes for limits on portfolio turnover.

## Regime ablations that answer identifiable questions

The proposal routes HMM information through several paths: expert features, gate context and dynamic policy rules. Removing HMM only from the gate tests its incremental value **at the gate conditional on the remaining system**. It does not establish the total contribution of regime information.

| Arm | Expert forecasts | Gate | Policy | Main interpretation |
|---|---|---|---|---|
| A | Frozen common pool | Fixed same-pool mixture | Frozen common policy P | Routing baseline |
| B | Identical to A | Learned gate, explicit HMM-derived context removed | Same P | Contextual routing versus fixed weights: B-A |
| C | Identical to A/B | Same gate architecture plus HMM posterior/context | Same P | Incremental explicit gate-regime input: C-B |
| D | Identical to C | Frozen fitted C | Registered static policy P0 | Policy change conditional on the same controller: C-D |
| E, optional | Identical to A | Fixed mixture | Same P0 | Difference-in-differences for controller-policy interaction |
| F, separate study | Experts rebuilt without HMM-derived features | No HMM in gate | No HMM in policy | Total-system regime removal; requires new expert fits and its own budget |

In B, remove posterior probabilities, hard states, transition flags, posterior entropy and regime-conditioned performance memory together. Retain raw volatility/trend context already shared by both arms. In C, those same raw context variables remain, so the test asks whether the HMM representation adds value beyond directly observable context. Keep the expert pool, training dates, regularization search, seeds and policy constant. If experts or P already depend on HMM, disclose that retained path explicitly.

For a low-cost precursor using current logistic/HGB experts, which do not require HMM features, a fixed non-regime policy can isolate the new gate path particularly cleanly. Label that choice as a restricted feasibility design rather than silently replacing the proposal's dynamic-policy experiment.

Regime-state fitting must be causal. For each context date, HMM parameters and feature scaling come from eligible prior observations; emit forward-filtered posteriors, not full-path smoothed/Viterbi labels. Relabel states using training-period quantities only and record the mapping. Use the same saved regime tape for every controller and every regime-conditioned result. A future-row mutation test must cover HMM refits as well as gate inputs.

**Regime robustness is not regime transfer.** Evaluating later bull/bear/volatile episodes is temporal generalization conditional on estimated states. Holding out an instrument is an additional cross-sectional transfer task; training transformations, gate labels and calibration must exclude its outcomes. Do not randomly split observations within regimes or describe economically overlapping ETFs as independent markets.

For regime metrics, retain the continuous account and the original chronological return series. Assign regime membership from the decision-time signal associated with each return interval using one registered lag convention. Do not splice disjoint regime dates into a fictional investable equity curve. Compute conditional return summaries with episode/date support; bootstrap contiguous original date blocks jointly with regime indicators. Recompute each method's worst-regime statistic within each replicate rather than selecting a favorable regime after observing results. Sparse regimes can make the proposed worst-regime success criterion unassessable; report that outcome instead of treating absent or undefined Sharpe estimates as zero.

## Two contracts, not one relabeled experiment

**Feasibility contract using existing artifacts:** preserve next-close entry, five-session return ending at `t+6`, fixed +/-1% classes, the qualified 22-symbol data and the original forecast files. Existing 2024/2025 observations remain consumed exploratory history. This enables cheap gate/HMM engineering and a fair routing comparison without regenerating experts.

**Proposal-faithful contract:** create a new data/label/protocol version for next-open execution, the cost/volatility threshold and the named expert pool. Audit adjusted-open construction, corporate actions, executable units, gap returns, decision timestamps and next-open fill accounting. Generate fresh out-of-time forecasts under that same target before training the gate. Use one common label multiplier selected from earlier development data, and freeze it across all compared models. A one-day sensitivity needs its own label maturity rules and forecasts.

The original targets and fills must not be patched in place or described as already equivalent. If the team ultimately chooses next-close as the main design because it matches the tested product, record a visible proposal amendment with its reason and scope. Either choice is defensible when evaluated honestly; combining close-trained forecasts with open-to-open outcomes without acknowledging the mismatch is not.

## Bounded resource plan to register

The following is a concrete proposed ceiling, not a claim that these experiments were run. Freeze exact configurations and measured hardware after a small development-only runtime pilot. If the pilot projects a ceiling violation, defer an optional arm before evaluating outer outcomes; do not expand the budget after seeing a promising result.

| Stage | Proposed budget | Output and stopping rule |
|---|---|---|
| Existing screen | Already completed: 0 base fits, 2 real meta-fits; about 4.96 seconds in its recorded environment | Preserve as preflight; do not reuse its performance as an untouched test |
| Gate-only runtime pilot | At most 2 synthetic/development gate fits, 2 HMM fits and one short account replay; 10 CPU-minutes hard cap | Validate shapes, memory, latency and provenance; no outer selection |
| Core gate comparison | Two HMM-context arms × two fixed `lambda_g` candidates × two chronological cutoffs = 8 inner-selection fits; two arms × three registered seeds × two cutoffs = 12 full eligible-history refits; 2 constant-mixture fits: **22 gate/mixture fits maximum** | Seed 42 is primary; 137 and 2026 diagnose optimization variation, not extra market samples |
| Causal regime tape | Fixed three-state diagonal-covariance market HMM; four 2024 quarterly origins plus two 2025 half-year origins, three deterministic initialization seeds each: **18 HMM fits maximum** | Fit on preceding data; pick initialization by training likelihood, never controller returns; failed convergence is recorded |
| Core resource ceiling | At most 2 CPU threads, 4 GiB resident memory, 30 CPU-minutes soft/60 CPU-minutes hard, no GPU/API/download requirement; gate fit capped at 500 optimizer steps | Actual performance must be measured. First time budget is reached, stop visibly and retain partial/failed evidence |
| Core economic replay | At most 10 primary policy streams × 2 historical half-years × 3 scenarios (base, doubled costs, one-session additional delay): **60 account histories maximum**; common initial capital, continuous within-block state, terminal holdings recorded | No new model fits for frozen-forecast cost/delay stress; replay the full policy/account so changed execution can change later feasible holdings |
| Faithful new-target expert pool, conditional later stage | 22 symbols × 8 quarterly origins across 2024-2025 × 4 named expert families = 704 fits; two extra LSTM seeds add 352, for **1,056 base fits maximum** after label definition is frozen | Includes 2024 OOF gate history and 2025 exploratory outer forecasts. No multi-package hyperparameter grid. Early stopping/calibration use eligible inner data |
| Larger-stage resources | Proposed hard ceilings: 8 CPU-hours, 16 GPU-hours if verified hardware is actually available, 10 GiB new artifacts; zero paid calls in the core experiment | Pilot must establish feasibility first. No GPU reservation, paid service or historical text coverage is assumed |

The core feasibility design uses two outer cutoffs consistent with the current cheap screen; HMM context construction on 2024 history is itself chronological. The latest 63 eligible dates before each cutoff can serve as inner validation, with fitting labels purged against its beginning. Refitting on all eligible history after selecting a hyperparameter is permitted only for the later outer block. Store every inner boundary, selected hyperparameter, fixed training duration, normalization statistic and seed. Additional probability calibration, two/four-state HMM sensitivity, neural gate, horizon variant or text expert adds trials and must receive a separate budget amendment before evaluation.

For the 1,056-fit ceiling, three LSTM seeds are repeated optimization runs of the same market samples. Their forecast outputs may be reused for every controller rather than retraining experts per gate or policy. If LSTM is infeasible under measured hardware/time, document its deferral rather than pretending the named four-expert objective is complete. If a label multiplier search requires fitting experts, those fits must be added explicitly before registration; they are not free preprocessing.

## Statistical and reporting decisions still to freeze

1. **Primary economic family:** at minimum, learned gate versus static mixture on overall and worst-regime net Sharpe, and HMM-on versus HMM-off on the same two outcomes. Register the full comparison family before evaluation. Predictive log loss is a separate endpoint/family. Report practical effect thresholds as well as intervals.
2. **Return conventions:** fix the risk-free/cash convention, daily versus annualized units, autocorrelation adjustment, costs, terminal marking and the account reset policy. Do not join separated evaluation blocks into an unbroken CAGR without saying capital was reset.
3. **Dependence:** paired resampling of original date blocks with all assets, forecasts, actions and regime indicators kept together. Fix primary block length and sensitivities. The current screen's marginal 95% intervals and Holm-adjusted p-values are different objects; do not label those intervals family-wise adjusted.
4. **Sparse support:** predeclare minimum dates/episodes for regime claims and report actual support. A 99% tail has only nominal 2.5 observations in 250 dates, before dependence; do not claim a precise extreme-loss estimate from the current screen. Separate historical stress cases from frequency estimates.
5. **Multiplicity:** the proposal mentions DSR, but incomplete historical search records cannot support an invented trial count or correlation estimate. Use the complete new registry and a defensible prespecified correction; treat any DSR as supplemental only when its inputs and assumptions are available. Preserve the earlier selection history and consumed-period labels.
6. **Acceptance:** enforce risk/operational validity first. A non-significant or underpowered routing result still answers the FYP question within its scope. No automatic promotion from predictive log loss, descriptive accuracy or a single favorable regime. Pilot dates already observed cannot be recovered as a fresh holdout.

## Recent research that sharpens the design without expanding core scope

The decision-focused literature supports distinguishing probability accuracy from downstream utility. SPO defines a loss based on decision regret and a tractable surrogate for specified optimization problems. It motivates a later separate training objective; it does not show that the current probability gate is already optimizing net Sharpe. [Elmachtoub and Grigas, *Smart Predict, then Optimize*](https://arxiv.org/abs/1710.08005).

DFF, published at AAAI 2025, adds a constrained bias-correction module near an existing predictor and studies decision-focused fine-tuning with limited data. It is a closer later extension to frozen experts than replacing the entire system with RL, but its trust region and optimization cost add assumptions and hyperparameters. Evaluate it only after the fixed-gate/policy reference is complete. [Yang et al., *DFF*](https://ojs.aaai.org/index.php/AAAI/article/view/34891).

IJCAI 2026 work uses Lagrangian decomposition to reduce the cost of decision-focused learning on larger optimization problems, including quadratic portfolio benchmarks. The relevant lesson here is to count solver calls and model-training cost; a 22-asset FYP need not adopt a large-scale method before simpler optimization has been timed. [Eilles-Chan Way et al., 2026](https://ijcai.org/proceedings/2026/247).

ICLR 2025's *Utility-Directed Conformal Prediction* constructs classification sets using downstream costs while retaining its stated coverage guarantees. It reinforces that uncertainty should be useful for decisions. Its classification/coverage results do not automatically extend to dependent, delayed stock-return labels or guarantee selected-trade profitability. [Cortes-Gomez et al., author-hosted paper](https://www.erichorvitz.com/Decision_Focused_Uncertainty_Quantification.pdf).

New foundation classifiers and time-series models reviewed in the companion literature can be expert candidates under a separate fixed budget. Architecture recency is not an experimental endpoint. LLMs remain optional direction experts or a slower research assistant; neither role receives authority to change live weights, risk limits or evidence records.

## Self-contained report and presentation repairs

The proposal repeatedly relies on “our presentation” for feature definitions, dynamic policy rules and algorithms. The next report must include these definitions, the exact label/execution timeline, gate equation/loss, role partitions, regime construction, costs, constraints, fit counts and resource limits directly. Figures should trace one decision from information cutoff through expert outputs, gate weights, risk projection, fill and label maturity. Include the actual evidence limits and negative results beside the methods they qualify.

Keep three claims visibly separate: published prototype, repaired/local research implementation and future proposal. The published GitHub URL must not stand in for a local artifact manifest. Likewise, intended October-April paper trading must not be reported as six months collected; cite unique valid receipts and missed-event evidence.

Section 3 on page 18 is visibly tighter in its paragraph spacing than the earlier methodology; normalize body paragraph/line spacing and table padding when the editable source is revised. Cover/student/project identifiers, workstream owners and Appendix A remain placeholders requiring verified details. Do not invent names, IDs, meeting approval or signatures. The final monthly report and its rubric mapping should be completed only after the promised rubric/tutor material is available; this document supplies the research alignment, not that final submission.
