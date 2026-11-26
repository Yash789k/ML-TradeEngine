# Controller research: optimization, uncertainty and sequential models

**Proposal-aligned scope update:** after this broad survey, the supplied proposal clarified that the primary study is a contextual gate over fixed classification experts with a shared policy. Utility heads, learned allocation, decision-focused optimization and reinforcement learning below are later extensions, not replacements for that question. See `PROPOSAL_ALIGNMENT_AND_GAPS.md` for the concrete core design.

Research date: **30 October 2026, Asia/Hong_Kong**. Literature and design review only; no new model was evaluated or promoted by this document. Rankings below are engineering/research judgments for this repository, not published performance rankings. Sources are original papers, conference proceedings, author pages or publisher pages checked during this review. The frozen collector, source seals and historical experiment records remain unchanged.

The strongest next step is a small learned decision layer behind explicit portfolio constraints. Start with a supervised estimate of the incremental utility of a few admissible actions, compare it with deterministic constrained allocation, and evaluate abstention separately. A large recurrent or reinforcement-learning controller should earn its complexity through controlled evidence.

## Repository evidence and the meaning of controller

Read before this review: `AGENTS.md`, `docs/research-brief/EXECUTION_PROMPT.md`, `docs/research-brief/EVIDENCE_AND_TEST_PLAN.md`, `docs/research-v2/EXPERIMENTS.md` and current `docs/research-v2/STATUS.md`. Inspected current implementations include `src/models/adaptive.py`, `src/research/adaptive_portfolio.py`, `src/risk/policy_engine.py` and `src/execution/account.py`; provenance examples include `data/research-v2/experiments/artifact_manifest_v2.json`, `data/research-outcomes-r7/reports/activation-20261027/manifest.json` and an archived r2 backup receipt. Inspection of these manifests is not a fresh verification of every referenced artifact.

Three roles need separate experiments:

| Role | Output | Example in the current repository |
|---|---|---|
| Forecast expert | Probabilities or a return distribution | Logistic and histogram-gradient-boosting experts in `AdaptiveModel` |
| Decision controller | Strategy mix, admissible action, exposure or target weights | Trend/model allocation and gating in `adaptive_portfolio.py` |
| Execution/risk contract | Feasible orders, reservations, fills and persistent state | `Account` and `ExecutionSession` |

The inspected adaptive model uses next-close-aligned five-session labels, a maximum 1,008-observation eligible history and small linear/tree experts. The separate controlled experiment uses distinct endpoint-purged calibration and gate blocks. Its documentation excludes legacy LSTM/XGB/LGBM/MoE outputs from direct comparisons because their old contracts differ. The old `PolicyEngine` contains rules for thresholds, size and eligibility; its presence is not evidence that the currently sealed collector uses that class. The active status also distinguishes the current outcome-only continuation from historical pilot declarations.

The research implication is to learn on top of a clearly specified action contract. Adding a more powerful forecaster alone does not establish a better controller. The archived pilot records are too limited to treat as a broad behavioral dataset for reinforcement learning; historical simulation remains the likely development environment and must be labeled accordingly.

## Ranked candidate set

| Priority | Candidate | Appropriate role | Why it fits, and what must be demonstrated |
|---|---|---|---|
| A | Constrained single-period optimizer with shrinkage risk estimate | Deterministic allocation/controller baseline | Small parameter count, explicit costs and constraints; improvement over current clipping/scaling must be measured |
| A | Ridge or shallow boosted-tree incremental-utility head | Learned controller over a finite action menu | Directly predicts benefit over the base action; avoids Bellman bootstrapping in the first experiment |
| A/B | Adaptive conformal or residual-quantile abstention | Wrapper around any forecast/controller | Useful hypothesis for uncertain decisions; interval coverage alone is insufficient to justify trades |
| B | Scenario CVaR-constrained allocation | Tail-risk controller | Direct tail-loss objective; sparse tails and scenario choice are major estimation limits |
| B | Tiny GRU or LSTM action-utility model | Recurrent controller | Tests whether recent sequences improve decisions beyond fixed trailing features; use one bounded architecture family |
| B/C | Model predictive control (MPC) | Multi-period allocation controller | Tests the value of planning turnover/cash ahead; needs a credible horizon model and an H=1 control |
| C | TFT or pretrained time-series model such as Chronos | Forecast expert, optionally feeding the same controller | Forecasting literature is relevant; controller benefit is a second, separate claim |
| C | Fitted Q iteration | Sequential value controller | Useful intermediate RL baseline only after state/action transitions are audited |
| D | PPO or SAC | Simulator-trained sequential controller | Requires many interactions and careful reward/cost/state modeling; replay episodes do not create new market histories |
| D | CQL or IQL | Offline sequential controller | Needs action/state support and qualified logged or simulated transitions; conservatism cannot create missing evidence |
| Exclude from first selection family | General-purpose LLM directly issuing allocations | Optional explanation interface later | No repository-specific evidence of incremental decision value; numerical validity, provenance and reproducibility remain unresolved |

Priority A means first research comparison; it does not authorize deployment. Do not fit every row in this table. Freeze a bounded subset after an implementation/runtime pilot that does not inspect outer outcomes.

## Constrained allocation and model predictive control

Boyd et al. formulate single- and multi-period trading as optimization over expected return, risk, transaction cost and holding cost. Their multi-period method plans a sequence and executes the first trade before replanning. The paper explicitly leaves forecast generation outside its contribution. It is therefore unusually well matched to the distinction between this project's experts and controller. [Boyd et al., *Multi-Period Trading via Convex Optimization*, 2017](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html).

**Proposed local experiment:** use the same eligible forecasts for every allocator and solve a long-only problem of the form

\[
\max_{w\in\mathcal F_t}
\hat\mu_t^\top w-\lambda w^\top\hat\Sigma_t w
-\kappa\lVert w-w_t^{\rm current}\rVert_1.
\]

`F_t` must represent current cash, pending reservations, position/group/gross limits and allowed turnover. Parameter units must agree with the forecast horizon; a five-session expected return cannot be silently combined with daily variance and annual risk coefficients. Model each actual transition through the existing execution contract. Record solver status, tolerance, constraint residuals, latency and a deterministic failure action. Infeasibility should be visible; the model must never widen hard limits to find a solution.

Compare current proportional clipping/scaling, turnover-aware single-period optimization and, only later, an H>1 receding-horizon variant. Hold the forecast and covariance inputs fixed for the first comparison. This identifies allocation benefit rather than conflating it with a stronger forecaster. Evaluate stressed covariance, forecast-error and cost scenarios; report an estimated risk limit as an estimate, not a guaranteed future volatility/drawdown bound.

For MPC, future forecast paths, rebalance times, transaction assumptions and terminal cost all need prior registration. H=1 is the mandatory ablation. Receding-horizon planning can model future costs, but inaccurate horizon forecasts can overwhelm the benefit. This is a project-specific reason to defer MPC behind a simple optimizer, not a universal statement that MPC underperforms.

## Scenario CVaR as a controller

Rockafellar and Uryasev give an auxiliary-variable formulation that makes scenario CVaR optimization tractable. For scenario losses `L_j(w)`, the empirical objective is

\[
\min_{w,\eta}\;\eta+\frac{1}{(1-\alpha)N}
\sum_{j=1}^{N}\max(L_j(w)-\eta,0).
\]

For linear portfolio losses this can be represented with linear constraints. The follow-up treatment handles general loss distributions, which matters for finite scenarios and ties. [Original optimization paper](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf), [Rockafellar and Uryasev, *Conditional Value-at-Risk for General Loss Distributions*, 2002](https://www.sciencedirect.com/science/article/pii/S0378426602002716).

**Proposed role:** a registered alternative risk penalty or constraint using causally available joint return scenarios. Compare against covariance-based sizing and current fixed caps under identical decision forecasts. Keep scenario dates aligned across all assets; independently sampling each ticker would destroy observed dependence. Synthetic stress scenarios may supplement observed history but must be separately labeled and their weights fixed before evaluation.

The present allocation code estimates covariance from 63 trailing dates. A 95% empirical tail at that length has nominal mass equivalent to only 3.15 observations; a 99% tail has 0.63. These are arithmetic illustrations, not recommended tail settings. A 99% CVaR number derived from such a short window can look precise while depending almost entirely on the worst observation. Register minimum scenario support, longer-window sensitivity and uncertainty. Do not assume a CVaR constraint guarantees a maximum realized loss or prevents gaps.

## Supervised incremental-utility and residual controllers

**Recommended first learned controller:** choose from a small action menu, such as retaining the base target, reducing its exposure, or using one fixed alternative expert mixture. Regress the cost-inclusive utility difference between each action and the base action on features available at the decision. Begin with a strongly regularized linear head, then one shallow tree model. Include current positions, cash/reservations, turnover required, matured expert performance and current forecast disagreement. The action set and all feature cutoffs must be registered.

For each eligible historical state, reconstruct each candidate action through the same simulator with a fixed holding/continuation policy, then attach its utility label only after the full outcome is available. A practical target is `net horizon return(action) - net horizon return(base)`; a risk penalty, if used, requires a fixed definition and coefficient. Use cross-fitted or genuinely out-of-time expert forecasts to train this head. Training it on in-sample expert predictions would reward optimistic expert behavior.

These labels are simulated counterfactuals under a price-taker/fill model. They are not observed real fills for every alternative action, and they are not independent observations simply because several actions share a date. The final controller still needs a continuous sequential ledger replay, because taking an action changes later cash, holdings and transaction costs. A one-period regression head should be called an action-utility model; it is not automatically an RL Q-function.

Fitted Q iteration repeatedly fits supervised regressors to Bellman targets using state/action/reward/next-state transitions. That extra step changes the scientific problem: future policy behavior and coverage now affect training targets. [Ernst, Geurts and Wehenkel, *Tree-Based Batch Mode Reinforcement Learning*, 2005](https://jmlr.org/papers/v6/ernst05a.html).

Residual RL has a precedent in combining a conventional controller with a learned correction in robotics. That motivates an architectural analogy, not a claim of established financial performance. Here the residual could be a bounded weight adjustment projected into the admissible set, with the unmodified base policy as an exact ablation. [Johannink et al., *Residual Reinforcement Learning for Robot Control*, 2019 conference paper](https://arxiv.org/abs/1812.03201).

## Uncertainty and abstention

Adaptive conformal inference updates the uncertainty procedure as outcomes arrive and targets coverage over time under distribution shift. EnbPI provides a different ensemble-based construction with approximate marginal coverage under stated time-series assumptions, including conditions on dependent errors. Neither is a proof that trades selected by an interval will be profitable. [Gibbs and Candès, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html), [Xu and Xie, ICML 2021](https://proceedings.mlr.press/v139/xu21h.html).

**Proposed experiment:** wrap the chosen return/utility forecast with a calibrated uncertainty interval and open new risk only if a predeclared lower-bound rule clears the cost/utility threshold. Compare no gate, the current confidence gate and the uncertainty gate using the exact same underlying predictions. Record both interval coverage (outcome inclusion) and action coverage (fraction of eligible opportunities acted on); these are different denominators.

Five-session labels arrive with a delay. Update residuals and adaptation state only when each label matures, preserve overlapping-label dependence and fit the calibration procedure on eligible past data. Do not import an immediate-feedback theorem unchanged into this delayed-feedback implementation. Marginal coverage across all forecasts does not establish coverage conditional on selected trades, instrument or stress regime. Evaluate these slices descriptively with counts and block uncertainty; watch for vacuously wide intervals and a gate that always chooses cash.

## Small recurrent controllers: GRU and LSTM

LSTM introduced gated memory for long dependencies; GRU arose in an encoder-decoder sequence model. Both provide sequence representations, not a portfolio-risk contract. [Hochreiter and Schmidhuber, *Long Short-Term Memory*, 1997](https://doi.org/10.1162/neco.1997.9.8.1735), [Cho et al., EMNLP 2014](https://aclanthology.org/D14-1179.pdf).

**Proposed role:** replace the utility head's fixed features with one short causal sequence encoder that sees past forecasts, matured errors, market features and portfolio state. Choose either GRU or LSTM initially, with one recurrent layer and a small hidden dimension; the exact lookback/width/budget belongs in the protocol. Include a linear model with the same lagged information so any gain is not merely access to more history. A pooled model may share information across instruments, but correlated assets do not multiply independent sample size.

Require consecutive observed sequence warm-up; no backfill from future bars. Fit scaling only on the training role. Reset or serialize hidden state consistently across folds, instrument changes and restarts, and test online versus batch inference parity. Sequence windows may contain eligible past context, but target endpoints must not cross role boundaries. Multiple seeds quantify optimizer variation, while date blocks quantify market uncertainty. The existing directional LSTM cannot be described as this controller without a new target, action interface and controlled evaluation.

## TFT and pretrained time-series models

TFT combines recurrent processing, attention, variable selection and quantile outputs for multi-horizon forecasting. Known-future inputs in that architecture mean inputs genuinely known at forecast time, such as calendars; future prices or later economic releases are not eligible covariates. [Lim et al., *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*, 2021](https://doi.org/10.1016/j.ijforecast.2021.03.012).

Chronos is a pretrained probabilistic forecasting approach based on tokenized/scaled time-series values and a transformer architecture. Its cross-domain forecasting experiments are a reason to consider a fixed pretrained expert, not evidence of tradable stock-return advantage. [Ansari et al., *Chronos: Learning the Language of Time Series*, 2024](https://arxiv.org/abs/2403.07815).

**Disposition:** optional forecast-expert research after simple baselines. Keep the controller and execution fixed while comparing predictive quality, then separately measure incremental net portfolio value. Freeze the exact checkpoint and preprocessing. Audit available pretraining-domain/date disclosures and possible overlap with evaluated instruments; unknown pretraining exposure is a limitation, and historical evaluation after unknown corpus exposure is not clean zero-shot evidence. Do not transfer forecast benchmark wins to financial alpha claims or treat attention weights as causal explanations.

## PPO, SAC and offline RL

PPO alternates interactions with an environment and policy optimization using a surrogate objective. SAC uses off-policy actor-critic learning with an entropy objective, particularly suited to continuous actions in its original benchmarks. These are general control algorithms; their benchmark evidence does not establish financial performance. [Schulman et al., *Proximal Policy Optimization Algorithms*, 2017](https://arxiv.org/abs/1707.06347), [Haarnoja et al., *Soft Actor-Critic*, ICML 2018](https://proceedings.mlr.press/v80/haarnoja18b.html).

Before either experiment, the state must expose positions, available/reserved cash, pending orders, drawdown state and information age; actions must map to feasible target changes. Costs, delayed fills, rejected orders, terminal positions and episode resets must match the comparison ledger. Use an explicit fixed reward rather than tuning toward an attractive equity curve. Never permit the optimizer to override feasibility. Replaying a historical path repeatedly creates more training interactions, not more independent market regimes. PPO/SAC are therefore deferred until simpler heads and simulator parity are established.

CQL addresses offline overestimation from policy/data distribution mismatch by penalizing Q estimates conservatively. IQL fits value/Q functions without querying out-of-dataset actions during training and extracts a policy through advantage-weighted behavioral cloning. Their assumptions and benchmark settings do not remove the need for data support in this project. [Kumar et al., *Conservative Q-Learning*, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html), [Kostrikov, Nair and Levine, *Offline Reinforcement Learning with Implicit Q-Learning*, ICLR 2022](https://openreview.net/pdf?id=EblVBDNalKu).

The key distinction is the source of transitions. If historical prices and a validated price-taker simulator can generate each admissible action's outcome, this is model-based counterfactual simulation, with fill/impact assumptions. If only a deterministic policy's logged actions are available, many alternative actions have no support. Importance weighting does not repair zero action probability. CQL/IQL can reduce some extrapolation errors; they cannot identify unobserved market impact or prove safety for unsupported portfolio states. Require action/state support diagnostics and behavior-cloning/simple-Q baselines before spending a deep-RL budget.

## General-purpose LLMs

Exclude direct LLM allocation from the first controller selection family. This is a scoped research decision, not a claim that language models can never help. Published work reports promising zero-shot forecasts in some settings, while later ablations of three LLM-based methods found that simpler replacements could match or improve their forecasting. These are different methods/settings and neither result certifies this trading pipeline. [Gruver et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3eb7ca52e8207697361b2c0fb3926511-Abstract-Conference.html), [Tan et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ed5bf446f59e2c6646d23058c86424b-Abstract-Conference.html).

An LLM could later explain already-recorded decisions using structured evidence, without choosing weights or changing controls. A direct-controller proposal would need a specific extra information hypothesis, fixed model/prompt/version, reproducible structured outputs, training-data exposure disclosure, latency/cost accounting and the same chronological comparisons. More fluent explanations are not evidence of better decisions.

## Minimum evaluation contract before any of these models run

1. **Freeze the question and subset.** Start with current rules, one constrained optimizer, one linear utility head and one shallow tree head. Add only one uncertainty wrapper as a distinct ablation. Register recurrent/RL variants as deferred rather than quietly expanding the search.
2. **Declare information roles.** Fit experts, controller, risk estimates, calibration and selection on explicit past partitions. Train the controller on out-of-time expert predictions. Purge according to complete utility/label endpoints. Log every previously inspected period; existing history remains development/exploratory evidence.
3. **Pin state and units.** Specify portfolio-level versus instrument-level actions, current/pending capital, adjusted-return proxy units, next-close execution, rebalance calendar, terminal accounting and no-trade behavior. A confidence score is not a cash quantity.
4. **Select on economic value.** Predeclare a cost-inclusive paired portfolio endpoint and practical effect threshold. Keep return, drawdown, tail loss, turnover, exposure, constraint breaches and trading coverage visible so a risk-adjusted summary cannot conceal cash-only behavior or large losses.
5. **Use matched baselines.** Include cash, costed equal weight, trend, current frozen controller and a fixed mixture. Compare equal dates/capital/costs/risk limits. Include same-information ablations for sequence context, optimization and uncertainty gating.
6. **Respect dependence and selection.** Resample contiguous dates while keeping the cross-section together; register primary block length and sensitivities. Count all tested controller configurations in the comparison family. Seeds and alternative-action labels are not new market observations.
7. **Bound compute before outcomes.** Record wall-clock/fit/seed limits, search list, early stopping using inner data only, hardware, failure rules and checkpoint hashes. Failed or stopped runs stay in the registry.
8. **Prove behavior.** Check future-row mutation invariance, action feasibility, cost arithmetic, stale inputs, solver failure, state serialization, retry idempotency, pending reservations, delayed label updates and sequential replay. Bind evidence to the new source/config/data version.
9. **Demonstrate the mechanism.** An interactive research replay can show expert forecasts → proposed action → binding constraint → accepted target → eventual cost/outcome, with chronological reveal and clear historical labels. It should expose adverse cases as readily as favorable cases. Risk controls belong in algorithm/design/implementation sections and ablations, with remaining market/model risk discussed separately.

The publishable question is whether context-dependent decisions add reliable net value under the same risk and execution contract. A negative result can still contribute through an auditable action interface, leakage controls, dependent-sample evaluation and an interactive explanation of why the controller abstained or reduced risk.
