# Controller research review v1

30 October 2026 · research development · previously inspected historical data

**Proposal-aligned update:** the subsequently supplied 29-page proposal defines the primary controller as a contextual convex mixture of fixed three-class expert forecasts, followed by a shared policy. Read `PROPOSAL_ALIGNMENT_AND_GAPS.md` first for that committed design. This broader review surveys options; utility heads, learned allocation and reinforcement learning are extensions rather than replacements for the core gate comparison. The completed screen uses next-close/fixed-threshold legacy labels, not the proposal's next-open/cost-and-volatility labels. The October rubric and additional tutor comments are still pending; this is research preparation, not the final monthly report.

## Recommendation

For the proposal, start with a **regularized softmax mixture gate**, then one small nonlinear gate using the same expert forecasts and context. Compare each with and without the HMM inputs while holding the policy fixed. Retain fixed mixtures, trailing class priors, trend, cash and a costed equal-weight portfolio as explicit controls. Logistic forecast stacking and delayed-feedback expert weighting provide additional controls; the stacking screen is not itself the proposed convex gate. Treat uncertainty and risk-policy changes as separate ablations. Direct utility heads and learned allocation remain later hypotheses.

This is a research-priority judgement based on the repository and literature, not a measured ranking of every architecture. The companion literature files explain the individual methods and link primary sources. The interactive comparison presents the same distinctions.

## What “controller” means here

At information cutoff t, expert models produce forecasts p(k,t). A forecasting controller can combine them into p(t), while a policy controller uses those forecasts, observable market context and current account state to propose long-only target weights. A separate risk layer checks or projects those targets into the feasible region. The execution engine accounts for reservations, delays, fills, costs and persistent state.

These roles have different targets. Predicting the correct direction, predicting that a proposed trade has positive net value, choosing the best expert and controlling whole-account risk are not interchangeable tasks. A binary “expert was correct” gate does not necessarily learn profitable switching. A controller trained to forecast direction is not yet a demonstrated allocation controller.

Recommended interface for a later implementation:

```
ControllerInput:
  decision_time, input_snapshot_hash, model_versions
  causal expert probabilities / forecast distribution
  mature past performance, market context, account state, pending reservations

ControllerProposal:
  expert_weights or selected_action, proposed_target_weights
  uncertainty/abstention reason, training_cutoff, latest_feedback_endpoint
  protocol_id, source/config hashes

RiskDecision:
  feasible_targets, binding_constraints, rejected_increases, persistent_halt

ExecutionReceipt:
  decision_id, orders/fills/rejections, cash/positions/reservations, reconciliation
```

A forecast-mixture weight is not an asset allocation. Switching between standalone expert equity curves also does not reproduce one investable account: holdings, cash and switching costs must be recomputed through the shared ledger.

## Repository findings and implications

- `src/models/adaptive.py` fits a standardized logistic classifier and shallow histogram-gradient-boosted classifier. It averages their probabilities, chooses a temperature, and compares against a trailing prior. Its original calibration and gate use the same block.
- `research_v2/experiments.py` contains the subsequent controlled implementation, including separated calibration and gate partitions. `research_v2/models.py` is a checkpoint loader, not a new controller architecture.
- `src/models/gating.py` contains a different logistic expert-correctness gate for the older XGB/LGBM/LSTM design. Its existence does not establish a comparable result under the validated v2 contract.
- `runtime_v2/policy.py` runs the trend shadow and keeps deployment in cash. Recorded classifier probabilities do not direct that shadow's trades.
- The audit verified all 42 files in the controlled-experiment v2 artifact manifest, five design-pinned sources and 848 model/prediction checkpoint pairs. These hashes establish artifact integrity; the separate partition checks support the inspected timing claims. They do not establish profitability. The historical results did not establish deployable superiority. Previously examined 2024–September 2026 history remains exploratory.
- The 22 symbols share market shocks and include overlapping ETFs. Rows, seeds and overlapping five-session labels do not supply independent repetitions of market history.

The current small prospective sample is useful for operation and failure analysis. It is unsuitable for selecting a large controller family or making a profitability claim.

## Candidate programme

**First stage: inexpensive identifiable changes.** Logistic stacking tests whether a learned mapping of genuinely out-of-time expert predictions improves probability quality. Logistic meta-labeling is a separate variant whose target is a proposed trade's outcome after specified costs. A shallow HistGB/LightGBM/XGBoost controller tests nonlinear interactions such as disagreement × volatility, with identical inputs and splits. Delayed Hedge tests whether recent revealed expert performance is useful without learning a large context model. Include a uniform mixture of the same experts to isolate the effect of adaptive weights.

**Second stage: context and risk.** Use a two- or three-state HMM as a past-fitted context feature or explicitly registered regime gate. Compute forward-filtered state probabilities; never label a past decision with a full-sequence smoothed state. Compare an HMM feature against simple observable trend/volatility buckets. Add a constrained optimizer or a fixed risk-sizing rule independently of the controller. Consider abstention based on calibrated uncertainty and measured coverage; do not assume conformal coverage survives dependence or distribution shift unchanged.

**Later stage:** a small GRU/LSTM may test whether temporal context adds something beyond lagged scalar features. A neural mixture-of-experts gate may add context-dependent weights but needs cross-fitted expert outputs and capacity controls. LinUCB/Thompson sampling require a clear bandit feedback problem; when every frozen expert's outcome is observable, full-information learning is a simpler formulation. PPO/SAC and offline CQL/IQL require a much stronger simulator, reward specification and state/action-support audit. TFT and foundation time-series models are candidate forecasters; their output does not remove the controller and execution problem.

An LLM may explain receipts or help navigate research. It should not override the deterministic financial controls, and no LLM trading-controller advantage has been established here.

## Why risk management is a CS component

Frame it as constrained decision-making and dependable stateful software. The methods section specifies the admissible actions; implementation explains enforcement and recovery; experiments show which constraints bind and what each control changes.

For example, propose weights by maximizing a declared forecast utility minus a covariance risk penalty and turnover/cost penalty, subject to long-only, gross exposure, position/group limits and executable cash. Model predictive control extends the planning horizon but executes only the first action and replans. That framework consumes forecasts; it does not establish their accuracy. [Boyd et al., Multi-Period Trading via Convex Optimization](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html).

Essential ablations are controller off/on with the same risk layer, then selected controller with one risk component changed at a time. Report net returns, drawdown/duration, turnover, cost, cash fraction, concentration, violations, rejected increases and latency. Compare risk and return jointly: moving to cash can reduce drawdown without showing predictive skill. A drawdown trigger cannot guarantee a maximum loss through market gaps or delayed fills. VaR/CVaR estimates are distributional summaries, not loss caps.

The risk layer must reserve cash for pending buys, avoid spending pending sells, distinguish HOLD from CASH/EXIT, preserve halts over restart, and reject stale inputs and invalid outputs. Those are testable engineering behaviors and belong directly in the implementation contribution.

## Evaluation design

1. Freeze the input universe, eligible timestamps, action definition, cost model, endpoints, hypotheses, splits, seeds and compute budget before each new comparison.
2. Generate meta-training rows from forecasts made without their outcomes. Fit scaling, feature selection, class-return mappings, covariance/regime models and calibration only on eligible past rows.
3. Purge by actual label endpoint. For the current label `close[t+6]/close[t+1]-1`, a decision cannot use a label whose endpoint is at or after its fitting cutoff. Online weighting must wait for that feedback too. Delayed-feedback learning is its own studied setting. [Joulani et al., 2013](https://proceedings.mlr.press/v28/joulani13.html).
4. Keep model fit, calibration, policy/gate selection and evaluation roles distinct. Proper probability scores include discrimination and calibration; a better log loss alone is not proof of better calibration. Use supported reliability plots as well. [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html).
5. Use chronological outer folds and inner-only selection. Preserve all attempted, failed and stopped trials. Do not repeatedly promote a winner on the same consumed audit.
6. Use paired date-level comparisons, retaining all symbols together inside bootstrap blocks; report the number of unique dates and sensitivity to block length. Register Holm families based on actual comparisons. Report effect sizes, intervals and practical thresholds, including inconclusive results.
7. Evaluate economics separately with matched capital, dates, costs, execution delay and constraints. Require a continuous ledger replay, not a weighted average of independent strategy returns. The predictive screen in this package does not perform that step.

## Interactive demonstration plan

The delivered model explorer supports selecting a candidate and inspecting its role, assumptions, evidence level and implementation priority. It is a research navigation tool.

The stronger subsequent demonstration is an **auditable historical replay**: select a date; see only inputs available then; inspect each expert forecast and the controller output; reveal which risk limits changed the target; step to the next eligible fill; reconcile the account; then reveal the matured outcome. Offer synchronized expert, equity/drawdown, exposure and cost views with visible sample counts and source versions.

A separate what-if mode can vary cost, delay or a registered risk limit, recompute the complete account and display the difference against the fixed baseline. Label it exploratory simulation. Changing a slider must never change a frozen historical receipt or suggest prospective performance. Include one losing case, one abstention, one constraint rejection and one stale-data failure, not only successful trades.

## Scope of this continuation

This package adds a literature review, meeting/report guidance, an interactive research comparison and an isolated registered predictive-controller feasibility screen. The screen result file records what actually ran. It does not claim a new production release, a completed economic controller trial, a final manuscript revision or a new untouched holdout.

Next experiment: register a comparable shallow tree controller, retaining this screen's same-expert uniform control and prior, using the established out-of-time meta-data contract. Before any allocation claim, implement the controller-to-target adapter in a new versioned module and test shared-ledger parity, cost sensitivity, restart behavior and fault cases. Freeze a genuinely future evaluation window only after selection; keep current pilot outcomes outside that selection process.
