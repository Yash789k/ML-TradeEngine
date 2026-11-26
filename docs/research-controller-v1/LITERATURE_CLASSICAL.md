# Statistical and tabular controller models

**Proposal-aligned scope update:** the later-supplied proposal makes the primary controller a convex mixture gate over fixed three-class experts. The binary meta-label and unconstrained forecast-stack options below are comparisons/extensions, not that exact gate. See `PROPOSAL_ALIGNMENT_AND_GAPS.md` for the primary formulation and `SCREEN_RESULTS.md` for the subsequently completed preliminary screen.

Research date: 30 October 2026, Asia/Hong_Kong. This is a literature review and proposed research design, not a fitted-model result or authorization to change the frozen pilot. Sources below are primary papers, author-hosted papers, or official project documentation. Priorities are engineering judgments for this repository, not performance rankings established by those papers.

## Recommendation

Start with a strongly regularized logistic controller, one shallow gradient-boosting challenger, and a delayed-feedback Hedge baseline. Use a filtered HMM as an optional context ablation. Treat a small mixture-of-experts gate as a direct extension of existing code. Defer contextual bandits until the project specifies why feedback is genuinely partial and how action-dependent portfolio state will be evaluated. Studying three boosting packages is useful; running a large three-package search is not yet justified.

The controller should answer three distinct questions: which qualified expert to trust, whether to abstain, and how much of the permitted exposure to request. A separate deterministic risk layer must retain authority over capital, concentration, pending cash, turnover, stale inputs and halts. Controller confidence must never authorize bypassing a risk constraint.

## What exists in this checkout

Read first: `AGENTS.md`, `docs/research-brief/EXECUTION_PROMPT.md`, and `docs/research-brief/EVIDENCE_AND_TEST_PLAN.md`. Inspected current source rather than treating its comments or historical status text as test evidence:

- `src/models/adaptive.py:81`: fits regularized logistic regression and histogram gradient boosting; averages their probabilities 50:50; chooses a temperature from a fixed grid; compares calibrated mixture log loss with a prior on the same calibration block to set one Boolean gate. That is not a contextual expert selector. The shared calibration/selection block remains an explicit limitation for claiming an independent gate benefit.
- `src/models/gating.py:180`: fits one regularized logistic correctness model per expert and normalizes positive scores into weights. The target is classification correctness, not costed utility. The numerical floor is applied before normalization, so the final weights are not guaranteed to exceed that same floor. This module is a useful baseline, not evidence that it improves portfolios.
- `src/models/gating.py:109`: performance context shifts rolling accuracy by `horizon + 1`. A new controller should use explicit outcome-availability timestamps rather than assuming every input row corresponds to an uninterrupted daily session.
- `src/features/regime.py:86`: already performs forward filtering with frozen HMM parameters. `add_hmm_regime` fits an initial prefix and leaves that fitting prefix unknown. The literature warning about full-sequence smoothing below is a regression guard, not an allegation that this code currently uses smoothing.
- `requirements.txt` names scikit-learn, XGBoost, LightGBM and hmmlearn; CatBoost is absent. These declarations do not establish installed versions or benchmark timings.

Source fingerprints inspected for this review:

| File | SHA-256 |
|---|---|
| `src/models/adaptive.py` | `17b17d73328b95f3e71ff5c7a0610e244c7a52254e30c022a8a7fe8a3621a071` |
| `src/models/gating.py` | `ee13e74a26140f6cdbb3dc128b5069e6f6aa8961e2f85b9dc97215fa8ac78f06` |
| `src/features/regime.py` | `f753a57f56292def1ca4000c9faaa684b62186cc5d7c8fe52f9bc615c5858c07` |
| `requirements.txt` | `97c5a96c650420daf9ab8b141787645906241b54446eb0e80b64037cecb458ec` |

Also read the original release manifest and the SPY model metadata for configuration context; this review did not reverify every release file or count prospective receipts. It makes no new claim about operational coverage or historical trading performance.

## Candidate comparison

All rows describe proposed controller uses; paper results from other applications are not comparable trading results.

| Candidate | Control output | Why test it here | Main limitation | Priority |
|---|---|---|---|---|
| Fixed weights, no learned gate | Fixed expert mixture; fixed acceptance rule | Necessary control for whether learning adds anything | Cannot adapt to conditions | Mandatory baseline |
| L2 logistic meta-label | Probability a proposed action passes a predefined costed success criterion | Small feature set, inspectable coefficients, inexpensive | Linear log odds; win probability omits payoff size | First |
| Elastic-net logistic meta-label | Same, with sparse coefficients | Tests whether reducing redundant context improves stability | Another tuning dimension; correlated-feature selection can be unstable | Bounded alternative |
| Histogram gradient boosting | Acceptance probability or expected utility score | Already available; limited nonlinear interactions | Easy to overfit short chronological histories | First nonlinear control |
| XGBoost | Acceptance probability or action-value estimate | Existing dependency; explicit regularized tree objective | Capacity and early-stopping choices must be counted | One challenger option |
| LightGBM | Same | Existing dependency; efficient tabular training | Unrestricted leaf-wise growth needs tight capacity limits | Alternative to XGBoost |
| CatBoost | Same; potentially pooled sector/instrument context | Useful categorical-feature hypothesis | New dependency; ordered boosting is not temporal leakage protection | Conditional challenger |
| Gaussian HMM context gate | Filtered state probabilities feeding a gate | Tests whether persistent latent context helps | State meanings and emission assumptions can be unstable | Context ablation |
| Logistic mixture-of-experts gate | Nonnegative weights summing to one | Direct extension of existing gate; shows changing expert reliance | Correlated experts and target mismatch can erase benefit | First-stage extension |
| Hedge / exponential expert weights | Evolving probability-mixture weights | Simple online adaptation with very few parameters | Delayed feedback and switching costs alter the trading interpretation | First online control |
| LinUCB | Discrete expert/action choice with uncertainty bonus | Interpretable contextual exploration baseline | Linear reward model; partial-feedback and sequential-state assumptions | Later sandbox |
| Linear Thompson sampling | Randomized expert/action choice | Alternative uncertainty-driven exploration | Prior/noise assumptions, seed variation and exploration losses | Later sandbox |

### Logistic meta-labeling and exposure

Fit `q_m(x) = sigmoid(beta_m · x)` for a fixed expert's proposed long action. Define its binary outcome before any selection: for example, whether an execution-aligned, costed hypothetical action outperforms a registered reference over the fixed horizon. Train on earlier out-of-sample expert proposals and matured outcomes. Regularized binary and multinomial logistic models are established convex estimation tools; L1/L2 and elastic-net penalties are described by Friedman, Hastie and Tibshirani. [Primary paper, 2010](https://www.jstatsoft.org/article/view/v033i01).

The proposed application is a meta-label design: the first model proposes an action and the second accepts or rejects it. A probability of a positive outcome is not expected profit: 80% small wins can coexist with large expected losses. Keep the first experiment to accept/abstain at a fixed permitted size; evaluate continuous sizing later against fixed-size and volatility-scaled controls. If utility regression is introduced, register payoff, downside and turnover terms separately rather than silently replacing the binary target.

Suggested initial context is compact: expert probabilities, disagreement, trailing realized volatility, drawdown, expert model age and matured trailing expert loss. Standardize inside each training fold. Do not fit on the same predictions used to fit the expert, nor select a confidence threshold from the final evaluation curve. Do not oversample positives or apply class weights and then present uncorrected scores as calibrated probabilities.

### Boosted tabular controllers

XGBoost supplies regularized tree boosting and efficient sparse/approximate training; its systems paper does not establish financial utility. The repository already lists the package, so shallow XGBoost is a plausible nonlinear comparison if its installed version is pinned. [Chen and Guestrin, 2016](https://arxiv.org/abs/1603.02754).

LightGBM introduces gradient-based sampling and exclusive feature bundling for efficient boosting. Its large-data speed results are not a reason to assume a benefit on this small controller table. Limit leaves, minimum leaf support and boosting rounds explicitly, and compare under the same runtime budget. [Ke et al., 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html).

CatBoost's ordered boosting and categorical handling address prediction shift from particular uses of training targets. They do not make random financial cross-validation valid. [Prokhorenkova et al., 2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html). Official `has_time` documentation states that object order can replace random permutations. Still purge chronological boundaries by outcome availability; for an initial controller, numerical features and fold-local one-hot categories avoid implying that ordered categorical statistics alone enforce point-in-time eligibility. [CatBoost parameters](https://catboost.ai/docs/en/references/training-parameters/common#has_time).

Project proposal: start with existing histogram boosting, then admit only one external boosting library to the first bounded comparison. CatBoost earns a separate trial if a pooled categorical-context hypothesis is registered. Avoid adding asset identifiers merely to let the controller memorize asset-specific historical winners; include an instrument-transfer evaluation if pooling is claimed to generalize.

### HMM regime context

HMMs represent hidden states and state transitions with an observation model. Forward filtering estimates the current state using information available so far; smoothing conditions on later observations. The underlying method is described in Rabiner's tutorial. [Rabiner, 1989, author paper hosted by Columbia](https://www.ee.columbia.edu/~sfchang/course/svia-F03/papers/rabiner-hmm-tutorial-89.pdf). Official hmmlearn documentation explicitly describes Viterbi and MAP decoding in terms of all emissions; using these over an entire test path would not establish historical as-of states. [hmmlearn API](https://hmmlearn.readthedocs.io/en/stable/api.html).

Project proposal: test two or three filtered state probabilities as extra context, not as financial truth labels. Begin with returns and volatility, low dimension, fixed refit schedule and fixed covariance choice. Learn the semantic state ordering from training data only; record remapping at every refit. Compare with simple observable volatility/trend context at equal information cutoffs. Report state occupancy, transition counts, filter entropy and convergence failures. A state named “bull” need not imply positive future net returns. The existing causal filter is reusable only after current-source behavior tests and provenance checks.

### Small mixture-of-experts gate

The classical mixture-of-experts construction combines specialized expert outputs through an input-dependent gate. [Jacobs, Jordan, Nowlan and Hinton, 1991](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf). Here a small softmax linear gate can emit `w_m(x)` and combine complete class distributions as `p(y|x) = sum_m w_m(x) p_m(y|x)`. This project-specific use should be evaluated separately from the original paper's task results.

Retain the current per-expert correctness gate as a baseline, but consider a gate fitted to mixture log loss as the main probabilistic alternative. Accuracy cannot distinguish well-calibrated confidence from overconfidence when the argmax is unchanged. Freeze expert predictions for the gate comparison so any difference is attributable to routing. Learn the gate on chronological out-of-sample expert predictions; calibrate resulting probabilities on a later eligible block. A portfolio controller is a further layer: better mixture log loss need not improve costed allocations.

### Hedge and delayed outcomes

Hedge maintains positive expert weights and updates them multiplicatively using observed expert losses. Its classical analysis is about loss relative to expert comparators under specified bounded-loss feedback, not guaranteed trading profit. [Freund and Schapire, 1997](https://www.sciencedirect.com/science/article/pii/S002200009791504X).

Project proposal: use bounded Brier loss, or preregistered clipped log loss, from the existing expert forecasts; update only when each target has matured. For matured feedback batch `M_t`, use weights proportional to `w_m exp(-eta sum_(s in M_t) loss_(s,m))`. Freeze the learning rate and any forgetting/share rule in the protocol. Store processed outcome IDs so a retry cannot apply an update twice. The prediction mixture and its eventual portfolio need separate evaluation. An arbitrary return-clipping rule or switching strategy is not automatically covered by the original guarantee.

Feedback delay changes online-learning analyses; it cannot be dismissed as a simple implementation detail. [Joulani, György and Szepesvári, 2013](https://proceedings.mlr.press/v28/joulani13.html). In this repository's next-close/five-session convention, labels end six observed sessions after the decision; actual availability and strict cutoff conventions must determine when the update is eligible. Do not update daily from unresolved five-session trades.

### Contextual bandits: LinUCB and Thompson sampling

LinUCB learns action values from context while choosing actions with an uncertainty bonus. Li et al. demonstrate the approach in news recommendation and evaluate using randomized logged traffic. This is evidence for a method, not transferable trading performance. [Li, Chu, Langford and Schapire, 2010](https://arxiv.org/abs/1003.0146).

Linear Thompson sampling samples from uncertainty over reward-model parameters and selects an action according to sampled values. Agrawal and Goyal analyze a linear-payoff contextual-bandit setting; linear realizability and the feedback model are substantive assumptions. [Agrawal and Goyal, 2013](https://proceedings.mlr.press/v28/agrawal13.html).

Project inference: if every expert's counterfactual prediction loss is observable after the market outcome, the model-combination task is full information. Discarding most of those observations to pretend it is a bandit needs a reason. Simulated returns for alternative allocations are available only under an explicit fill/impact model, and are not observed broker counterfactuals.

More fundamentally, buying changes holdings, future turnover and cash. A one-step bandit abstraction does not automatically account for those state transitions. A sandbox must define its decision horizon, action set, costs and state reset or carry rule; then compare against full-information Hedge and supervised action-value models. If only logged chosen actions are available, support/overlap and known or defensible propensity estimates are required. Doubly robust evaluation combines a reward model with logged-policy correction, but cannot create support for never-observed actions. [Dudík, Langford and Li, 2011](https://icml.cc/2011/papers/554_icmlpaper.pdf). Keep all exploration in isolated simulation; it does not imply permission for broker orders.

## Evaluation design to register before fitting

This section is a proposed next experiment, not a preregistration already frozen by this review.

1. Separate three endpoints: expert probability quality; accept/abstain quality and coverage; net portfolio utility after identical execution and constraints. Define a primary predictive endpoint and a primary economic endpoint before choosing models.
2. Construct expert forecasts chronologically. In each outer fold, earlier inner folds produce expert out-of-sample predictions; a subsequent eligible partition trains the controller; another block calibrates probabilities or selects thresholds. Outcome intervals crossing each boundary are purged by recorded endpoint/availability. When a final expert is refitted on more history, log that change and evaluate its effect rather than assuming training and final score distributions are identical.
3. Include cash, costed equal weight, the registered trend policy, the best eligible single expert selected using past data, fixed mixture, and no-gate variants. Equalize dates, capital, fills, fees, execution delay and exposure rules. Report both equal-exposure diagnostics and the actual constrained accounts.
4. Count distinct decision dates and matured target intervals, not just instrument-date rows. Overlapping labels and shared market shocks reduce independent information; keep same-date instruments together in folds and date-block uncertainty calculations.
5. Measure net effect, confidence interval, acceptance coverage, turnover, exposure, drawdown, expert-weight concentration, gate stability and worst supported slices. Report how much any drawdown change comes from holding less exposure. Feature importance is descriptive; compare registered removals for ablation evidence.
6. Use bounded candidate budgets and an explicit trial registry. A reasonable starting proposal is two logistic regularization choices, one shallow boosting configuration and one Hedge learning rate, plus fixed controls; fit a development runtime pilot before freezing the actual budget. Register an HMM on/off comparison separately. Do not run all model, horizon, feature and risk-control combinations.
7. Require append/change-future-data invariance; matured-feedback-only updates; date alignment; finite normalized outputs; deterministic serialization/restart; no duplicate update on receipt replay; and the same risk projection in replay, CLI and UI.

The inspected historical periods stay exploratory. No source here establishes that a more complex controller will beat the project's simple investable controls. A controller that consistently abstains can be a valid safety outcome, but must be compared with cash and cannot be advertised as predictive success.

## Useful interactive demonstration

For one recorded as-of date, reveal expert probabilities, eligible context, controller weights or rejection probability, requested exposure, the risk layer's allowed target and the final ledger action. A timeline can then reveal the later outcome and when feedback becomes eligible. Allow model and cost comparisons on an explicitly labeled historical replay; clearly distinguish replay, hypothetical perturbation and actual prospective receipt. This demonstrates the CS contribution—causal routing, constraints and reproducibility—without inventing a financial success story.

## Source-verification notes

All linked sources were searched or opened on 30 October 2026. Journal/conference pages verified bibliographic titles and authors. Rabiner and Jacobs et al. were opened as university-hosted primary PDFs; their scanned-text extraction is limited. The Freund–Schapire publisher record was returned by search with authors, 1997 volume 55(1), pages 119–139 and DOI `10.1006/jcss.1997.1504`; direct open returned HTTP 403. That access limitation does not convert its theoretical guarantee into evidence about this project. No unverified source result is used as an empirical project claim.
