# Causal Forecasts, Constrained Portfolios, and Prospective Operation

## An auditable empirical study of ML-TradeEngine

Research v2 | Historical evidence through 25 September 2026 | Prospective sections interim

## Abstract

@RESULT abstract

## 1. Motivation and research questions

A trading research system must answer two questions that are easy to confuse: whether its calculations represent the stated strategy, and whether that strategy has useful economic performance. Correct accounting is necessary for the second question, but it cannot answer it. Likewise, a profitable curve cannot validate data that were unavailable at the simulated decision time. This study follows the entire chain from public observations to predictions, constrained targets, simulated fills, persisted account state and research claims. The output is an executable local paper-trading product and a versioned empirical record. No external broker execution is part of the experiment.

The starting repository contained classifiers, a strategy zoo, an ensemble, a dashboard and saved backtests. Earlier audits identified accounting and timing defects. Subsequent repairs produced a sealed adaptive experiment and a short technical report. We preserve that experiment and examine its full prediction and ledger records instead of inferring behavior from headline accuracy or isolated screenshots. This follow-up adds class and instrument diagnostics, controlled refits, transfer tests, execution sensitivities and operational evidence. Previously inspected historical observations are explicitly exploratory. A new analysis of an old holdout does not restore its independence.

Our first question is whether causal forecasts improve probability quality and directional classification over a trailing class-frequency estimate and simple forecasting controls. The primary predictive endpoint is the paired, date-averaged log-loss difference against the prior. Accuracy, balanced accuracy, class F1 and multiclass Brier score explain different aspects of the result. A correct class prediction and a profitable trade are separate events, so no accuracy threshold is used as a proof of alpha.

The second question is whether a hybrid adds net economic value over its components and comparable simple policies when every strategy must finance its holdings through one account. The primary economic endpoint is the daily net-return difference from a costed, risk-constrained equal-weight account on matching dates. We distinguish this account from a descriptive frictionless SPY reference. The third question concerns where errors occur: classes, instruments, calendar blocks, observed volatility conditions, disagreement between experts and confident mistakes. Such conditional descriptions identify hypotheses; they do not by themselves establish a causal mechanism.

The fourth question is sensitivity to features, calibration, gate construction, training history, refit cadence and execution assumptions. Controlled comparisons change a specified factor while preserving the eligible information set and outcome definition. The fifth question concerns operation: whether the same policy can run from fresh inputs with persistent capital, restart after faults, refuse invalid state and produce durable evidence on its registered schedule. These questions define separate engineering, historical-evidence, prospective-operation and manuscript decisions. Broker readiness is a further decision outside the authorized scope.

## 2. Related work and the scope of comparison

Machine learning can model nonlinear conditional relationships in asset returns. Gu, Kelly and Xiu compare a broad range of methods for risk-premium prediction and examine the importance of interactions [gu2020]. Their result motivates careful predictive comparisons; it does not supply a performance target for this project. Their universe, prediction horizon, information set and portfolio construction differ from our small daily classification study. We do not copy their reported economic gains into a benchmark table or interpret a different task as a replication.

A simple trend rule is a demanding control for a more elaborate trading architecture because it can capture persistent price behavior without learning a high-dimensional decision boundary. Time-series momentum has been studied across futures markets [momentum2012]. Our rule combines a trailing price-average condition with intermediate-horizon momentum on the existing equity/ETF universe. It is long-only and financed through the same constrained local ledger as the ML policy. It is therefore a project control inspired by trend-following, not a reproduction of the published futures strategy.

Gradient boosting provides a flexible additive model built through successive loss reduction [friedman2001]. We pair a shallow histogram boosting classifier with regularized multinomial logistic regression to contrast nonlinear and linear decision surfaces. The logistic model is both a component and a useful simplicity control. Fixed capacity and chronological fitting make the comparison interpretable. Increasing model count or training duration after seeing poor outer-period results would change the scientific question and enlarge the search history; it would not repair a failed hypothesis.

Probability forecasts require more than class ranking. Proper scoring rules reward distributions that represent the forecaster's uncertainty [gneiting2007]. Log loss penalizes confident wrong answers strongly, whereas accuracy only considers the largest probability. The Brier score measures squared error across the complete class distribution. Temperature scaling is a low-dimensional way to alter sharpness while preserving each row's ordering [guo2017]. Evidence from other application domains does not imply that temperature scaling must improve financial forecasts. We therefore compare raw and calibrated mixtures on identical outer observations and disclose the samples used to fit the temperature.

Dependent observations complicate uncertainty. Adjacent forward-return labels overlap, and symbols respond to common market shocks. Resampling instrument-date rows independently would overstate the effective sample size. Block resampling follows the general motivation of dependent-data bootstrap methods [kunsch1989], while HAC estimators provide a complementary approach for autocorrelated averages [newey1987]. Our implementation preserves the entire cross section within a sampled date and examines several block lengths. The assumptions remain approximate when the return-generating process shifts over time.

Multiple testing is especially consequential in strategy research. Financial factor evidence illustrates how an extensive search can make conventional single-test thresholds misleading [harvey2016]. Holm adjustment controls a declared family of comparisons [holm1979], but no adjustment can recover undocumented researcher choices. The Deflated Sharpe Ratio and probability-of-backtest-overfitting literature address aspects of selection risk [bailey2014, bailey2015]. We do not report those estimators here because a complete comparable trial history and their required assumptions are not available. The current registry makes the new comparison family explicit; it does not erase earlier exposure.

Covariance shrinkage is motivated by the instability of estimated risk matrices [ledoit2004]. Our risk rule uses a fixed blend of sample covariance and its diagonal. This is a transparent regularization assumption, not an implementation of the cited paper's optimal shrinkage estimator. Exchange timing follows the official NYSE calendar [nyse2026]. The operational design also recognizes that a scheduled workflow file does not guarantee execution; workflow failures require direct evidence [github2026]. These references establish methods and constraints. All empirical performance claims below come from this project's saved or newly executed experiments.

## 3. Data, availability and evidence accounting

### 3.1 Universe and provider snapshots

@RESULT data

The universe was inherited from the project rather than sampled from historical index constituents. It mixes individual surviving companies, broad-market funds, sector funds and alternative-asset funds. Economic exposures overlap: holding a growth stock and a fund that contains it does not create two independent bets. The declared growth-group limit captures only one part of that overlap. We cannot estimate delisting bias from this universe, and we do not interpret the results as a market-wide stock-selection claim.

Raw provider OHLCV and adjusted series are separate artifacts. The term raw refers to the provider's returned fields before this pipeline's adjustment transformation; it does not certify an archival, unadjusted exchange tape. Adjusted prices incorporate the vendor's present interpretation of splits and distributions. Each snapshot records retrieval time and hashes. Historical availability uses an explicit completed-session convention because the provider does not supply an immutable historical arrival timestamp for every bar. A current-vintage reconstruction is not equivalent to a point-in-time archive.

For the local simulation, quantities are total-return proxy units. Future provider revisions can rescale historical adjusted prices, so daily marks chain the newly observed adjusted return onto stored marks. That convention avoids treating a uniform rescaling as a windfall or loss in existing holdings. It does not implement broker shares, dividend cash payments, fractional-share restrictions or corporate-action entitlements. Raw responses and available action events are retained for provenance, while broker accounting remains a separate unimplemented contract.

@TABLE data

@FIG dataset

### 3.2 Admission, missingness and labels

Every admitted price panel must have unique, monotonically ordered session keys, finite positive OHLC values, valid high/low ranges and positive observed volume. The exchange calendar, rather than a generic weekday index, defines expected sessions. There is no holiday padding, synthetic zero-volume bar, stale-price forward fill or silent fallback to the last successful forecast. An unavailable bar stops the applicable batch before account decisions. Dropping a record, skipping an event and predicting FLAT are different states and remain distinguishable in the evidence.

For information observed at the end of session t, the first modeled fill is the next completed close. The target return is therefore:

`r_target(t) = Close(t + 6) / Close(t + 1) - 1`.

A return below -1% is DOWN, above +1% is UP, and an observed return between those thresholds is FLAT. A target whose ending close has not occurred is unobserved, not FLAT. Strict fitting eligibility requires its endpoint to precede the refit decision. The distinction matters twice: at the tail of the study, where decisions outnumber mature labels, and at partition boundaries, where adjacent labels can otherwise reveal later evaluation prices.

Dataset accounting is performed separately for price rows, usable feature rows, decision rows, label-complete instrument-dates and account sessions. A portfolio needs marks even when an outcome label is not yet mature. Thus the last account sessions legitimately contribute to equity measurement while being excluded from classification. The tables and claim map preserve those denominators instead of calling every row a sample with independent statistical information.

### 3.3 Exposure log and optional information

All historical dates inspected in the prior tasks or saved release are listed in the evaluation-exposure log. The old development, selection and audit labels are retained as names for reproducible date ranges, not as a claim that the current analyst is blind to them. The new controlled experiments use development-era chronological blocks and do not replace the previously selected prospective shadow. Historical results are useful for diagnosing a design and measuring its behavior; independent confirmation must come from later observations under a frozen rule.

Macro, news and crypto are excluded from the primary information set. A working public endpoint does not establish historical vintage eligibility. Revised macro observations need release/vintage timestamps; news requires retained content, version and availability times; a bounded recent crypto endpoint does not supply the requested long archive. Legacy XGB, LGBM, LSTM, MoE and zoo outputs remain preserved. Their different targets, information sets and splits prevent treating their old headline scores as controlled architectural comparisons. The registry records scope exclusions rather than silently dropping weak models.

## 4. Methods and the shared execution contract

### 4.1 Features and forecasting models

@TABLE features

@TABLE models

The scale-free feature set combines trailing returns, volatility, prices relative to moving averages, market-relative returns, intraday range, overnight gap, volume ratio and intermediate-horizon momentum. Rolling operations use only the observed prefix. There is no full-sample normalization or learned feature screening outside the fitting partition. The logistic scaler is fitted on eligible fitting rows and then applied unchanged to calibration and evaluation. Feature order is part of the inference schema; matching names in a different order is insufficient.

The logistic expert minimizes regularized multinomial cross-entropy. The tree expert fits shallow histogram-based boosting with fixed iteration count, leaf budget and regularization. Their raw mixture is the arithmetic mean of their probability vectors. Temperature scaling applies:

`p_calibrated(c | x) = exp(log(p_raw(c | x)) / T) / sum_k exp(log(p_raw(k | x)) / T)`.

The temperature is selected only from its registered candidate set using eligible past calibration labels. Because this transformation preserves class ordering, its direct purpose is probability quality rather than creating a different argmax classification. The training-frequency control uses Laplace-smoothed class counts from the fitting sample. Always-UP is a directional control; its near-deterministic probabilities make it a poor probability benchmark when another class occurs, which is why the prior is primary for log loss.

The sealed release calibrated temperature and estimated gate benefit on the same trailing block. That is chronologically causal but permits internal optimism from evaluating a selected temperature on its fitting block. The follow-up separates fitting, temperature selection and gate evaluation with endpoint-based purges. This experiment asks whether the gate's apparent past advantage survives a distinct eligible block. It does not assume that separating the blocks must improve returns; fewer calibration or gate observations may increase estimation variance.

### 4.2 Chronology, folds and refitting

@FIG architecture

@FIG timeline

At a refit decision, eligible labels are determined by their actual session endpoints. Fitting, calibration and gate partitions are chronological. Purging removes any earlier-partition target whose outcome reaches the next partition's first decision. This rule is derived from execution timing rather than a conveniently chosen gap. A transform learned from a preceding partition is applied to later partitions without refitting. All fold records retain first/last dates, label endpoints, sample counts and source/configuration hashes.

The controlled outer dates comprise four disjoint blocks, each containing the first 63 sessions of a development half-year. Date-block resampling stays within each outer block. The bounded experiment design varies one registered factor at a time. There is no Cartesian hyperparameter search. If an inner choice is fitted, that choice is made entirely before its outer block; outer results never select a replacement pilot policy. Training-history and refit-cadence experiments align features, targets and evaluation dates, preventing a model trained on a different information set from masquerading as evidence about architecture. Additional seeds cover six representative instruments; they do not cover every instrument. Registered seeds characterize estimator variability separately from uncertainty across dates. Seeds and correlated instruments are not independent replications of the market.

The operational contract has an explicit exchange-session refit anchor and an explicit startup model. Research replay and the daily runner use the same anchor. Weekly target changes occur on the last eligible exchange session of the week, including holiday weeks, with an initial allocation decision. Other sessions process pending fills, marks and drawdown controls without routine rebalancing. These v2 rules are versioned changes; they do not retroactively alter the sealed experiment's dates or Friday-only results.

### 4.3 From probabilities to constrained targets

Expected target return is the probability-weighted mean of the fitting sample's class-conditional target returns. A positive prediction only requests an allocation when this estimate exceeds the registered round-trip cost hurdle. The trend intent requires a positive price/long-average ratio and positive intermediate-horizon momentum. A fixed hybrid combines the two intents; the adaptive hybrid allows the ML contribution only when the past-data gate accepts it. The forecasting mixture and the trading hybrid are distinct objects and are named separately in the result tables.

Before account submission, target weights pass long-only, per-instrument, gross and group caps. Trailing covariance is estimated from eligible observed returns, annualized, and shrunk halfway toward its diagonal. If estimated target volatility exceeds the registered cap, all risky target weights are scaled down proportionally. This rule does not guarantee future volatility or drawdown, and estimation error remains visible in sensitivity tests. Idle cash earns zero under the simulated account contract.

Orders represent changes from current holdings toward target holdings, not repeated purchases for repeated UP signals. Pending buys reserve cash; pending sales are not spendable buying power. Orders can be rejected when a price gap exceeds the reserved budget or projected exposure violates limits. A drawdown halt requests reductions and persists through restart. Marked market drift can temporarily breach a target cap between decisions, so the system records a breach and blocks risk-increasing orders rather than pretending a target limit is a pathwise guarantee.

The balance identity is:

`Equity_t = Cash_t + sum_s Quantity_(s,t) * Mark_(s,t)`.

Net realized trade profit includes both entry and exit costs. The simulator applies proportional commission and slippage on each fill. Partial exits allocate the corresponding cost basis; terminal holdings remain marked rather than receiving an invented liquidation. This is one account per policy with fixed initial capital, shared constraints and explicit rejection behavior. Summing independent ticker return curves would describe a different, potentially unfundable strategy.

### 4.4 Persisted decisions and operational stages

The collector validates release, dependencies and state before inference. A decision plan records session, information cutoff, data manifest, model versions, probabilities, targets and marks before account processing. The plan is immutable for that session. Account commits and final receipts are recoverable stages: a restart resumes the same plan and uses event IDs to avoid duplicate fills. Concurrent CLI, dashboard and scheduler requests share the same lock and state owner.

Release/model storage is separate from mutable account state. Backups retain original receipts and database state, and restoration is checked at the recorded original path after moving the original state out of the way. Relocating embedded absolute artifact paths is outside the supported recovery contract. A stale writer, truncated file, missing model or changed source must cause a visible halt; a success-shaped fallback would contaminate both the account and the research dataset. Changes to the collector after qualification require a new release and evidence. The old source and sealed releases are preserved as rollback and historical reproduction assets.

## 5. Experimental design, inference and computational budget

@RESULT experiments_design

@TABLE experiment_design

The primary predictive comparison averages each day's candidate-minus-prior log-loss difference across symbols, then resamples contiguous date blocks. Keeping the cross section intact avoids treating a market-wide error as many independent surprises. Circular wrapping allows every date to start a block, but short samples provide limited distinct market episodes. Block-length sensitivity examines a label-horizon-scale block, the registered primary block and a longer block. Confidence intervals describe the behavior of this historical sample under the resampling assumptions; they do not make the consumed audit confirmatory.

Economic comparisons use identically indexed daily net returns. Rebased charts make date-range returns visually comparable but do not reset account holdings. Where a fresh-capital replay is reported, it explicitly begins a new account and is separately labeled. CAGR compounds the observed return over its annualized duration. The new portfolio diagnostics use a zero risk-free rate for Sharpe and a zero minimum acceptable return for Sortino, consistent with zero-yield cash. The sealed release separately reports Sharpe relative to a fixed annual hurdle; those differently defined values are never interchanged. Sortino uses downside deviation relative to its disclosed target. Undefined zero-variance statistics remain undefined rather than being converted into evidence for a strategy.

Hypothesis families come from registered contrasts. Holm-adjusted values apply separately to the specified predictive and economic families. Additional class, symbol and regime slices are descriptive unless entered into a distinct family before computation. This avoids presenting a favorable post hoc slice as a new independent experiment. Effect sizes and intervals take precedence over binary significance language. Approximate minimum-detectable-effect calculations use the estimated sampling variability; they are design aids, not guarantees of future power.

Execution robustness perturbs clearly named assumptions. Cost changes can retain decisions while changing financing and fills. Delaying execution or restricting fill size may alter exposure and subsequent account behavior. Missing-bar tests are admission/recovery tests and must fail before decisions rather than interpolate a fictional observation. Risk-parameter changes create a different target policy and are classified accordingly. The table records which components remain frozen in each experiment so that a changed return is not assigned to the wrong cause.

Runtime evidence is separate from correctness evidence. Measurements disclose hardware, interpreter, thread limits, sample counts and cold/warm conditions. Quantiles from small samples are descriptive order statistics, not reliable tail service-level estimates. Compute budgets are frozen after a development-only pilot, attempted trials retain status and errors, and cache keys include data, source, configuration and fold boundaries. Exact repetition from saved source is a numerical reproduction claim; operation across a genuinely new market period is a different claim.

## 6. Prediction quality and error structure

@RESULT forecasting

@TABLE forecast_summary

@FIG confusion

@TABLE class_report

Per-class reporting prevents a common majority class from concealing weak recall elsewhere. Precision asks what fraction of a model's class assignments were right; recall asks how many observed examples of that class it recovered. F1 combines these two rates, and support states the denominator. Row-normalized confusion matrices emphasize which true classes are confused, while raw counts identify where most mistakes occur. The dominant transition can change when class prevalence changes even if the conditional error rates remain stable.

@RESULT error_structure

@FIG calibration

@TABLE calibration

Confidence-coverage curves retain only predictions above fixed confidence cutoffs and report both accuracy and remaining observations. A higher accuracy at low coverage can be practically irrelevant when almost no decisions remain. These curves are diagnostics, not an opportunity to choose the best threshold after viewing outcomes. Reliability-bin uncertainty is also limited by correlated dates; bins with little support cannot justify a confident calibration claim. For trading, an apparently well-calibrated probability still needs useful class-return estimates and enough expected advantage to pay costs.

@FIG forecast_inference

@RESULT uncertainty

## 7. Controlled ablations, adaptation and transfer

@RESULT ablations

@TABLE ablations

@FIG ablations

A feature-importance ranking is not a feature ablation. The experiments remove a registered feature family, refit the eligible model and compare outer predictions on matching dates. This evaluates the value of allowing that family under the stated model and budget. Correlated families can substitute for one another; removing one does not estimate an isolated economic causal effect. The sign of an ablation also depends on regularization and sampling variance. We retain unfavorable and near-zero changes alongside any improvement.

@RESULT calibration_gate

@TABLE components

@RESULT learning

@TABLE learning

@FIG learning

@RESULT seeds

Learning curves report fitting and outer-period loss together. A widening gap can be consistent with variance or time shift, while high loss on both partitions can indicate limited representation or model capacity. A smaller history can adapt faster but discard rare conditions; a larger history can stabilize estimation while retaining outdated relationships. Refit-cadence tests ask whether refreshing a fixed model matters under aligned information. No cadence is promoted merely because it is best on these consumed development blocks.

@RESULT transfer

@TABLE transfer

@FIG generalization

The instrument-transfer experiment withholds target instruments from pooled fitting and respects the temporal outer boundary. That is stronger than reporting each single-instrument model on its own later observations, but it is still limited by overlapping economic exposures and a shared market feature. Excluding an instrument's labels does not remove all information about the market in which it trades. Transfer results therefore describe the declared source/target grouping, not universal generalization to unseen exchanges or delisted securities.

Regime slices use conditions observable at the decision time. Realized future volatility is not used to assign a model's historical input state. Support counts accompany every slice, because a dramatic result from few dates is less stable than an aggregate over many market episodes. Instrument, period and regime analyses describe heterogeneity; they do not license a selector trained retrospectively to keep only the favorable cells.

## 8. Economic performance, costs and failure mechanisms

@RESULT controlled_economics

@TABLE controlled_economics

@RESULT economic

@TABLE portfolio

@FIG portfolio

@FIG costs

Prediction loss and portfolio return need not rank models identically. The trading rule uses estimated payoff magnitude, abstention, weekly decisions and constrained capital. A prediction can be directionally wrong while its account is in cash; a correct signal can still lose after a delayed fill or costs. The decomposition therefore examines exposure, turnover, fill rejections, realized net outcomes and open holdings rather than attributing every equity change to classifier accuracy.

@RESULT attribution

@TABLE episodes

@TABLE failures

The reported failure cases follow registered selection rules: representative observations, the largest confident forecast errors and the largest adverse account contributions. They are not handpicked success stories. A loss concentration can reveal concentrated exposure or a difficult period, but explaining why a model produced a probability requires tracing its actual inputs and fitted state. Descriptive associations between a gate and losses are not causal gate benefits. Component and gate interventions provide the relevant controlled evidence.

@RESULT sensitivity

@TABLE stress

@FIG stress

The simulator's costs are assumptions rather than measured execution. Fixed proportional slippage cannot represent all auction impact, spreads, illiquidity or market halts. Bounded fill scenarios explore one aspect of capacity without certifying real fill quality. A small simulated account may reduce some capacity concerns, but that does not make every close executable at the modeled price. Broker deployment would require timing, corporate-action and reconciliation evidence that this study deliberately does not claim.

## 9. Operational evaluation and prospective collection

@RESULT operations

@RESULT runtime

@TABLE latency

@TABLE operations

@FIG operations

The local scheduler uses exchange-session events rather than a fixed Hong Kong clock. Standard close+30 occurs at different local times across US daylight-saving changes, and early-close sessions advance the event. Only an eligible transient failure permits the registered retries; deterministic integrity or risk failures halt. The watchdog checks for an absent receipt as well as an explicitly failed one. Pre-open checks do not create replacement strategy decisions. Weekly archival and milestone reviews assess coverage, latency, risk and protocol deviations without choosing winners from pilot returns.

The collector cannot execute while its host is asleep or unavailable. On recovery it must distinguish an overdue event from a legitimate current event, record missed coverage and refuse backfilled prospective decisions. A real launchd trigger receipt demonstrates that the operating system invoked the current runner and it reached its persistent state. It does not demonstrate a future session-close invocation, sustained host availability or an always-on service. Those are observations that must accumulate.

@TABLE schedule

@RESULT prospective

@TABLE prospective

Mature outcomes attach to earlier immutable decisions only after the execution-aligned endpoint is observed. After the final decision, six terminal sessions process outstanding fills and marks without generating new allocation targets. A halted account can still retain valid matured labels, but missing terminal accounting prevents a complete economic freeze. The read-only final-report command reconstructs the costed equal-weight comparator from those same observed snapshots and releases inferential results only at the registered final freeze; interim views are descriptive. The last decision date and the last label-maturity date are different. A missing expected session remains a recorded outage and does not shift the registered end until results become favorable. The operational observation period is not a promise of adequate power for a small economic effect. A strategy can complete the operational pilot and still fail the statistical or economic gate.

## 10. Reproducibility and claim integrity

@RESULT reproducibility

The evidence package contains the protocol, exposure log, registry, per-trial configurations, saved predictions, fold boundaries, portfolio records, source tables and plot data. Original repaired experiments and sealed releases remain available. New results occupy versioned directories. The manuscript builder reads a single compiled statistics source and checks its inputs' hashes; generated empirical sentences and table cells carry source selectors in the claim map. A changed metric must update the dependent prose or fail validation. A missing figure source prevents a complete build.

Tests are recorded with command, runtime, collected test IDs, pass/fail/skip status and source hashes before and after execution. A passing JUnit file by itself cannot qualify unrelated source or an empty suite. Critical behavior includes mathematical accounting examples, future-data mutation, model serialization, release tampering, full-policy replay, concurrent execution, fault recovery and the dashboard-to-ledger path. Provider integration runs are explicitly separated from the default offline suite. Successful optional-provider checks do not admit those feeds into the primary model without historical eligibility evidence.

Clean-environment reproduction tests use the pinned core runtime and preserved artifacts. Byte-identical artifact checks, exact numeric comparisons and tolerance-based comparisons are separately named. This matters because serialized files may embed metadata even when numerical outputs agree. The package states what was actually rerun; it does not transform a saved verification claim into a new training run. Source inspection and local tests likewise do not certify a remote Actions deployment.

## 11. Discussion and limitations

@RESULT discussion

The most useful scientific distinction is between a weak forecasting result and an invalid experiment. Weak predictive or economic results remain publishable evidence when their design, uncertainty and limits are explicit. Conversely, stronger reported performance would not compensate for unavailable labels, inconsistent capital accounting or a search procedure hidden from the reader. This investigation makes those dependencies visible and creates a path for prospective replication without requiring the model to win.

Several limitations remain structural. The inherited universe has selection and survivorship bias. Current-vintage vendor adjustments do not reconstruct exactly what a historical participant saw. Overlapping ETFs and common market returns limit the independence of the cross section. Execution-aligned labels overlap in time. The short recent audit and finite development folds provide limited evidence across distinct regimes. Bootstrap intervals rely on local dependence assumptions and do not establish robustness to every structural break.

The controlled training budget is intentionally bounded. It evaluates a family of fixed, interpretable models rather than exhaustively searching all architectures. Excluding larger legacy models from a controlled table avoids an invalid comparison; it does not prove those models cannot work. A later architecture study would need its own registered information set, chronological partitions, compute budget and independent confirmation. Adding news or macro data would require actual availability evidence before any claim of incremental benefit.

Operational qualification also has a narrow scope. A local scheduler, persistent account and successful recovery test make collection feasible, but the user's host remains an availability dependency. Synthetic fault injection does not reproduce every filesystem or power-loss condition. The total-return proxy ledger omits real broker entitlements and market microstructure. No external orders are enabled, and no result here should be read as authorization for them. The deployment account remains cash unless independent evidence and separate execution gates eventually support a change.

## 12. Conclusions

@RESULT conclusion

## References

@REFERENCES

## Appendix A. Complete instrument diagnostics

@TABLE instruments

These rows use the same mature-label definition as the main analysis. Instrument rankings are descriptive and do not define a new selection policy. Machine-readable tables retain full precision, all registered variants and additional class/period slices omitted from the main text for readability.

## Appendix B. Fold, trial and evidence inventory

@TABLE evidence

@TABLE registry

@RESULT continuation

## Appendix C. Reproduction commands and interpretation

@COMMANDS

Every command writes new evidence or reads a sealed artifact. Do not overwrite the existing release to conceal a changed method. Historical diagnostics can be regenerated immediately from the saved data. Prospective receipts and outcomes require the actual passage of exchange sessions, host availability and completed label endpoints. The historical manuscript can be substantively complete while the full prospective study remains interim.
