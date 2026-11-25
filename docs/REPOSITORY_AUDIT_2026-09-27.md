# ML Trade Engine — repository audit, 27 September 2026

**Verdict: substantial research infrastructure exists, but profitable autonomous trading has not been demonstrated. The project is at experimental Phase 08D implementation, before Phase 09 sandbox validation. Several earlier validation gates need to be reopened.**

The README describes an older five-ticker milestone. The later [blueprint](/Users/a/Downloads/ml-trade-engine/docs/ml-trade-engine-mvp.md:282), source, local artifacts, and July additions tell a different story. The repository contains 22 trained ticker models, eight strategy implementations, 176 strategy/ticker runs, ranking, environment analysis, a regime allocator, broker integration, and a Streamlit dashboard. “Implemented” does not mean “validated.”

## Scope and evidence

Reviewed the repository structure, phase logs 01–07, updated blueprint, paper draft, July MoE/policy/sentiment reports, presentation text, feature/training/backtest/risk/research/live paths, tests, local Parquet/JSON/SQLite artifacts, GitHub Actions history, and provider connectivity. Local HEAD was `f194283`; pre-existing uncommitted work was included and preserved. Much of Phase 08 exists only in that local work.

Evidence files:

- [Reproducible offline audit](/Users/a/Downloads/ml-trade-engine/scripts/repository_audit.py): numerical checks, minimal defect reproductions, and accounting sensitivity.
- [Full metrics and artifact inventory](/Users/a/Downloads/ml-trade-engine/docs/audit-2026-09-27.json): all 22 ML tickers and 176 zoo runs.
- [Runtime evidence](/Users/a/Downloads/ml-trade-engine/docs/audit-runtime-2026-09-27.json): read-only provider/account checks, CI failure, test results, source hashes, and limitations. No credentials are included.

The complete existing test suite passed in two groups: **260 offline tests + 23 data integration tests = 283 passed**. The default shell `python3` is Python 3.9.6; the working environment was `/Users/a/miniconda3/bin/python` 3.13.11. `alpaca_trade_api` is absent from that environment, although direct authenticated HTTP reads of the paper account worked. Tests use a mocked broker and therefore do not prove live order compatibility.

No models were retrained, no production data caches refreshed, no orders submitted, no alerts sent, and no workflow re-enabled. Test-specific caches were exercised. The dashboard UI was not launched. This is a forensic baseline, not a new clean backtest or a completed repair.

## Where development stopped

- **01–05:** data, features, three-class ML, simulation, and risk code exist. Saved models and historical results exist. Accounting, causality, calendar, and model/inference defects invalidate the stronger “look-ahead-free” and “production-grade” descriptions.
- **06A/06B:** all eight strategies and the ranking system exist and have been run on 22 equities/ETFs. Crypto is a feature source, not part of the evaluated 22-asset trading universe.
- **06C:** regime/cost/decay/factor artifacts exist. Their interpretation inherits upstream defects; factor output does not record whether actual FF3 or ETF proxies were used.
- **06D:** meta-allocation code and draft v0.2 exist. The paper still has placeholders and methodological claims that contradict the code.
- **06E/07:** broker integration and seven-page dashboard exist. There is no demonstrated daily paper-trading operation or assessed dashboard deployment.
- **08A–08C:** MoE gating, dynamic policy, and FinBERT sentiment experiments exist. MoE and sentiment are not in the production ensemble. Policy is opt-in and only partially matches its backtest behavior.
- **08D:** the meta-strategy is reachable through `--strategy-mode meta`. The scheduled workflow still defaults to the ensemble. The meta live path does not yet faithfully implement the research portfolio.
- **09:** the required broker-fill/backtest reconciliation and sustained sandbox track record have not been established.

The newer presentation proposes generated strategy plugins, broader expert orchestration, and constrained portfolio construction. Those are proposals beyond the current implementation.

## What the performance numbers actually show

### Classifiers and ML trading

The saved classifier predictions average **41.53% accuracy** across 22 tickers. An always-UP classifier scores **48.73%** on those same samples; it beats all 22 models on accuracy. The 33.3% random-class baseline in earlier notes is too weak for this imbalanced dataset. Accuracy alone does not determine trading profitability, but the claimed predictive improvement is not established against this simple baseline.

The saved per-ticker ML results average **−0.070 Sharpe**, **2.86% CAGR**, and **56.28% closed-trade win rate**. These are arithmetic means over unequal periods, not portfolio performance: five core tickers have 995 rows beginning July 2022; the other 17 have 2,085 rows beginning May 2018. No ticker has saved Sharpe above 1.0.

For concrete examples, AAPL's current saved result is **Sharpe 0.052, CAGR 4.50%, maximum drawdown −30.65%, win rate 57.38%**. SPY is **−0.556, −0.66%, −18.62%, 52.27%**, respectively. The README's older values should not be used as the current scorecard.

An independently calculated, hypothetical daily equal-weight portfolio of the 22 saved ML return streams, restricted to their common dates **4 July 2022–24 April 2026**, has **Sharpe 0.200, CAGR 6.30%, drawdown −8.68%**. Thus the build log's “portfolio Sharpe −0.07” conflates mean ticker Sharpe with portfolio Sharpe. This reconstructed portfolio is not an executed account and has no extra cross-asset rebalancing costs.

The saved risk-managed ticker results average **Sharpe −0.716, CAGR −0.59%, win rate 45.43%, drawdown −24.32%**. Their hypothetical equal-weight portfolio on the common period has **Sharpe −1.747, CAGR 1.14%, drawdown −2.19%**. Negative Sharpe can coexist with positive dollars here: metrics subtract a fixed 5% annual risk-free rate while idle cash earns zero in the simulations.

### Strategy zoo and meta-allocation

The saved meta portfolio covers **19 May 2021–15 May 2026**, 1,303 business-day rows. Its summary reconciles to the saved returns: **Sharpe 0.712, CAGR 9.49%, drawdown −4.54%, Calmar 2.089, raw-return t-statistic 3.423**. A hypothetical $10,000 becomes $15,978 in that saved curve. These are research results with unresolved bias, not demonstrated achievable returns.

The strongest saved standalone equal-weight strategy portfolio is Momentum 12-1 at roughly **Sharpe 0.555, CAGR 11.1%, drawdown −18.9%**. Alpha Trends is about **0.378 / 7.56% / −8.69%** on the meta report's dates. Mean Reversion has low drawdown but only about **0.096 Sharpe**. The other portfolios do not establish a stronger risk-adjusted edge. Ranking individual ticker/strategy winners is not equivalent to validating a diversified portfolio selected in advance.

The paper's saved SPY benchmark is defective: **308 of its 1,303 dates lack SPY observations and are filled with zero returns**. A separately fetched Yahoo adjusted-price snapshot, covering the exact meta period and preserving the legacy business-day calendar, yields approximately **SPY Sharpe 0.550, CAGR 13.52%, drawdown −24.50%**, instead of the report's **0.653 / 14.02% / −18.76%**. The fresh snapshot can also reflect vendor revisions. Rebuild the benchmark from a frozen matching data vintage before making alpha or outperformance claims.

### What the accounting correction changes—and does not establish

An isolated sensitivity calculation retained the saved long-only trade dates, fill prices, fees, and allocations while correcting only omitted exit-day NAV changes. It was checked against an independent cash/share ledger on 100 randomized bars.

- ML equal-weight Sharpe moves **0.200 → 0.237** and CAGR **6.30% → 6.72%**.
- Meta Sharpe moves **0.712 → 0.751** and CAGR **9.49% → 9.88%**, holding the original allocations and switching costs fixed.
- Including entry costs in closed-trade outcomes lowers average ticker win rate **56.28% → 52.73%**; AAPL falls **57.38% → 55.74%**.

The simulator error does not uniformly inflate returns. These sensitivities are **not corrected performance claims**: the signals and historical allocations still contain leakage, and allocations were not reselected on repaired histories. Leakage-free results remain unknown.

### Later experiments

MoE reports small changes on a three-ticker pilot, with SPY still negative and AAPL worse; its rolling-accuracy inputs leak unresolved future labels. Dynamic policy reports smaller average drawdown on 20/22 tickers but worse average Sharpe (approximately −0.72 → −0.95). Sentiment has in-sample SHAP feature-importance evidence, not an out-of-sample incremental P&L result. None currently supports promotion to autonomous trading.

## Confirmed defects and their consequences

### P1 — cash accounting and trade outcomes disagree

[Simulator exit handling](/Users/a/Downloads/ml-trade-engine/src/backtest/simulator.py:109) calculates an exit's proceeds but sets cash from yesterday's equity at line 121. An entry at $100 followed by an exit at $110 leaves $10,000 NAV unchanged with zero costs, although the trade log reports $1,000 profit. Both ML Phase 04 and the zoo use this simulator.

The short recurrence at line 151 is also wrong: prices $100 → $90 → $90 produce $10,000 → $11,000 → $13,000 while the unchanged final price should leave $11,000. The saved core results are long-only, so this short defect is a separate blocker for long/short research.

The exit trade P&L excludes entry fees. At 15 bps per side, a $100 → $100.20 round trip is reported as **+$4.96** on $10,000 although its net outcome is **−$10.04**. Win rate, payoff ratio, expectancy, and downstream Kelly calibration therefore need rebuilding.

### P1 — regime features are noncausal and incorrectly aligned

[decode_hmm](/Users/a/Downloads/ml-trade-engine/src/features/regime.py:94) indexes the valid subset against itself, then writes those positions into the full array. Warm-up missing rows are not respected. A deterministic example returns `[0,1,2,-1]` instead of `[-1,0,1,2]`. Every current ticker feature file has ten final `hmm_regime=-1` rows. This shifts later observations' labels into earlier rows; 15/22 saved feature selections include `hmm_regime`.

Separately, [feature computation](/Users/a/Downloads/ml-trade-engine/src/features/pipeline.py:190) fits and decodes the HMM over the entire feature history before training splits. Alpha Trends fits on its entire input too. Full-series parameter estimation and Viterbi decoding are not permissible causal test features. Calling this leakage “minor” has no measured justification. Correct indexing alone will not remove it; use past-only estimation and filtered state probabilities available at each decision time.

### P1 — supervised selection and calibration see future outcomes

[ModelTrainer.prepare_data](/Users/a/Downloads/ml-trade-engine/src/models/trainer.py:172) passes the full feature/label dataset to SHAP selection before walk-forward splitting. Despite the comment about a held-out slice, the helper fits and explains all supplied rows. Class weights are also derived from the full label vector. Move supervised transforms inside each training fold.

The printed fold ensemble uses F1 measured on that same test fold to set weights. Persisted OOS probabilities instead use fixed weights, so this particular issue contaminates printed fold metrics rather than being the weighting method behind saved trading curves. The distinction matters.

[RiskEngine](/Users/a/Downloads/ml-trade-engine/src/risk/engine.py:211) calibrates its payoff ratio from the full Phase 04 backtest, including the future of early risk simulation dates. Moreover, `p_up` predicts a thresholded forward return, not the probability that a stop-managed trade ends profitable. That mapping needs trade-outcome calibration on past data.

[MoE rolling accuracy](/Users/a/Downloads/ml-trade-engine/src/models/gating.py:129) shifts correctness by only one bar even when the label matures five bars later. At decision t=20, changing the still-unavailable label from t=19 changes gate input from 1.00 to 0.95. An embargo on fitting the gate does not repair leaked evaluation inputs. Warm-up volatility fills in gating/policy also use full-series medians.

### P1 — macro data lack publication-time/vintage alignment

[Macro joins](/Users/a/Downloads/ml-trade-engine/src/features/cross_asset.py:109) backfill initial missing history and join observations by observation date without publication lag. CPI is especially problematic: the month being measured is not the date its value became available, and revised historical values are not necessarily the contemporaneous observations. ALFRED/FRED explicitly distinguish what is known today from what was known at a historical date. [FRED real-time-period documentation](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html).

### P1 — live execution has different position semantics

[LiveEngine's FLAT branch](/Users/a/Downloads/ml-trade-engine/src/live/engine.py:216) skips processing; backtests interpret FLAT as exiting to cash. All default zoo strategies are long-only, and the meta generator emits UP or FLAT, so the live meta path does not close positions when its strategy turns flat. The position may instead remain until a bracket leg closes it.

Repeated UP signals can add another position each new day: [execution](/Users/a/Downloads/ml-trade-engine/src/live/engine.py:311) buys a fresh allocation instead of calculating the difference between target and held shares. The heat check considers filled positions before the order, not projected exposure plus pending orders. Multiple queued orders can bypass the intended total exposure limit. Exceptions in risk/idempotency checks are logged and processing continues.

Backtests fill at next close, whereas the live path queues market orders after close and adds a 3×ATR profit target absent from the research simulations. The risk simulator uses 95% deployment plus a Kelly filter; live ensemble sizing uses Kelly fractions capped per position. Policy size caps are not passed into live sizing; meta mode applies only policy ATR distance. A shared target-position policy, account ledger, pending-order reservations, and common execution convention are required before any reconciliation can be meaningful.

### P1 — live inputs and model artifacts are inconsistent

[Latest inference](/Users/a/Downloads/ml-trade-engine/src/live/signal_generator.py:135) silently drops absent model columns. AAPL's current feature cache lacks `btc_return_21d`; SPY lacks both BTC columns. Direct cached inference reproduces **24 versus 25** and **23 versus 25** scaler feature errors, respectively.

The LSTM receives one row at line 145. Its predictor returns uniform probabilities when given fewer than the 60-row sequence length, so the live “three-model ensemble” does not obtain a real LSTM prediction. Passing only each test fold to the LSTM similarly causes a neutral warm-up at each fold boundary.

Final models are the last fold's model objects; the comment claiming a refit on all available data is not implemented. All 22 original `training_results.json` files lack horizon/threshold metadata even though newer code now saves it. Scalers and LSTM weights exist locally but are excluded from Git; the workflow relies on cache restoration and has no cold-start model-bundle retrieval or validation step.

Missing supporting feeds can be swallowed at debug level; batch inference skips failed tickers without populating `RunResult.errors`. A run can finish without expected signals and still look successful. The date attached to a signal can also be newer than the last fully valid inference row. Require schema, model version, as-of time, and expected-ticker checks before signal promotion.

### P1 — benchmark coverage and artifact provenance are inadequate

[Meta benchmark construction](/Users/a/Downloads/ml-trade-engine/src/research/meta_strategy.py:288) fills missing benchmark history with zero. Comparative alpha, beta, and information ratio must be recomputed on actual matching observations.

Raw, feature, training, zoo, and research artifacts cover different date windows and are overwritten in place. There are three stale Alpha Trends trade logs (BAC, XLK, TLT) attached to current completely flat equity curves. [ZooRunner persistence](/Users/a/Downloads/ml-trade-engine/src/research/zoo_runner.py:169) only writes a trade log when trades exist, leaving an older log untouched on zero-trade reruns. Use immutable run IDs and a manifest tying code, configuration, data, signals, fills, and metrics together.

### P2 — calendars, statistics, and strategy labels need correction

[Gap filling](/Users/a/Downloads/ml-trade-engine/src/data/loader.py:79) creates every weekday as an equity trading session. Current raw equity files contain 49–50 zero-volume rows each; the ML window even begins on 4 July 2022. This changes horizons, indicators, turnover, and annualization. Use the exchange session calendar and distinguish missing bars from market holidays. Hourly equity filling currently also creates overnight/weekend hours.

The Sortino denominator is the standard deviation of negative returns rather than downside deviation relative to the target. Raw-return regression is labeled Jensen's alpha without subtracting the risk-free series. Ranking uses `abs(t_stat)`, so evidence of a reliably negative return can improve that component. IID t-tests and unadjusted selection among 176 runs do not establish discovery significance. The meta's raw t=3.42 is not evidence of excess performance over cash; under the same IID convention its excess-return t would be only about 1.62. Add serial-dependence-aware inference, multiple-testing controls, and a truly untouched evaluation period.

Momentum is time-series momentum, not top-quintile cross-sectional selection. Pairs/StatArb trades one asset against a benchmark signal; it has no hedged second leg or cointegration selection. Carry is a yield-curve/trend proxy, with a silent trend-only fallback when macro is unavailable. Those are valid research candidates if named and described accurately.

Sentiment aligns to a fixed 21:00 UTC close, which misattributes after-close headlines during US daylight saving time and early closes. Its cache has no normal refresh TTL. It has not yet affected the original trained ensemble, but must be corrected before evaluating its incremental value.

## Data ingress and operational status

GitHub lists **67 workflow runs, all failed**, and the workflow is **`disabled_inactivity`**. The latest [failed run, 1 September 2026](https://github.com/Yash789k/ML-TradeEngine/actions/runs/33571092654) reaches `Alerts()` and crashes on `int("")` because the optional `SMTP_PORT` secret is empty. This happens before fetching market data. The same failure reproduces locally with an empty environment value. Older failed runs were counted but not all individually diagnosed.

Direct checks today returned HTTP 200 for Yahoo, Kraken, FRED, Alpaca News, and the Alpaca paper account. The source-level data integration tests also passed. Therefore an ongoing outage of all these providers is not supported by today's evidence. The system's stale state comes with confirmed scheduler/startup and feature-schema failures.

Local raw caches end on **18 May** for 19 equities and **10 July** for AAPL/NVDA/SPY. BTC ends **3 July**, ETH **16 May**, macro's index **13 July**. A macro frame's final index is not proof each series has a fresh release because the loader forward-fills observations.

The default request for 1,825 BTC daily bars currently returns only **720**. Kraken documents that older OHLC history cannot be paged through that endpoint, and that its last bar is unfinished. Obtain a separate historical archive and exclude unfinished bars. [Kraken OHLC documentation](https://docs.kraken.com/api-reference/market-data/get-ohlc-data).

The configured paper account returned **$100,000 equity and cash**, no positions, and no orders in the queried order list. Local SQLite has **13 signals, 0 orders, 0 equity snapshots**, last signal logged 13 July. This does not establish a historical trading profit. It is a point-in-time read, not a certified lifetime account audit.

Supabase mirroring code exists, but the local environment only configures Alpaca keys/base URL. No Supabase connection or remote contents were verified. The sync omits circuit-breaker state, and its failures return success, so it is not a validated recovery store.

## Recommended continuation and acceptance gates

### 1. Establish an immutable baseline and repair correctness

Preserve the local July work and capture complete model bundles, source hashes, and configuration. Repair the simulator ledger, both-side trade costs, HMM row alignment, causal regime estimation, fold-local feature selection/calibration, macro availability, MoE label maturity, and exchange calendars. Add tests that fail on the concrete cases in this audit. The acceptance gate is agreement between independent fill/ledger/NAV calculations, plus future-data perturbations that cannot change earlier inputs or decisions.

### 2. Rebuild the experiments before tuning more strategies

Freeze one eligible universe and common history, regenerate features/predictions, and run nested/purged walk-forward evaluation with horizon-aware gaps. Hold out a final date range from every tuning and allocation decision. Compare always-UP/majority prediction, cash, SPY, equal-weight assets, simple momentum, the ML ensemble, and meta-allocation using the same dates and costs. Evaluate cash yield, realistic slippage, turnover, and capacity explicitly. Keep all tried configurations in the experiment registry.

Select candidates by out-of-sample net excess return and risk, with confidence intervals, minimum independent trade evidence, rolling-window stability, and multiple-testing-adjusted evidence. Only then apply a normalized composite ranking; negative significant returns should fail eligibility rather than receive an absolute-t reward. Sentiment/MoE/policy each need an ablation on the same frozen folds. The existing MVP targets (Sharpe >1, CAGR >15%, drawdown <20%, win rate >55%) are hypotheses/gates, not properties obtained automatically by adding tickers.

### 3. Implement one capital-constrained portfolio/execution path

Convert strategy output into target weights and reconcile those targets to current positions and pending orders. Include shared cash, gross/net exposure, position and sector concentration, covariance, turnover, and liquidity constraints. Make FLAT exit consistently. Model the chosen entry convention, bracket/stop/target behavior, fees, and fractional-order rules identically in simulation and paper execution. Persist fills and portfolio state, not just signal intent.

### 4. Begin Phase 09 with simulated capital after those gates pass

Fix the optional email-port parsing, validate full model bundles on a clean runner, fail visibly on missing/stale inputs, and make runs idempotent by market session and order intent. Then restore scheduling deliberately in paper mode. Use a fixed experiment capital allocation, one frozen candidate policy, daily NAV/position/fill reconciliation, slippage measurements, and predetermined stop rules. Record a sustained prospective sample; the existing roadmap proposes 3–6 months, but elapsed time alone cannot prove edge. Paper execution itself omits some live-market frictions, as [Alpaca documents](https://docs.alpaca.markets/us/docs/paper-trading).

### 5. Build the paper from the validated experiment registry

Revise the thesis into a testable question about whether regime selection adds value after costs and selection bias. Remove the current claims of causal HMM features, top-quintile momentum, and validated meta profitability until supported. Report negative findings and ablations as well as successful candidates. Recreate figures/tables from versioned run manifests; name factor proxies distinctly; use matched benchmarks and robust statistical inference. Paper preparation can proceed alongside sandbox engineering, but results should come from the corrected protocol.

## Reproduce this audit

```bash
/Users/a/miniconda3/bin/python scripts/repository_audit.py --output docs/audit-2026-09-27.json
/Users/a/miniconda3/bin/python -m pytest tests/ --ignore=tests/test_data_pipeline.py -q
/Users/a/miniconda3/bin/python -m pytest tests/test_data_pipeline.py -q
```

The audit script reads existing artifacts and writes only the requested JSON. Its accounting-only output deliberately retains the original decisions and cannot replace the clean research rerun above. Runtime provider/CI observations are dated snapshots in the separate runtime evidence file.
