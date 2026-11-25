# Controlled historical development experiments

This package addresses R3–R7 with fresh, registered refits. It does not upgrade previously inspected 2024–2025 history to an untouched final holdout, and it does not use the 2026 selection/audit windows or prospective outcomes. The fixed-universe reference dataset remains current-vintage adjusted OHLCV; prices and quantities are return-proxy units.

## Registration and execution

- Parent protocol: `data/research-v2/protocol.json`.
- Runtime pilot: `data/research-v2/experiments/pilot.json` (two first-fold development symbols, four estimator fits; no outer performance used for design).
- Frozen design and full 848-trial list: `data/research-v2/experiments/design.json`.
- The runtime registration records 1,696 estimator fits, plus four pilot fits, under the 1,800-fit parent budget. Each trial fits one logistic classifier and one histogram gradient boosted tree.
- Append-only attempts and outcomes: `data/research-v2/experiments/trials.jsonl`. One registration attempt failed because XOM was absent from the already fixed universe; XLE was substituted before registration or outcome evaluation. That failed attempt fitted no models.
- Immutable completed checkpoints include probabilities, fitted estimators, model manifests, training hashes, exact fit/calibration/gate boundaries, timing, sizes, and input/source/config cache keys. Incomplete checkpoints cause a visible halt instead of being silently overwritten.

Run from the repository using the recorded Python environment:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/a/miniconda3/bin/python -m research_v2.experiments run
```

The run reuses only completed checkpoints whose exact design, source and data provenance agrees. It refuses changed source. Do not delete or rewrite a registration to get past that guard. Amendments require a new versioned design.

## Questions and contrasts

The prediction study samples the first 63 exchange sessions in each half-year of 2024 and 2025, four outer blocks with 252 dates. All 22 fixed symbols participate in component, feature and history/fraction experiments. AAPL, JPM, GLD, XLE, TLT and IWM were registered as a cross-asset subset for cadence, seed replay, and held-instrument transfer; these names were chosen before evaluating outcomes.

The base uses up to 1,008 eligible past observations. The last 63 eligible observations form gate validation. The preceding 63 eligible observations whose label endpoint precedes the first gate decision form temperature calibration. Fitting observations must have label endpoints preceding the first calibration decision. On the regular aligned dataset this leaves 870 estimator-fitting rows; the implementation derives the two purges from timestamps, rather than assuming a fixed row gap. A label is `close[t+6]/close[t+1]-1`, classified DOWN below −1%, UP above +1%, otherwise FLAT, and unavailable when its endpoint has not occurred. The first permissible execution is the following close.

Every learned mean and scale is fitted inside the estimator-fitting partition. Logistic regression fixes `C=.1`, `max_iter=500`. Histogram gradient boosting fixes 100 iterations, seven leaves, regularization 10, learning rate .05, minimum leaf 40, and disables early stopping. Equal component weights are fixed. Temperature candidates {.75, 1, 1.5, 2, 3} are chosen on the calibration partition only. That is a chronological inner calibration choice; there is no outer hyperparameter tuning or winner selection requiring an additional nested optimization loop.

The same estimators produce seven probability views: prior, logistic, tree, raw equal mixture, calibrated equal mixture, calibration-reused gate-to-prior, and separately validated gate-to-prior. For the last two *diagnostic forecast* views, gate closure returns the fitted prior. This is an explicit experimental intervention, not a description of the existing production model's forecast semantics. The production portfolio gate determines the allocation between trend and model intent while its hybrid probability vector remains calibrated. Our economic experiments use those allocation semantics.

The calibration-reused gate judges the same calibrated probabilities on the block used to select their temperature. The separated gate judges them on a later, distinct, endpoint-purged block. Both require a .01 log-loss improvement over the fitted prior. Both share identical base estimators and calibration temperature, so their contrast isolates the evaluation-block choice. It is not an exact re-run of the older 126-session calibration implementation, whose estimator-fitting window differed.

## Controlled variants

Four disjoint feature families are removed and each model is refitted: momentum (six trailing return horizons plus 12-minus-1-month momentum), trend (three moving-average deviations), risk (two trailing volatilities, opening gap, intraday range), and context/volume (two SPY-relative returns plus volume ratio). This is a controlled removal experiment; feature importance is not substituted for refitting.

Data-efficiency variants retain the most recent 50% or 75% of eligible estimator-fitting rows. The window variant uses 504 eligible observations instead of 1,008. Selecting the most recent subset preserves chronology but changes both sample size and historical coverage, so this study cannot separate pure sample-size effects from recency. Training and outer-validation log losses are saved for each variant and outer block. A 21-session refit cadence is compared with the 63-session base on the same six-symbol dates. Model-age slices are descriptive and can be confounded by differing market dates.

Held-instrument transfer fits a pooled model on all 21 *other* symbols and evaluates the excluded target in each later outer block. The target is absent from estimator fitting, normalization, calibration, gating, fitted priors and class-return estimation. There is no target-symbol dummy or target-specific learned parameter. Contemporary SPY history remains a causal covariate. Held labels provide a genuine instrument holdout, but ETF constituents overlap economically, so the study is not a claim of independent-sector or unseen-market transfer.

Seeds 42, 137 and 2026 are replayed on the six-symbol base subset. The chosen algorithms use no observation subsampling and disable early stopping, so identical seed results are possible and should be reported as such. Seeds are never counted as additional market observations.

Legacy XGB/LGBM/LSTM/MoE saved results are excluded from direct comparison: their artifacts do not bind their transformations, label endpoints, calibration roles and execution labels to these outer partitions. Rebuilding every architecture's contract exceeds the fixed experiment budget. The fixed linear and nonlinear classifiers supply eligible baselines; no claim is made against the excluded architectures. Macro/news/crypto covariates are excluded for missing historical availability evidence, not replaced by artificial zeros.

## Dependence, multiplicity and portfolios

There are 17 predictive contrasts registered by name in the design. Each paired difference is first averaged over the same instruments on each date. Bootstrap resampling keeps all instrument observations together, uses circular contiguous blocks *within each outer quarter-length block*, and retains the separate historical block boundaries. Two thousand replicates and seed 20260928 are fixed. Block lengths 6, 10 and 21 are reported; 10 is primary. Holm adjustment uses the actual full 17-contrast predictive family for each block sensitivity, not an unexplained fixed count. Approximate 80% detectable effects use 2.802 times the block-bootstrap standard error; these are descriptive power limits.

Economic comparisons run ten policies through the existing shared cash-constrained execution ledger: logistic, tree, raw mixture, calibrated mixture, fixed trend/ML mixture, calibration-reused gated allocation, separately gated allocation, trend, equal weight, and cash. Every historical block starts with the same $10,000, costs and risk constraints. Target orders execute at the following observed close. Rebalancing uses the last actual exchange session of each week, including holiday-Friday handling, with the first observation as initialization. Quarterly capital resets prevent gaps between the sampled historical blocks from being misrepresented as a continuous investable track record. Terminal open positions remain marked to market. The eight registered policy-versus-equal-weight contrasts form a separate Holm family; there is no selection of a profitable subset of contrasts.

All exact targets, events and daily account trajectories are retained under `experiments/portfolios/`. Model returns use class means estimated only from eligible fitting labels; each forecast's expected return is paired with the same fixed .3% intent threshold. The adaptive allocation is `(1-.5*gate)*trend + .5*gate*ML`, while the fixed mixture uses .5 of each without a gate. The same concentration, growth-group, volatility, gross-exposure and turnover constraints apply to every noncash policy.

## Validation and reproduction

`tests/test_research_v2_experiments.py` checks strict label-endpoint ordering, disjoint calibration/gate partitions, eligible mature labels, future-row mutation invariance, training-only scaler means, exclusion of held-target data from all learned transformations, probability finiteness and normalization, schema rejection, serialization equality, checksum failure, and bootstrap/step-down multiplicity mathematical controls. Passing these tests is evidence about the implementation; it is not evidence of profitable trading.

The following command **actually refits** a base AAPL model, its momentum-family ablation, and a pooled model trained on all 21 other symbols with AAPL held out. It compares all seven saved probability views, endpoints, training hashes, temperature and gate decisions. It is a representative numerical reproduction, not a claim that every trial has been independently rerun.

```sh
EXPERIMENT_REPRODUCTION_OUTPUT=data/research-v2/experiments/clean-reproduction.json \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
data/research-v2/clean-env/bin/python data/research-v2/experiments/clean_reproduce.py
```

The declared numeric tolerance is `rtol=1e-10, atol=1e-12`. Existing reproduction reports are never overwritten. Runtime reports disclose single-thread settings, exact Python, hardware, peak process memory, compressed model bytes, and fit/inference times; peak process memory is not incremental per-fit memory and the process is warm after import.

## Result locations

The authoritative machine-readable result is `data/research-v2/experiments/summary.json`; `run.json` supplies runtime and completion accounting. All model decisions are in `predictions.parquet`, per-fit metadata in `fits.parquet`, and raw paired per-date effects in the `R*-daily.csv` files.

Manuscript tables are `data/research-v2/tables/exp_predictive.csv`, `exp_contrasts.csv`, `exp_learning.csv`, `exp_transfer.csv`, `exp_seeds.csv`, `exp_model_age.csv`, `exp_runtime.csv`, `exp_reliability.csv`, `exp_portfolios.csv`, and `exp_economic_contrasts.csv`. Figures use those records directly: `data/research-v2/figures/exp_contrasts.png`, `exp_learning.png`, `exp_runtime.png`, and `exp_reliability.png`. Reliability bin counts are explicit in the source table; sparse visual bins do not constitute standalone statistically significant calibration findings.

All favorable, adverse and inconclusive contrasts must remain in the manuscript. Historical observations alone do not approve the historical-edge or prospective gate.

Output compatibility: root `tables` and `figures` are symlinks into the canonical `data/research-v2/` directories. This output-only path correction preserves the registered source hash; details are in `experiments/output_path_compatibility.json`.

## Immutable checkpoint portability

The registered `python -m` invocation serialized the two research classes under `__main__`. The added `research_v2/models.py::load_checkpoint` maps exactly those two names during trusted-local, checksum-verified loading, without changing caller globals, old model files, manifests or the registered fitting source. Its tested joblib/zlib implementation belongs to the pinned runtime. The actual CLI-produced AAPL checkpoint is loaded and its seven probability views checked against all saved forecasts in the seventh behavior test. `test_manifest_portable.json` binds the loader and all seven passing test IDs to their exact source and JUnit. The original six-check test report is preserved separately.

```python
from research_v2.models import load_checkpoint
model = load_checkpoint("data/research-v2/experiments/checkpoints/base-AAPL-F0-R0-S42/model")
```

## Completed result digest (generated from saved statistics)

The registered run completed 848 of 848 trials and 1696 estimator fits, with 4 additional runtime-pilot fits. It saved 352,800 probability-view rows; these comprise variants and seeds, not independent samples. The base sample contains 5,544 symbol-date pairs over 252 distinct test dates, spanning 2024-01-02 00:00:00+00:00 to 2025-09-29 00:00:00+00:00 in the four predeclared blocks. Registered fitting/checkpoint generation took 395.86 seconds (postprocessing is separate).

The fitted prior attained log loss 1.063799 and accuracy 42.190%; the calibrated mixture attained 1.077345 and 40.133%. Calibration improved the raw mixture by -0.014551 log-loss units, with block-10 95% interval [-0.022307, -0.005981] and Holm-adjusted p=0.017991. Improving over an overconfident raw model did not establish an advantage over the fitted class prior.

The gate opened on 39.773% of base fitting instances when reusing calibration, versus 27.273% using separate validation. Its independent-block predictive contrast was -0.004618, with interval [-0.012411, 0.002938], so no incremental benefit was established. Gate-to-prior forecast diagnostics and trend/ML portfolio allocation remain different interventions.

Removing any individual feature family did not survive the registered multiplicity control. The 50% fit fraction worsened loss by 0.036537; the 504-observation window worsened it by 0.044181. Both survived the primary family correction. The cadence contrast was 0.000416 with an interval spanning zero. Pooled held-instrument transfer changed log loss by -0.022180, with interval [-0.046636, 0.000213] and Holm p=0.458271; this is inconclusive.

All eight registered economic contrasts had primary block-bootstrap intervals spanning zero. None supports superiority over matched equal weight. Linear/tree expert predicted classes disagreed on 36.003% of the 5,544 base samples. All three registered seeds produced identical probabilities on the selected replay subset (maximum absolute difference 0.0); repeating these seeds adds no independent market information.

Integrity QA checked 1696 model/prediction file hashes and 1328 instrument-level role partitions, including all 24 true held-instrument pooled fits. Every partition passed strict endpoint ordering. `final_verification.json` preserves package versions and this check; `exp_fit_boundaries.csv` exposes every boundary.

Additional descriptive source tables are `exp_seed_variance.csv`, `exp_transfer_class_metrics.csv`, `exp_transfer_confusion.csv`, `exp_disagreement.csv`, `exp_confidence_coverage.csv`, and `exp_reliability_uncertainty.csv`. Confidence thresholds were fixed before this postprocessing and do not choose an operating threshold. Calibration bands resample date blocks and preserve the ratio of class events to bin observations, including zero-support dates; they are descriptive pointwise intervals, not a new inferential comparison family or simultaneous bands. The figure `exp_calibration_uncertainty.png` uses exactly that source table. `supplement.json` hashes its input and outputs.

## Minimal-core clean-environment adapter

The first clean-environment reproduction attempt stopped before any fit: importing `src.models.adaptive` triggered the legacy `src/models/__init__.py`, which eagerly imports `ModelTrainer` and requires the unrelated optional `mlflow` stack. The failure and original log hash are retained in `trials.jsonl`; it is not counted as a successful reproduction.

The separate `experiments/clean_reproduce.py` compatibility wrapper verifies every registered source hash against both current files and the source archive, creates namespace-only packages to bypass that initializer, and executes the exact archived `experiments.py` and `adaptive.py`. It preserves registered data/output paths and changes no fitting mathematics, settings, partitions or observations. Its output records the adapter hash and executed archived source paths. This is the command shown above for the minimal-core clean environment; the ordinary `-m ... reproduce` path remains usable only in the fuller original environment. Reproduction completion is established by the generated clean-reproduction receipt, not by the existence of this wrapper.

## Additional interpretation limits

The block intervals condition on the four registered historical blocks. They do not resample a broad universe of market regimes or eliminate universe survivorship and historical-exposure limitations. The 2.802×SE detectable-effect approximation uses an unadjusted two-sided 5% test; multiplicity control would require larger effects. Consequently these approximate thresholds are optimistic lower bounds for a family-controlled deployment claim.

Early base fits overlapped other work on this host, and variants ran in a fixed order. Latency differences between variants therefore reflect observed operational conditions; they are not controlled evidence that dropping a feature family causes faster training. The plotting correction `exp_learning_v2.png` moves the legend away from the bars and uses the same `exp_learning.csv` values. The first figure and its original source remain preserved.

A separately registered validation-only budget in `full_reproduction_protocol.json` permits a full clean-environment replay without adding model candidates, selection opportunities, endpoints, or hypotheses. It preserves the original research-selection budget and artifacts. Success requires actual refitting of all registered trials plus numerical comparison of every forecast field and non-runtime fit boundary, followed by rerunning all portfolio histories from the reproduced probabilities. The final `full_reproduction.json` receipt determines completion; registration alone is not success.

## Full clean reproduction completed (generated from receipt)

The separately registered validation replay completed all 848 trials and 1696 estimator fits in 273.880 seconds. It regenerated 352,800 forecast-view rows and compared every saved forecast field and every non-runtime fitting-metadata field against the original artifacts. Maximum absolute probability error was 0.0. The tolerance was rtol=1e-10, atol=1e-12; matching was actual numerical evaluation, not a hash-only assertion.

The reproduced forecasts then drove 40 complete portfolio histories comprising 2,520 account sessions. All saved ledgers, target panels and complete event lists matched their originals; economic metric and block-bootstrap/Holm tables also matched. These are quarterly reset histories, not 2,520 independent market dates. The initial missing-mlflow import failure remains preserved separately in `clean-import-failure.log` and the trial log; it was followed by successful representative and full archived-source replays.

The authoritative full receipt is `full_reproduction.json`; its pre-execution budget is `full_reproduction_protocol.json`, command output is `full_reproduction.log`, and all newly refitted models/predictions/accounts are retained in `full-reproduction/`. Original source, models, manifests and outputs remain unchanged. Original research and pilot fits total 1,700; the earlier representative clean check used six further fits, and this full validation used 1,696. These validation refits do not add candidate-selection trials or independent statistical evidence.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
data/research-v2/clean-env/bin/python data/research-v2/experiments/full_reproduce.py run
```

The command reuses the completed validation receipt; it does not silently repeat the budgeted work. A new independent reproduction requires a new output location and registered validation budget.
