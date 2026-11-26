# Causal Evaluation and Capital-Constrained Replay of Trading Rules and Classifiers

Working methods draft, revised 28 September 2026. **Not a publication-ready
profitability claim.** The earlier draft is retained in `docs/archive/`.

## Research question

After repairing cash accounting, temporal validation and execution semantics,
do directional classifiers or a regime-dependent selection rule provide
repeatable net performance beyond passive and simple-rule baselines?

## Implemented methods

We use a frozen adjusted-price snapshot for 22 equities and ETFs. Observations
are restricted to completed XNYS sessions with valid OHLCV, using the common
sample across instruments. Adjusted history may include vendor revisions and
the universe is selected retrospectively; neither is represented as a
historically reconstructed investable universe.

The first 252 eligible return/volatility observations calibrate a Gaussian HMM.
Training-only scales and frozen parameters are used for later forward filtering;
the calibration window remains unknown in historical features. We do not use
full-series Viterbi smoothing as a trading-time observable. Model features
undergo supervised selection within purged chronological training folds.
Class weights come from eligible training labels. Inner validation and outer
prediction splits remove overlapping forward-label horizons. XGBoost,
LightGBM and unidirectional LSTM probabilities use fixed static ensemble weights.
The LSTM receives the required preceding feature sequence for each prediction.

A separately fitted MoE gate uses outcomes only after labels have matured;
its validation training observations are not described as independent gate
performance. Static and dynamic stop/Kelly simulations are exploratory risk
ablations; they are distinct from the executable shared-account contract.
Classifier accuracy is compared with always-UP and training-majority baselines.

Training ends in 2023; 2024 is the declared selection period; 2025 onward is
historical evaluation. These dates were examined before this repair. We therefore
call all historical findings exploratory, even when the newly fitted models
have not seen the evaluation targets. The prospective protocol reserves a
future holdout beginning no earlier than 29 September 2026.

Strategy signals are also executed through a shared cash and position account.
Targets produce holding deltas, with pending cash reservations, per-instrument,
gross, turnover and exposure-group limits. Fees and modeled slippage are charged
on both sides. Fills occur at the next observed close. Terminal holdings remain
marked, and every close reconciles cash, holdings and net P&L. This account has
no leverage, shorts, profit targets or broker bracket orders. Prospective
broker fill timing has not been validated.

The implemented momentum rule is a time-series filter, not cross-sectional
quintile selection. The pairs rule trades a single relative-value leg, not a
hedged cointegration portfolio. In the absence of eligible macro data, the
carry implementation is reported as an SMA trend fallback. Macro and sentiment
features without release, vintage and content-availability evidence are
excluded from confirmatory interpretations.

## Statistical reporting

Metrics use matching dates, explicit costs, a 252-session annualization and a
fixed 5% annual hurdle. Sortino uses downside deviation relative to that hurdle;
alpha uses excess returns; drawdown includes starting NAV. Individual-ticker
averages are not portfolio results. Evidence includes signed HAC mean statistics,
block-bootstrap uncertainty, turnover, exposure, fill counts, cost stress and
Bonferroni adjustment over the declared comparison family. All searches and
training budgets are recorded; sensitivity cases do not become new selected
winners based on evaluation outcomes.

## Findings and claims

Numerical tables and figures are generated from the immutable experiment
outputs linked in `REPAIR_CHECKLIST.md` and the readiness report. We do not
carry forward the previous Sharpe, alpha, win-rate, novelty, outperformance or
"production-grade" claims. The corrected quick screen did not establish a
qualifying strategy. Its training budget was deliberately bounded, so it does
not prove that no better model or strategy can exist.

The supported engineering result is that the tested account reconciles and
historical replay and local paper execution share an event contract. A simulated
profit is not prospective evidence or proof of durable excess returns. Before
publication, the future holdout, provider-vintage evidence, broker reconciliation,
robustness across unseen periods and claim-to-artifact review remain necessary.
