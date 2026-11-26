# Adaptive release protocol, registered before the new data download

Recorded 28 September 2026 (Hong Kong). This is a new experiment, not a
replacement for the unsuccessful fixed-model results. Parameters below are
fixed before downloading the July–September extension. No parameter search
or candidate redesign is permitted after viewing its results within this run.

- Universe: the same 22 cached US equities and ETFs. Current membership creates
  survivorship and selection limitations; this is not a market-wide claim.
- Download ten years of completed daily observations into a new, immutable run
  directory. Require the latest completed XNYS session for every instrument.
  Never fabricate observations to bridge an unavailable provider history.
- Exploratory development: 2024–2025. Selection period: 2 January–10 July 2026.
  Reserved recent historical audit: 13 July–25 September 2026. These dates
  follow the last observation in *any* previously inspected cache. The audit
  is retrospective and short; it does not replace prospective evidence.
- Prospective pilot remains reserved from 29 September 2026, for 63 completed
  sessions. No backfilled decision counts as a prospective decision.
- Predict the five-session return from the next close: close[t+6]/close[t+1]-1.
  DOWN below -1%, UP above +1%, otherwise FLAT. These execution-aligned labels
  differ from the earlier screenshot's close[t+5]/close[t]-1 labels.
- Per instrument, refit every 21 observations with at most 1,008 eligible rows.
  At decision t, require every training label to have ended strictly before t.
  Use 126 trailing rows only for temperature calibration, with a six-session
  purge between fitting and calibration. Minimum model fitting history: 504.
- Fixed features: lagged returns, trailing volatility, price/average ratios,
  intraday range/gap, volume ratio, and relative returns versus SPY. No price
  levels, full-sample scaling, HMM, revised macro or unarchived news inputs.
- Models: scaled regularized multinomial logistic regression (C=0.1), shallow
  histogram gradient boosting (100 iterations, seven leaves, L2=10, learning
  rate 0.05, minimum leaf 40), and their equal probability mixture. No class
  balancing. Temperature choices: 0.75, 1, 1.5, 2, 3, selected only on calibration.
- Report accuracy, balanced accuracy, macro F1, log loss and multiclass Brier
  score against always-UP and trailing training-frequency baselines. Statistical
  resampling uses dates as blocks, keeping instruments together.
- Trading candidates: trend rule; calibrated ML; adaptive hybrid (half rule,
  half ML when calibration log loss improves over its training prior by at
  least 0.01; otherwise rule only). Trend requires price above trailing 200-day
  average and positive 12-minus-1-month momentum. ML requires positive expected
  return after 30 bps round-trip costs. Weekly Friday targets, next-close fills.
- One $10,000 account each; total risky target at most 80%, individual 10%,
  growth group 35%, daily turnover 40%, drawdown stop 20%; 10 bps commission
  plus 5 bps slippage per side. Trailing 63-session covariance, 50% diagonal
  shrinkage, annual portfolio volatility cap 12%; no leverage. Idle cash earns
  zero. Benchmarks: identically costed 80% equal-weight universe and cash;
  report SPY total-return proxy as an additional descriptive reference.
- Select one candidate using selection-period net Sharpe, then lock it for
  the recent audit. Monthly retraining is part of the frozen algorithm; no
  outer-test parameter tuning. Audit improvement must survive a one-sided HAC
  comparison, 10-date circular block bootstrap and Bonferroni family of 333
  (330 earlier comparisons plus three new candidates). A short audit alone
  cannot approve a profitable deployment claim.
- Approvals are separate: software contract tests; current-data/model readiness;
  historical edge; prospective operation (63 sessions); publication artifacts;
  profitable-trader claim. The combined decision remains blocked when required
  evidence is absent. Eligible local deployment is cash until both historical
  edge and prospective evidence pass. Research shadow accounts may trade.
- Preserve source, configuration, provider retrieval time, raw/adjusted snapshots,
  probabilities, fold cutoffs, models, fills and checksums. External broker
  submission is outside this release's authorization and execution contract.
