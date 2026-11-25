# Monthly progress report: Adaptive Ensemble Trading System (CECI2)

*Generated 25 November 2026 from `data/review-v1/` by `python -m review_v1.build report`. Every number below is read from the saved statistics. Live demo: `http://127.0.0.1:8510` (see `docs/review-v1/REPRODUCE.md`).*

## Summary

The pipeline works end to end: it collects and validates daily data for 22 instruments, produces forecasts, applies a decision rule and risk limits, and records paper orders and fills with a hash-checked audit trail. In October 2026 it recorded 5 decisions and 25 paper fills, and every one of them can be reproduced exactly from the inputs recorded at the time.

The strategy is not yet shown to be good. On 14,960 forecasts the classifier is worse than the prior (log loss 1.0762 against 1.0641). The trend rule that actually places the paper orders made +35.0% on the 2024 to September 2026 history, less than holding all 22 instruments equally (+55.2%), and lost 2.5% in a simulated replay of the 41 sessions since. A working pipeline is not evidence of a profitable strategy, and this report does not claim one.

## What has been built

| Stage | What it does | Role in a decision |
|---|---|---|
| Data collection | Downloads daily bars for 22 instruments after the US close into immutable, hashed snapshots; validates them against the exchange calendar | Input |
| Features | 17 price and volume features per instrument | Input |
| Classifier | Logistic regression and boosted trees, averaged and calibrated, giving P(DOWN), P(FLAT), P(UP) for a five-session horizon | **Recorded, not used** |
| Decision rule | Hold an instrument when its price is above its 200-session average and its 12-month momentum is positive | **Decides** |
| Risk limits | 10% per instrument, 35% growth group, 80% gross, 12% volatility target, 40% turnover per rebalance, 20% drawdown stop | **Decides** |
| Paper ledger | Orders fill at the next close with 15 bp cost; cash reserved; every event stored | Executes (paper only) |
| Market-regime HMM *(new)* | Causal rolling two-state hidden Markov model on SPY | **Diagnostic** |
| News sentiment | FinBERT scores of headlines | **Not connected** |
| Review dashboard *(new)* | Nine sections, a step-by-step replay of recorded decisions, a button that runs the current pipeline | Demonstration |

No broker is connected and no external order has ever been sent. The deployment account holds cash; the trend portfolio runs as a paper *shadow* account to gather evidence.

## Data collected

- **Market history:** 22 instruments, 55,242 daily bars, 2016-09-29 to 2026-09-25, from Yahoo Finance public daily chart, sealed with hashes on 27 September 2026.
- **Scheduled collection, October 2026:** 10 hashed snapshots, 5 decision dates (1 October, 2 October, 14 October, 15 October, 22 October), 51 paper orders, 25 fills, and 88 forecasts later matched to their outcomes.
- **Where collection broke:** 6 collection attempts were registered between 29 September and 22 October. Together they scheduled 18 decision sessions and recorded 5 (week1: 0 of 5; r2: 2 of 5; r3: 0 of 3; r4: 2 of 3; r5: 0 of 1; r6: 1 of 1). 5 attempts halted: 4 windows passed with no run (the collector is a local launch agent that needs a logged-in, awake machine) and 1 download was rejected by validation because one instrument's open was above its high. The system halts by design instead of backfilling, so the recorded sample is small but clean.
- **News:** 40,816 cached headlines for AAPL, NVDA, SPY, plus 794 headlines fetched for this review covering the pilot weeks (none were returned for 10 of the 22 instruments, mostly funds).
- **Data quality:** the 902 daily returns behind the simulated replay were compared with a second source (Alpaca market data v2, consolidated daily bars, adjusted): largest difference 0.01% of return. The largest move in the window, NVDA +31.5% on 23 November 2026, appears in both sources.

## What the classifier result means

> On next-close labels with a fixed ±1% threshold, neither the logistic/tree mixture (848-trial study) nor the Hedge and stacking controllers (30 October screen) beat a simple class-frequency prior.

**The label.** After each close the classifier forecasts the return from the *next* session's close to the close five sessions later: UP above +1%, DOWN below −1%, FLAT otherwise. The ±1% band is the same for every instrument and every market condition.

**The baseline.** The class-frequency prior ignores today's data and predicts each instrument's class shares from its training window as probabilities (on average DOWN 32%, FLAT 27%, UP 41%), so its single best guess is UP on 76% of instrument-days.

**The metric.** Log loss: the average of −ln(probability given to what happened). Lower is better; guessing one third each scores 1.0986. Comparisons are paired by date with a block bootstrap.

**What it tells us.** Under this exact setup (17 price and volume features, a five-day horizon, a fixed 1% band, these 22 instruments, 2024 to 2025), the models' day-to-day probabilities add **no measurable information** beyond long-run class frequencies. When the models move away from the base rate they are wrong often enough to lose what they gain. "Did not beat" for the controllers means "no reliable difference", not "proven worse": the studies could only detect gaps of roughly 0.01 to 0.02 in log loss.

**What it does not tell us:**

- **Not whether the pipeline works.** Data collection, validation, the decision rule, risk limits and the paper ledger are separate components, verified separately in this review.
- **Not how the logged trades performed.** Every recorded order came from the trend rule. The classifier's probabilities were saved next to each decision as a diagnostic and did not influence it.
- **Not that prediction is impossible.** Only this label, these features and these models were tested. The proposal's label (next open, with a threshold scaled to cost and volatility), text features and the learned gate are untested.
- **And the reverse:** a pipeline that executes a decision correctly is no evidence that the decision was a good one. Trading performance is judged separately, against a trading benchmark.

The full explanation, with the table for both studies, is on the dashboard's first page.

## One decision, traced from inputs to outcome

Dashboard section 3 replays the recorded decision for **NVDA at the close of 1 October 2026** in seven steps. Labels on every panel say whether a fact was *recorded* at the time, *reconstructed* now from recorded inputs, or *simulated*.

1. **Inputs (recorded).** At 20:30 UTC the collector downloaded 2,513 daily bars for NVDA; last close $230.86. No news was collected.
2. **Features (reconstructed).** Price +15.4% against its 200-session average; 12-month momentum +16.7%.
3. **Models.** Recorded forecast: DOWN 40%, FLAT 21%, UP 39%. HMM (reconstructed): calm, P(turbulent) = 0.000. **Neither was used.**
4. **Decision (recorded).** Trend rule on, target 3.64% of equity; 12 of 22 instruments qualified (43.6% gross). No portfolio limit was binding; the 40% turnover cap was binding and scaled every order to 91.7%.
5. **Order (recorded).** Buy 1.443876 units, reserve price $237.79 (close + 3%), $343.85 cash reserved. Paper only.
6. **Fill (recorded).** Next close, 2 October 2026: $233.95, cost $0.51. After all 12 fills the account stood at $9,993.96.
7. **Outcome.** Five sessions after entry NVDA was -2.00% (class DOWN; the classifier's top class was DOWN). The pilot segment halted on 8 October 2026 after missed sessions, so the recorded ledger stops at $9,993.96. Marked with later public prices (reconstructed), the NVDA position stood at -113.95 USD (-33.7%) on 24 November 2026, and the recorded portfolio simply held at $9,761.16 (-2.39%).

The replay re-checks the receipt's registered hash and stops if any recomputed probability, weight or order quantity differs from the recorded one. One case shows the mechanism; it says nothing about skill.

## Evaluation

### Classification, against the class-frequency prior

| Sample | Status | Forecasts | Model log loss | Prior log loss | Model − prior (95% interval) | Accuracy, model / prior |
|---|---|---:|---:|---:|---|---|
| History, 2024-01-02 to 2026-09-17 | Out-of-fold, previously inspected | 14,960 | 1.0762 | 1.0641 | +0.0121 (+0.0044 to +0.0194) | 40.2% / 42.8% |
| October 2026 collector | **Recorded**, 4 dates | 88 | 1.1189 | 1.1060 | too few dates for an interval | 27.3% / 30.7% |
| 29 September to 16 November 2026 | **Simulated** replay | 770 | 1.1285 | 1.1234 | +0.0051 (-0.0273 to +0.0333) | 31.0% / 30.9% |

By 95% interval the classifier is worse than the prior on the full history and 2024 to 2025; not distinguishable from the prior on January to July 2026 and July to September 2026. Split by HMM regime: calm +0.0107 (worse than the prior), turbulent +0.0172 (worse than the prior), Holm-adjusted. Balanced accuracy is 38.7% for the classifier against 37.4% for the prior; that measure averages recall over the three classes, and the prior names DOWN as the likeliest class on only 4% of rows.

### Trading, against an equal-weight benchmark

Assumptions for every row: $10,000.00 paper capital; 10 bp commission + 5 bp slippage on traded notional, both sides (15 bp per side); fills at the next observed close after the decision; fractional units; long only; limits 80% gross, 10% per instrument, 35% growth group, 12% annualised volatility target, 40% turnover per rebalance, 3% cash buffer, 20% drawdown stop; Sharpe annualised, zero risk-free rate, 252 sessions. Benchmark: equal weight of the same 22 instruments through the same ledger and costs; cash; SPY buy-and-hold as an unmatched reference.

**History, 2024-01-02 to 2026-09-25 (686 sessions, 138 weekly rebalances; previously inspected, so exploratory)**

| Portfolio | Net return | P&L (USD) | Sharpe (95% interval) | Max drawdown | Fills | Costs | Mean invested |
|---|---:|---:|---|---:|---:|---:|---:|
| Equal weight, all 22 instruments (benchmark) | +55.2% | +5,522 | 1.40 (0.36 to 2.67) | -15.7% | 2,689 | $55.67 | 76% |
| Trend rule (pilot shadow) | +35.0% | +3,497 | 1.30 (0.23 to 2.46) | -10.4% | 2,432 | $184.81 | 61% |
| Equal weight × (1 − P(turbulent)) | +35.1% | +3,509 | 1.24 (0.15 to 2.42) | -9.0% | 2,662 | $179.05 | 63% |
| Trend rule × (1 − P(turbulent)) | +22.6% | +2,259 | 0.98 (-0.13 to 2.11) | -9.2% | 2,400 | $216.52 | 51% |

The trend rule's Sharpe ratio minus the benchmark's is -0.10 (95% interval -0.67 to +0.50): not distinguishable from the benchmark, so no edge is shown. Over the same dates buy-and-hold SPY returned +68.5%, so a positive return here is what a rising market gave any invested portfolio.

**Simulated replay, 2026-09-29 to 2026-11-24 (41 sessions; not recorded trading)**

| Portfolio | Net return | P&L (USD) | Max drawdown | Fills | Costs | Mean invested |
|---|---:|---:|---:|---:|---:|---:|
| Equal weight, all 22 instruments (benchmark) | -4.9% | -486 | -8.8% | 166 | $13.60 | 74% |
| Trend rule (pilot shadow) | -2.5% | -250 | -4.3% | 120 | $19.01 | 42% |
| Equal weight × (1 − P(turbulent)) | -1.8% | -180 | -3.8% | 189 | $24.54 | 56% |
| Trend rule × (1 − P(turbulent)) | -1.1% | -107 | -2.5% | 125 | $19.32 | 32% |

Every invested portfolio lost money in this window (SPY buy-and-hold -5.9%). **No Sharpe ratio is reported: 41 sessions containing one market move cannot separate skill from noise.** The recorded paper ledgers are shorter still: at most two marked sessions per segment, so the recorded P&L is just the cost of the first purchases (about $6.04 per segment).

### Sandbox experiment: does the HMM help as an exposure control?

Registered before any result (`data/review-v1/registration.json`): Does scaling the weekly targets by one minus the HMM turbulence probability improve the risk-adjusted result of its parent portfolio after costs? Budget: five portfolios, two contrasts, Holm-adjusted.

| Contrast | Sharpe difference (95% interval) | Holm p | Verdict |
|---|---|---:|---|
| trend_regime − trend | -0.32 (-0.82 to +0.17) | 0.41 | not established |
| equal_weight_regime − equal_weight | -0.16 (-0.90 to +0.69) | 0.72 | not established |

**Not established.** With the overlay the trend rule's maximum drawdown went from -10.4% to -9.2%, its net return from +35.0% to +22.6% and its Sharpe ratio from 1.30 to 0.98. In the simulated replay the overlay version returned -1.1% against -2.5%; the HMM's first turbulent close in that window was 5 November, after which the overlay held less. That is one episode, not a test. The HMM's own checks pass: 93 of 93 refits converged, removing later data changes no earlier probability, and volatility over the next 21 sessions averaged 20.4% after turbulent closes against 12.5% after calm ones (states: about 10% and 29% annualised volatility).

### Sentiment

Sentiment has no path into a decision: the pilot's 17 features are all price and volume, and every recorded decision reproduces from prices alone. Two diagnostics were run on cached headlines scored with FinBERT.

- **Is tone related to anything the system uses?** Rank correlations of daily tone, with the classifier's score: AAPL +0.04, NVDA +0.02, SPY +0.00 (no interval excludes zero); with the trend flag: AAPL +0.10, NVDA +0.19, SPY +0.05 (interval excludes zero for AAPL and NVDA); with the next five-session return: AAPL -0.01, NVDA -0.12, SPY -0.08 (interval excludes zero for NVDA).
- **Does it help the classifier?** Adding six FinBERT columns to the classifier for NVDA and SPY changed log loss by +0.0462 (95% interval +0.0103 to +0.0848; lower is better): the forecasts got worse on these two tickers. Exploratory: two tickers, one feature design.

## What the demo proves, and what it does not

**Proves (operational):**

- The pipeline runs from download to paper order on a current session in about half a minute (dashboard section 8).
- All 5 recorded decisions match their registered hashes and reproduce exactly: probabilities, trend flags, weights and order quantities.
- The evaluation harness reproduces the sealed September accounts to the cent, so the trading numbers above come from the same ledger code as the pilot.
- The HMM is causal and stable; the statistics agree with an independently generated table.

**Does not establish:**

- That the classifier predicts anything beyond base rates.
- That the trend rule, with or without the regime overlay, beats holding everything equally.
- That sentiment or the HMM improves a decision. Neither has ever influenced one.
- That the collector can run unattended: 5 of 6 attempts halted, and only 5 of 18 scheduled decision sessions were recorded.
- The proposal's main claim (a learned gate beats a static ensemble). That experiment has not been built.

## Status against the proposal

9 components work and are verified, 2 are diagnostic only, 5 exist but are not validated, 6 are missing. The dashboard's last section lists the evidence and a check for each.

- **Works (verified):** Daily market data collection and validation; Feature pipeline; Classifier software; Decision rule and portfolio risk constraints; Paper ledger; Receipts and audit trail; Classifier statistics; Trading statistics; Review dashboard and decision replay.
- **Diagnostic only:** Hidden Markov model of market regimes; News sentiment.
- **Exists, not validated:** Unattended daily collection; Predictive value of the classifier; Trading edge of the trend rule; Summer prototype stack; Prototype HMM in `src/features/regime.py`.
- **Missing:** Main experiment: learned gate versus static ensemble; Proposal label; HMM state-count selection and use as a gate input; Text data beyond headlines; Price history back to 2010; A prospective sample long enough to judge.

## Plan before the final presentation

1. **Run the collector without gaps.** Move it to an always-on machine and log the trend, equal-weight and overlay portfolios in shadow every session, aiming for at least 63 recorded sessions. Nothing else substitutes for prospective evidence.
2. **Implement the proposal's label** (next-open entry, threshold scaled to cost and volatility), refit, and rerun the classifier-against-prior comparison with the same statistics. With the fixed ±1% band the class mix moves with volatility; a scaled band removes that, and whether the classifier then beats the prior is an open question.
3. **Build the main experiment:** the learned gate over the expert models against the static ensemble, through the same decision and ledger code, with contrasts registered first. Include regime probabilities and sentiment as gate inputs, each with its own with/without comparison.
4. **Separate exposure from selection.** Compare the trend rule with an equal-weight benchmark scaled to the same average exposure.
5. **Extend the history only when the stress-period test needs it** (the proposal's 2011 and 2015 to 2016 periods), and keep the second-source price check in the build.

The next experiment to run is item 2: it reuses the existing features, walk-forward harness and statistics, and its result decides whether the classifier has any claim to a place in the decision.

*Latest demo run: session 2026-11-24, 8 of 22 instruments pass the trend rule, 8 pending paper orders, regime turbulent.*
