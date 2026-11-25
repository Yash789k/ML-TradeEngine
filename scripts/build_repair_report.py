"""Generate the review report directly from sealed experiment outputs."""

from pathlib import Path
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
research = ROOT / "data/repairs/2026-09-28-causal-v2"
replay = ROOT / "data/repairs/2026-09-28-shared-v2"
classification = pd.read_csv(research / "classification.csv")
scorecard = pd.read_csv(research / "scorecard.csv")
portfolios = pd.read_csv(replay / "portfolio-results.csv")
manifest = json.loads((research / "manifest.json").read_text())
bootstrap = json.loads(
    (ROOT / "data/repairs/2026-09-28-bootstrap-final/summary.json").read_text()
)


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] * len(headers)) + "|",
        ]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


model_rows = []
for name, g in classification.groupby("model"):
    model_rows.append(
        [
            name,
            f"{g.accuracy.mean():.2%}",
            f"{g.always_up_accuracy.mean():.2%}",
            f"{g.majority_train_accuracy.mean():.2%}",
        ]
    )
portfolio_rows = []
for r in portfolios.itertuples():
    portfolio_rows.append(
        [
            r.strategy,
            f"${r.final_equity:,.2f}",
            f"{r.sharpe_ratio:.3f}",
            f"{r.max_drawdown:.2%}",
            r.fill_count,
            f"{r.bonferroni_p:.3f}",
        ]
    )
report = f"""# Repair report and readiness assessment — 28 September 2026

The accounting, causal-training safeguards and local shared-account implementation
have been repaired and tested. **No candidate qualified for promotion. No external
broker order, alert, remote persistence write or prospective trial was performed.**

## Reproducible evidence

- Full offline verification: **301 passed**, 17 network-provider tests deselected;
  [test output](repair-evidence/verification-final.txt).
- Clean-artifact bootstrap: three models, real LSTM inference, 60 local replay
  events, **{bootstrap['fills']} fills**, restart equality and ledger residual
  {bootstrap['reconciliation_max']:.2e} dollars; [output](repair-evidence/bootstrap-final.txt).
- Research: **22 tickers, 8 zoo implementations, 3 classifiers, static/MoE
  ensembles, static/dynamic risk ablations and 2 baselines**; {len(scorecard)}
  period-specific metric rows, no failed experiments. Training budget: three
  Optuna trials per tree model and ten LSTM epochs, seed 42. This remains a
  bounded experiment, not exhaustive model tuning.
- Frozen source, input and output hashes verified successfully;
  [verification](repair-evidence/hash-verification.txt).
- Five shared-capital account replays: **{int(portfolios.fill_count.sum()):,} fills**,
  maximum reconciliation error {portfolios.max_reconciliation_error.max():.2e}
  dollars; [run output](repair-evidence/replay-final.txt).

The [task-by-task record](REPAIR_CHECKLIST.md) identifies confirmed defects,
changes, checks, affected history and remaining limitations for all seven tasks.

## Classifier results

Evaluation: **2 January 2025–18 May 2026**. Values below are arithmetic means of
per-ticker classification accuracy, with unmatched trailing labels excluded;
they are not portfolio returns. The training-majority class is learned before
evaluation. No model row averages above either listed naive baseline.

{table(['Model','Accuracy','Always-UP','Training-majority'],model_rows)}

Source: [classification.csv](../data/repairs/2026-09-28-causal-v2/classification.csv).
Do not attribute differences from legacy headline numbers solely to accounting:
periods, data admission, regimes, selection and training have also changed.

## One-account historical replay

Each scenario starts with **$10,000** and uses the same 22-symbol observed-price
panel, next-close fills, 10 bp commission plus 5 bp slippage per side, 80% gross
limit, 10% per-symbol cap, 40% per-event turnover limit and 35% growth-group cap.
Each active symbol targets 80%/22 before policy adjustment. One account reserves
pending cash and bears all rebalancing costs. These are model assumptions; they
are not measured broker fills. The fixed 5% annual risk-free hurdle is used only
for excess-return metrics; idle cash earns zero.

{table(['Scenario','Final equity','Sharpe','Maximum drawdown','Fills','Adjusted excess-return p'],portfolio_rows)}

The validation-selected scenario chooses only among zoo strategies using 2024
Sharpe, with CAGR as a tie-break. The momentum, static ensemble, size-policy and
meta scenarios are comparisons; this evaluation was not used to replace the
validation selection. The meta account targets the selected strategy's current
holdings; its warm-up allocation is cash. Static/dynamic stop/Kelly runs in the
research scorecard are separate exploratory ablations, not these account rules.

**No scenario has positive adjusted evidence against SPY.** All active-return
bootstrap confidence intervals include zero. SPY is a frictionless buy-and-hold
reference on matching dates and has a different exposure level; the chart labels
that assumption explicitly. Cash and per-instrument buy-and-hold baselines also
appear in the research scorecard.

![Account replay and drawdown](../data/repairs/2026-09-28-shared-v2/portfolio-comparison.png)

Source: [account results](../data/repairs/2026-09-28-shared-v2/portfolio-results.csv).
Rejections are recorded rather than silently assumed filled: price gaps beyond
the reserved buffer and projected-risk failures reject orders. Last-day pending
orders remain reserved; holdings remain marked. No terminal fill is invented.

## Readiness decisions

- **Accounting:** trustworthy for the tested documented local contracts, with
  exact ledger/P&L reconciliation. Broker margin, short borrow, liquidity,
  fractional-order eligibility and real fills are not verified.
- **Causality and reproducibility:** corrected feature/training paths pass
  prefix/mutation and label-maturity tests; research inputs, source, dependencies,
  configuration and outputs are frozen. Vendor revisions, retrospective universe
  selection and prior inspection prevent a claim of untouched historical evidence.
- **Trading edge:** none established. Zero candidates passed the declared
  closed-trade, positive-performance, block-bootstrap and family-adjusted gates.
  A bounded failed screen is not proof that profitable strategies cannot exist.
- **Capital and risk:** the shared local account executes constrained target
  deltas, reserves pending cash and supports rejection, partial fill, restart,
  idempotency and atomic recovery. Market drift can breach a target cap between
  decisions; increases then fail closed and the breach is logged.
- **Prospective readiness:** ready for further *local* dry runs. Not ready to
  claim broker execution parity or to promote a strategy. Production caches are
  stale and legacy production bundles remain unpromoted. Fresh provider checks,
  remote CI, a qualified candidate and verified authorized broker adapter remain
  gates. [Operational inventory](repair-evidence/operational-status.json).
- **Research claims:** supported claims concern repaired mechanics, reproducible
  exploratory comparisons and the observed lack of qualifying evidence. Durable
  alpha, superiority of HMM/MoE/sentiment, production readiness and prospective
  trading success are unsupported.

## Historical preservation and remaining work

Original data/models/results were preserved. The pre-existing working-tree diff
was saved before edits. Historical README and paper drafts are retained in
`docs/archive/`; the dashboard and legacy paper-number extractor now label their
old outputs. The first quick corrected run and interrupted replay remain separate
from the final runs; only the `causal-v2` and `shared-v2` outputs support the tables
above. MoE's 2024 fitting slice is deliberately ineligible as independent gate
validation; its placeholder cash curve must not be ranked as a gate result.

Macro, crypto and news sentiment were excluded where historical availability or
coverage was inadequate. This is an explicit evidence limit, not a zero-filled
substitute or an inference that those features cannot help. Source guidance:
[FRED real-time periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html)
distinguish present knowledge of revised history from information available at a
past date; [hmmlearn's tutorial](https://github.com/hmmlearn/hmmlearn/blob/main/doc/source/tutorial.rst)
describes whole-sequence decoding, which the historical feature path replaces
with forward filtering.

Bounded follow-up: acquire eligible historical vintages and an independently
specified universe; calibrate trade-outcome probabilities on matured training
outcomes; preregister one additional training budget using the existing validation
slice; keep the reserved future holdout untouched. Any sentiment ablation needs
immutable article content and actual availability timestamps first. Do not
repeatedly tune against the 2025–2026 results.

Follow the [prospective protocol](PROSPECTIVE_PAPER_PROTOCOL.md) before external
paper execution. No automation or broker trial was started by these repairs.

## Commands and artifacts

```bash
python -m pytest tests/ -q
python scripts/local_sandbox.py --output data/repairs/another-bootstrap --capital 10000
python scripts/reproduce_run.py --run data/repairs/2026-09-28-causal-v2 --verify-only
python scripts/reproduce_run.py --run data/repairs/2026-09-28-causal-v2 --output data/repairs/reproduction
python scripts/replay_portfolio.py --research data/repairs/2026-09-28-causal-v2 --output data/repairs/another-replay --capital 10000
```

Use the saved dependency versions when reproducing; numerical equality across
hardware/backends is not guaranteed. The verification command checks immutable
files. A complete second model-training reproduction was not run after the final
experiment; the research rerun itself and the bootstrap both completed.
"""
(ROOT / "docs/REPAIR_REPORT_2026-09-28.md").write_text(report)
print(ROOT / "docs/REPAIR_REPORT_2026-09-28.md")
