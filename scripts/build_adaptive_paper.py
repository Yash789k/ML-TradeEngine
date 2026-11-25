"""Build the same complete manuscript as Markdown and a paginated PDF."""

import argparse
import csv
import html
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
)

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--release", type=Path, required=True)
a = p.parse_args()
release = a.release.resolve()
out = ROOT / "output/pdf"
out.mkdir(parents=True, exist_ok=True)
readiness = json.loads((release / "readiness.json").read_text())
diag = readiness["diagnostics"]
cls = list(csv.DictReader((release / "experiment/classification.csv").open()))
perf = list(csv.DictReader((release / "experiment/portfolio_metrics.csv").open()))
verification = (
    json.loads((release / "verification.json").read_text())
    if (release / "verification.json").exists()
    else {}
)
test_count = verification.get("tests_passed", "pending")
repro = ROOT / "data/releases/2026-09-28-reproduction/verification.json"
repro_text = (
    "A second run from archived source and snapshots reproduced every checked probability, account curve, metric, fold record and event record exactly."
    if repro.exists()
    else "A full second reproduction is pending; input and output checksum verification is available."
)

styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="BodyText2", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=8
    )
)
styles.add(
    ParagraphStyle(name="TableText", fontName="Helvetica", fontSize=8.2, leading=10.8)
)
styles.add(
    ParagraphStyle(
        name="Small2",
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#475569"),
        spaceAfter=8,
    )
)
styles["Title"].fontSize, styles["Title"].leading = 24, 29
styles["Title"].textColor = colors.HexColor("#0f172a")
styles["Heading1"].fontSize, styles["Heading1"].leading = 17, 22
styles["Heading2"].fontSize, styles["Heading2"].leading = 12, 16
story, md = [], []


def para(text, style="BodyText2"):
    story.append(Paragraph(html.escape(text), styles[style]))
    md.append(text + "\n")


def heading(text, level=1):
    story.append(
        Paragraph(html.escape(text), styles["Heading1" if level == 1 else "Heading2"])
    )
    md.append("#" * (level + 1) + " " + text + "\n")


def table(headers, rows, widths=None):
    cells = [
        [Paragraph(html.escape(str(x)), styles["TableText"]) for x in row]
        for row in [headers, *rows]
    ]
    t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
            ]
        )
    )
    story.extend([t, Spacer(1, 10)])
    md.append(
        "| "
        + " | ".join(headers)
        + " |\n| "
        + " | ".join("---" for _ in headers)
        + " |"
    )
    md.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    md.append("")


def page():
    story.append(PageBreak())


title = "Causal Rolling Models and Shared-Capital Execution: A Negative-Result Study of ML-TradeEngine"
story.append(Paragraph(title, styles["Title"]))
md.append("# " + title + "\n")
para(
    "ML-TradeEngine project | Technical research manuscript | 28 September 2026",
    "Small2",
)
heading("Abstract")
para(
    "We evaluate whether a rolling hybrid classifier adds predictive or economic value after repairing accounting, chronology and execution mismatches in an experimental trading system. A protocol registered before a new public-data download specifies 22 US equities and ETFs, execution-aligned five-session labels, purged rolling training, a fixed logistic/tree mixture, and one constrained $10,000 account per policy. The recent historical audit contains 54 trading sessions, with 48 label-complete sessions. Hybrid accuracy is 37.59%, compared with 35.89% for always-UP and 40.63% for a trailing class-frequency baseline. The hybrid portfolio returns -0.56% after modeled costs, versus +2.03% for the identically costed, risk-capped equal-weight benchmark. No candidate satisfies the declared statistical eligibility gate. The contribution is a reproducible implementation and a documented negative result, not a demonstrated trading advantage. Prospective profitability and broker execution remain unverified."
)
heading("1. Research question and contribution")
para(
    "The project has two distinct objectives: build a reliable simulated trader and establish a strategy whose performance warrants an empirical research claim. These objectives require different evidence. Accounting identities and recovery tests establish implementation properties; they cannot establish an economic edge. Conversely, an attractive return curve cannot validate an incorrect ledger or a future-informed model."
)
para(
    "We ask whether rolling adaptation improves prediction beyond simple class-frequency baselines, whether a fixed blend of statistical and trend signals improves net portfolio performance, and whether the exact account contract can be reused in daily local simulation. The experiment is deliberately bounded: three trading candidates, no search after viewing the recent audit, and no extrapolation from simulated returns to real-money profits."
)
para(
    "The earlier corrected classifiers achieved average accuracies of 30.26%-34.98%, below an always-UP baseline of 41.94%. Those findings motivated this new design. They remain part of the evidence record and are not replaced by a selectively reported favorable run. Repeated strategy searches can generate misleading apparent winners even without an intentional coding error [1]."
)

page()
heading("2. Data and chronological design")
para(
    "Yahoo Finance daily OHLCV data were downloaded on 27 September UTC / 28 September Hong Kong into a separate versioned snapshot. All 22 instruments pass completed-session, freshness, uniqueness, finite-value and OHLC-range checks through 25 September 2026. That Friday is the latest completed US session at retrieval. Ten years were requested, and the shared observation calendar is used for portfolio construction. Raw and split/dividend-adjusted snapshots are retained with retrieval metadata and SHA-256 hashes."
)
para(
    "The universe is AAPL, AMZN, BAC, DIA, GLD, GOOGL, GS, IWM, JPM, META, MSFT, NVDA, QQQ, SPY, TLT, TSLA, XLE, XLF, XLI, XLK, XLP and XLV. It mixes individual stocks and overlapping ETFs. It is a fixed, pre-existing project universe, not a survivorship-free sample of all listed securities. Exchange sessions and closing times follow the XNYS calendar; the standard core session ends at 16:00 Eastern [4]."
)
table(
    ["Stage", "Dates", "Use"],
    [
        [
            "Development",
            "2 Jan 2024 - 31 Dec 2025",
            "Historical diagnostics; already exploratory",
        ],
        [
            "Selection",
            "2 Jan - 10 Jul 2026",
            "Choose one of three fixed policies by net Sharpe",
        ],
        [
            "Recent audit",
            "13 Jul - 25 Sep 2026",
            "Registered before new download; 54 sessions",
        ],
        [
            "Prospective pilot",
            "From 29 Sep 2026",
            "63 completed sessions; none yet observed",
        ],
    ],
    [85, 145, 260],
)
para(
    "The recent audit begins after the latest observation in any previously inspected cache. It is a held-aside extension relative to the available project artifacts, but remains retrospective: the design was registered after calendar events had occurred. It is not described as an untouched prospective experiment. A model may refit during the audit using only labels whose endpoints are already in the past; the frozen adaptive algorithm, rather than one frozen set of coefficients, is being evaluated."
)
heading("2.1 Execution-aligned outcome")
para(
    "For a decision after close t, the first allowed fill is at close t+1. The prediction target is r(t) = Close(t+6) / Close(t+1) - 1. UP requires r(t) > 1%; DOWN requires r(t) < -1%; otherwise the class is FLAT. The final six rows have unavailable labels and are excluded from classification metrics. Consequently, 54 account sessions provide 48 labeled dates and 1,056 instrument-date classifications. The older screenshot used close-to-close labels starting at t, a different evaluation period, and mean per-ticker summaries; its percentages are not a like-for-like improvement benchmark."
)

page()
heading("3. Models, adaptation and portfolio execution")
para(
    "Seventeen scale-free features describe returns over 1, 5, 21, 63, 126 and 252 sessions; trailing volatility over 21 and 63 sessions; SPY-relative returns; price relative to 20-, 50- and 200-session means; overnight gap; high-low range; volume relative to its 20-session mean; and 12-minus-1-month momentum. All use information through the decision close. No revised macro data, unarchived news, full-sample HMM, or full-history feature selection enters this release."
)
para(
    "Each instrument refits every 21 observations using at most 1,008 eligible rows. Labels must end strictly before the refit decision. The final 126 eligible rows are held out for calibration; a six-session purge separates fitting from calibration. At least 504 fitting observations are required. Scaling is fit on the fitting partition only. Chronological splitting and explicit gaps address the availability issue represented by time-series cross-validation tools [2]."
)
para(
    "The experts are multinomial logistic regression (C=0.1) and histogram gradient boosting (100 iterations, seven leaves, minimum leaf 40, learning rate 0.05, L2=10). Class weights are not balanced. Their probabilities are averaged equally, then temperature-scaled using the calibration partition and a fixed choice set {0.75, 1, 1.5, 2, 3}. The temperature minimizes calibration log loss. The prior baseline is the Laplace-smoothed fitting class frequency. We use calibration and scoring concepts described in the scikit-learn documentation [3]; no future observations fit the scaler, experts or calibrator."
)
para(
    "The trend policy is long when price exceeds its trailing 200-session mean and 12-minus-1-month momentum is positive. The ML policy is long when its probability-weighted fitting class return exceeds 30 basis points. The adaptive hybrid is half trend and half ML only when calibration log loss improves on the prior by at least 0.01; otherwise it uses the trend rule. This gate is estimated on the same calibration block as temperature, so its internal performance can be optimistic. The outer audit measures the complete gate's behavior."
)
para(
    "Each active symbol initially receives 0.8/22 of capital. Targets are capped at 80% gross, 10% per symbol and 35% for the declared growth group. A trailing 63-session covariance matrix with 50% diagonal shrinkage reduces targets when annualized volatility exceeds 12%. No leverage is introduced. Weekly Friday decisions execute at the next observed close; the initial decision is also eligible. Non-rebalance sessions process pending fills and risk marks without generating routine weight-maintenance trades. Turnover is capped at 40% per session, and a 20% account drawdown requests liquidation subject to execution constraints."
)
para(
    "The ledger applies 10 basis points commission and 5 basis points slippage per side. Pending buys reserve cash, pending sales provide no buying power, and fees reduce cash. Equity equals cash plus marked positions; net realized profit includes entry and exit costs. Terminal positions remain marked. Cash earns zero; Sharpe uses a fixed 5% annual hurdle. Adjusted prices define total-return proxy units rather than executable broker shares. Daily local operation chains adjusted returns onto stored marks to remain invariant to uniform vendor rescaling."
)

page()
heading("4. Prediction results")
para(
    "Table 1 reports the recent audit. Accuracy measures the fraction of correct three-class labels. Balanced accuracy averages class recalls; its always-UP value is one third. Macro F1 gives each class equal weight. Log loss and the sum-of-classes multiclass Brier score evaluate probability quality; smaller values are better. Always-UP is nearly deterministic and therefore has very poor probability loss when its prediction is wrong. The probabilistic prior is the more informative comparison."
)
labels = {
    "linear": "Logistic",
    "tree": "Shallow tree",
    "hybrid": "Calibrated mixture",
    "prior": "Trailing prior",
    "always_up": "Always-UP",
}
table(
    ["Model", "Accuracy", "Balanced", "Macro F1", "Log loss", "Brier"],
    [
        [
            labels[r["model"]],
            f"{float(r['accuracy']):.2%}",
            f"{float(r['balanced_accuracy']):.2%}",
            f"{float(r['macro_f1']):.3f}",
            f"{float(r['log_loss']):.3f}",
            f"{float(r['brier']):.3f}",
        ]
        for r in cls
        if r["period"] == "audit"
    ],
    [123, 72, 75, 70, 75, 75],
)
para(
    "The hybrid is 1.70 percentage points above always-UP, but 3.03 points below the trailing prior. Its log loss of 1.088 also trails the prior's 1.076. The date-block 95% interval for the hybrid's accuracy improvement over always-UP is [-2.27, +5.30] percentage points. The interval contains zero. These diagnostic intervals are not adjusted for every model comparison and should not be promoted to discovery claims."
)
table(
    ["Period", "Hybrid accuracy", "Always-UP", "Trailing prior"],
    [
        [
            period.capitalize(),
            *[
                f"{float(next(r['accuracy'] for r in cls if r['period'] == period and r['model'] == model)):.2%}"
                for model in ["hybrid", "always_up", "prior"]
            ],
        ]
        for period in ["development", "selection", "audit"]
    ],
    [130, 120, 120, 120],
)
para(
    "The results illustrate why an absolute accuracy target is insufficient. Three-class frequency shifts change the difficulty of the task, while correct and incorrect predictions have unequal financial consequences. The new model is more causally aligned and adapts over time, but these properties do not establish predictive superiority. In this sample, a simpler rolling class-frequency estimate remains the strongest accuracy and probability benchmark."
)
heading("4.1 Dependence and uncertainty")
para(
    "Classification uncertainty resamples 10-date circular blocks, retaining all instruments within each sampled date to preserve cross-sectional dependence. Portfolio evidence uses signed heteroskedasticity-and-autocorrelation-consistent active-return statistics and 1,000 circular block bootstrap samples. The trading gate requires a positive lower active-return interval and a one-sided p-value below 0.05 after a Bonferroni factor of 333: the 330 earlier declared comparisons plus three new policies. This is a conservative bookkeeping family, not proof that all informal historical researcher choices have been accounted for. We do not claim to implement the probability-of-backtest-overfitting estimator in [1]."
)

page()
heading("5. Net portfolio results")
para(
    "The selection period chooses the trend rule (Sharpe 0.544) over the hybrid (0.509) and ML policy (0.017). This choice was fixed before inspecting audit outcomes. Each strategy has its own $10,000 shared-capital account running continuously from January 2024. The table reports period returns from those accounts, not averages of isolated ticker backtests."
)
names = {
    "trend": "Selected trend",
    "ml": "ML policy",
    "adaptive_hybrid": "Adaptive hybrid",
    "equal_weight": "Equal-weight benchmark",
    "cash": "Cash",
    "spy_reference": "SPY reference",
}
table(
    ["Policy", "2024-25", "Selection", "Audit", "Audit drawdown"],
    [
        [
            names[name],
            *[
                f"{float(next(r['total_return'] for r in perf if r['period'] == period and r['strategy'] == name)):+.2%}"
                for period in ["development", "selection", "audit"]
            ],
            f"{float(next(r['max_drawdown'] for r in perf if r['period'] == 'audit' and r['strategy'] == name)):.2%}",
        ]
        for name in names
    ],
    [150, 85, 85, 80, 90],
)
fig = out / "figures/audit_accounts.png"
story.append(Image(str(fig), width=490, height=204))
md.append("![Recent audit account curves](../output/pdf/figures/audit_accounts.png)\n")
para(
    "Figure 1. Audit-period curves rebased to $10,000 for visual comparison. Rebased curves are not separately restarted accounts. The equal-weight comparator uses the same cash, cost, risk-cap and execution rules. SPY is a frictionless adjusted-price reference, not an identically executed account.",
    "Small2",
)
para(
    "All three candidates have negative audit returns and adjusted active-return p-values of 1.0. For the selected trend policy, the 95% annualized active-mean interval is [-33.94%, +6.27%]; annualization of this short window produces substantial uncertainty. Doubling per-side costs changes audit returns to -0.65% (trend), -0.77% (ML) and -0.70% (hybrid). At four times costs they become -0.79%, -1.34% and -0.96%. No cost sensitivity case is used to select a new winner."
)

page()
heading("6. Implementation evidence and readiness")
para(
    f"The experiment contains {diag['folds']} rolling refits. Every recorded latest training-label endpoint precedes its decision. Five shared accounts produce {sum(diag['fills'].values()):,} simulated fills, with a maximum absolute accounting residual of ${diag['max_reconciliation_error']:.2e} and minimum cash of ${diag['minimum_cash']:,.2f}. These checks establish arithmetic consistency in the tested replays, not market realism or prospective profitability."
)
para(
    f"The offline verification record reports {test_count} passing tests. Regression coverage includes future-data perturbation, label availability, purged fitting/calibration boundaries, model serialization and tampering, exact feature contracts, pending-cash limits, gap rejection, scheduled hold sessions, drawdown exits, retries, restart recovery, missed-session rejection and account reconciliation. A real-data commissioning run loaded all fresh model bundles and prepared the daily decision record. Before the reserved pilot start it creates no prospective fills."
)
para(repro_text)
table(
    ["Readiness decision", "Outcome", "Meaning"],
    [
        [
            "Software contract",
            readiness["software_contract"].replace("_", " "),
            "Scope is the tested local simulation",
        ],
        [
            "Current data and models",
            "Approved at release",
            "22 complete instruments through 25 Sep",
        ],
        ["Historical trading edge", "Not approved", "Audit eligibility tests fail"],
        [
            "Prospective trading evidence",
            "Not approved",
            "Zero observed pilot sessions",
        ],
        [
            "Profitable trader / combined",
            "Not approved",
            "No defensible basis for approval",
        ],
        [
            "Broker execution",
            "Not approved",
            "No external orders or adapter certification",
        ],
        [
            "Research manuscript",
            "Completed technical report",
            "Negative results; not peer-reviewed acceptance",
        ],
    ],
    [150, 130, 210],
)
para(
    "The production-eligible local allocation is cash. A separately labeled research shadow may exercise the selected trend policy prospectively without implying qualification. The daily command verifies archived artifacts, retrieves current data, refreshes models on their fixed schedule, records targets and model hashes, and processes the same ledger contract as research. It refuses missing or stale inputs and refuses to backfill missed pilot sessions. Model updates are separate immutable versions. There are no broker, email or messaging calls in this release path."
)
heading("6.1 Reproducibility package")
para(
    "The release directory data/releases/2026-09-28-adaptive contains the registered protocol, raw and adjusted market snapshots, dependency versions, archived training source, model manifests, fold cutoffs, probabilities, target panels, SQLite account states, full event histories, classification/portfolio tables and checksums. The daily CLI is trader.py; the current console is app.py. Reproduce the numerical study with scripts/reproduce_adaptive.py using a new output directory. The package records actual dependency versions; a clean-environment portability run across different library versions has not been claimed."
)

page()
heading("7. Limitations and interpretation")
para(
    "The central result is negative: this experiment does not establish a useful predictive advantage or a profitable strategy after costs. It does not prove that all ML strategies must fail. The models, universe, horizons and training budgets are limited, while the recent audit is only 54 trading sessions. A positive future result would still need uncertainty analysis and independent confirmation."
)
para(
    "The fixed universe includes surviving stocks and ETFs selected before this experiment; overlapping funds introduce concentrated economic exposures beyond the explicit growth group. Portfolio covariance is estimated from a short trailing window. The data provider's adjusted history is a current-vintage reconstruction rather than an immutable historical feed. Scale-free features and stored snapshots improve auditability, but cannot prove that every observation was available in exactly that form at its historical timestamp."
)
para(
    "The execution model assumes next-close fills, fixed proportional costs and fractional total-return proxy units. It does not model closing-auction participation, spread dynamics, volume impact, taxes, borrow, market halts or cash corporate-action processing in a broker account. A drawdown threshold requests reduction after observation; it does not guarantee a maximum realized loss. Drift can temporarily move marked exposure beyond a target cap. External execution requires a dedicated adapter, verified timing and corporate-action reconciliation."
)
para(
    "The prospective 63-session pilot is reserved from 29 September 2026 and has not occurred. No generated report, statistical threshold or test count can substitute for observing it. If the algorithm is changed after seeing pilot results, it becomes a new registered experiment. The current manuscript is complete as an implementation-and-negative-result technical study. It does not support a title, abstract or conclusion claiming consistent profits, verified alpha, novel market prediction, or production broker readiness."
)
heading("References")
refs = [
    "[1] Bailey, D. H., Borwein, J. M., Lopez de Prado, M., and Zhu, Q. J. The Probability of Backtest Overfitting. Author manuscript. https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf",
    "[2] scikit-learn. TimeSeriesSplit documentation, including chronological folds and gap. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html",
    "[3] scikit-learn. Model evaluation and probability calibration documentation. https://scikit-learn.org/stable/modules/model_evaluation.html and https://scikit-learn.org/stable/modules/calibration.html",
    "[4] New York Stock Exchange. Trading information and core-session hours. https://www.nyse.com/trade/trading-information",
    "[5] ML-TradeEngine. Registered adaptive protocol, 28 September 2026; immutable run artifacts and source manifests. docs/ADAPTIVE_PROTOCOL_2026-09-28.md; data/releases/2026-09-28-adaptive/.",
]
for r in refs:
    para(r, "Small2")

(ROOT / "docs/ADAPTIVE_RESEARCH_PAPER.md").write_text("\n".join(md))


def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
    canvas.line(52, 40, 560, 40)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(
        52, 27, "ML-TradeEngine | Reproducible research | 28 September 2026"
    )
    canvas.drawRightString(560, 27, str(doc.page))


doc = SimpleDocTemplate(
    str(out / "ML_TradeEngine_Research.pdf"),
    pagesize=(612, 792),
    rightMargin=56,
    leftMargin=56,
    topMargin=46,
    bottomMargin=54,
    title=title,
    author="ML-TradeEngine project",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(out / "ML_TradeEngine_Research.pdf")
