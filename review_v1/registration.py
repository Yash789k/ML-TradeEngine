"""Registration written before any new evaluation in this review.

``python -m review_v1.registration`` writes ``data/review-v1/registration.json``
once and refuses to overwrite it. Every evaluation output embeds the hash of this
file; ``require()`` stops a stage if the registration is missing or has changed.
"""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from review_v1.paths import ADAPTIVE_RELEASE, OUT, R7_RELEASE, ROOT, read_json, rel, sha256, write_json
from review_v1.regime import RegimeConfig

PATH = OUT / "registration.json"
SYMBOLS = [
    "AAPL", "AMZN", "BAC", "DIA", "GLD", "GOOGL", "GS", "IWM", "JPM", "META", "MSFT",
    "NVDA", "QQQ", "SPY", "TLT", "TSLA", "XLE", "XLF", "XLI", "XLK", "XLP", "XLV",
]
HISTORY = ("2024-01-02", "2026-09-25")
POST_RELEASE_START = "2026-09-29"
PERIODS = {
    "development": ("2024-01-02", "2025-12-31"),
    "selection": ("2026-01-02", "2026-07-10"),
    "audit": ("2026-07-13", "2026-09-25"),
}
SANDBOX_ARMS = ["cash", "equal_weight", "trend", "trend_regime", "equal_weight_regime"]
SANDBOX_CONTRASTS = [("trend_regime", "trend"), ("equal_weight_regime", "equal_weight")]
BLOCK = 10
REPS = 2000


def build() -> dict:
    # The decision contract is bound here. Review code is hashed in the build
    # manifest instead, so a later bug fix there is visible without implying
    # that the registered analysis changed.
    sources = ["runtime_v2/policy.py", "runtime_v2/model.py", "runtime_v2/risk.py", "runtime_v2/account.py"]
    return dict(
        id="REVIEW-V1-20261125",
        registered_at=pd.Timestamp.now(tz="UTC").isoformat(),
        role=(
            "Progress-review diagnostics plus one bounded sandbox portfolio comparison. No model is "
            "tuned, no winner is promoted to the pilot, and nothing here is prospective evidence."
        ),
        symbols=SYMBOLS,
        inputs=dict(
            adaptive_release_manifest_sha256=sha256(ADAPTIVE_RELEASE / "manifest.json"),
            outcome_recovery_protocol_sha256=sha256(R7_RELEASE / "protocol.json"),
            source_sha256={p: sha256(ROOT / p) for p in sources},
        ),
        exposure=dict(
            history=dict(
                dates=HISTORY,
                status="previously inspected; exploratory only",
                note=(
                    "Used by the sealed adaptive release, research-v2 and the 30 October controller "
                    "screen. The trend rule was the selected shadow on this history."
                ),
            ),
            post_release=dict(
                start=POST_RELEASE_START,
                end="latest completed XNYS session when the build runs",
                status="short and partly seen; descriptive only; not an untouched holdout",
                note=(
                    "Five decision dates in this window (US 1, 2, 14, 15, 22 October) and outcomes for "
                    "four of them were already reported for the trend shadow and the hybrid forecast. "
                    "Before this registration no HMM value, sandbox arm or continuous replay had been "
                    "computed on it, and no price after 28 October had been inspected."
                ),
            ),
        ),
        label_contract=dict(
            decision="information cutoff at the XNYS close of session t",
            entry="close of session t+1 (first permissible execution)",
            exit="close of session t+6",
            forward_return="close[t+6] / close[t+1] - 1 on dividend/split-adjusted prices",
            classes=dict(DOWN="return < -1%", FLAT="-1% <= return <= +1%", UP="return > +1%"),
            note="This is the repository's verified contract, not the proposal's next-open, cost-and-volatility label.",
        ),
        regime=dict(
            config=asdict(RegimeConfig()),
            series="SPY adjusted close",
            observation="daily log return and log 21-session realised volatility",
            label_rule="turbulent when the filtered P(turbulent) >= 0.5 at the decision close, otherwise calm",
            role=(
                "Diagnostic for every recorded decision (it did not exist then). A decision input only "
                "in the two sandbox overlay arms below."
            ),
        ),
        classifier_statistics=dict(
            inputs="saved out-of-fold forecasts in the sealed adaptive release; no model is fitted",
            models=["linear", "tree", "raw_mix", "hybrid", "prior"],
            endpoints=dict(primary="log_loss", secondary=["accuracy", "balanced_accuracy", "macro_f1", "brier"]),
            regime_contrasts=[
                dict(model="hybrid", comparator="prior", regime="calm"),
                dict(model="hybrid", comparator="prior", regime="turbulent"),
            ],
            inference=f"daily cross-sectional mean difference; {REPS} circular date blocks of length {BLOCK}; Holm over the two regime contrasts",
            status="exploratory: every date was inspected before this registration",
        ),
        sandbox=dict(
            question=(
                "Does scaling the weekly targets by one minus the HMM turbulence probability improve the "
                "risk-adjusted result of its parent portfolio after costs?"
            ),
            arms=dict(
                cash="hold cash",
                equal_weight="0.8/22 per instrument, then the shared risk constraints (benchmark)",
                trend="0.8/22 per instrument whose price is above its 200-session average with positive 12-1 momentum, then the shared risk constraints (the registered pilot shadow)",
                trend_regime="trend target multiplied by (1 - P(turbulent)) at the decision close",
                equal_weight_regime="equal-weight target multiplied by (1 - P(turbulent)) at the decision close",
            ),
            targets=dict(
                history="sealed trend_targets.parquet and equal_weight_targets.parquet from the adaptive release, reused unchanged",
                post_release="recomputed with runtime_v2.policy and runtime_v2.risk from a fresh public snapshot",
            ),
            execution=dict(
                ledger="runtime_v2.account.Account through ExecutionSession.on_close (the pilot ledger)",
                capital=10000,
                fills="next observed close after the decision",
                rebalance="first session and the last XNYS session of each ISO week",
                costs="10 bp commission plus 5 bp slippage on traded notional, both sides",
                limits="80% gross, 10% per instrument, 35% growth group, 40% turnover per rebalance, 3% cash reservation buffer, 20% drawdown stop",
                cash_return="zero",
            ),
            periods=dict(history=HISTORY, post_release=[POST_RELEASE_START, "latest completed session"], subperiods=PERIODS),
            primary_endpoint="difference in annualised Sharpe ratio (zero risk-free rate), overlay arm minus its parent, on the history period",
            contrasts=[dict(arm=a, parent=p) for a, p in SANDBOX_CONTRASTS],
            inference=f"paired circular date-block bootstrap, block {BLOCK}, {REPS} replicates, two-sided; Holm over the two contrasts",
            secondary=[
                "net return", "CAGR", "Sharpe with block interval", "maximum drawdown", "mean daily return difference",
                "turnover", "fees", "fills", "mean gross exposure", "probabilistic and deflated Sharpe ratio",
            ],
            decision_rule=(
                "An overlay is 'supported on this history' only if its Sharpe difference is positive with Holm p < 0.05. "
                "Otherwise it is 'not established'. Neither outcome changes the pilot."
            ),
            post_release_rule="report every arm descriptively; no inference and no Sharpe interpretation for fewer than 63 sessions",
            deflated_sharpe=dict(
                trials=["trend", "ml", "adaptive_hybrid", "equal_weight", "trend_regime", "equal_weight_regime"],
                note=(
                    "Counts only strategies whose continuous daily returns on the history period are available here. "
                    "Earlier prototype backtests and the research-v2 block-reset policies are not counted, so the true "
                    "number of trials is larger and the deflated value is an upper bound on confidence."
                ),
            ),
            budget="five arms, two periods, no parameter search; any further variant needs a new registration",
        ),
        replay=dict(
            kind="simulated continuous replay of the pilot contract; never a pilot observation",
            method="runtime_v2.policy.replay, revealed one session at a time, fresh isolated accounts",
            start=POST_RELEASE_START,
            models="carried 25 September models; scheduled refits every 21 sessions from the 29 September anchor",
            classification="hybrid and training-frequency prior scored on matured labels only; descriptive; block length 5",
        ),
        sentiment=dict(
            kind="diagnostic join; no model is retrained and no decision used sentiment",
            decision_dates="FinBERT features for the 22 instruments at the five recorded decision dates, from Alpaca headlines available before each cutoff",
            history="AAPL, NVDA and SPY cached headlines joined to saved forecasts from 2 January 2024 to the end of each cache",
            statistics="Spearman rank correlation with the model score P(UP) - P(DOWN), the trend flag and the realised forward return; descriptive",
            optional_ablation=dict(
                description="price-only versus price-plus-sentiment hybrid for NVDA and SPY, runtime configuration, refit every 21 sessions",
                endpoint="paired daily log-loss difference; block 10",
                status_if_skipped="recorded as not run",
            ),
        ),
        walkthrough=dict(
            case_rule="earliest recorded decision whose orders were filled at the next close by the scheduled collector: segment r2, US session 1 October 2026",
            symbol_rule=(
                "NVDA, chosen from decision-time inputs only: it has cached news history, the trend rule bought it, "
                "and the classifier's most likely class was DOWN. Chosen before its outcome was looked at."
            ),
        ),
    )


def register() -> dict:
    if PATH.exists():
        raise FileExistsError(f"{rel(PATH)} already exists; a registration is never overwritten")
    write_json(PATH, build())
    return dict(path=rel(PATH), sha256=sha256(PATH))


AMENDMENTS = OUT / "registration_amendments.json"


def amend(identifier: str, made_before: str, change: str, reason: str) -> dict:
    """Append a dated amendment. The original registration file is never edited."""
    rows = read_json(AMENDMENTS, [])
    if any(r["id"] == identifier for r in rows):
        raise FileExistsError(f"Amendment {identifier} already recorded")
    row = dict(
        id=identifier,
        recorded_at=pd.Timestamp.now(tz="UTC").isoformat(),
        registration_sha256=sha256(PATH),
        made_before=made_before,
        change=change,
        reason=reason,
    )
    write_json(AMENDMENTS, rows + [row])
    return row


def require() -> dict:
    """Return the registration and its hash, or stop the calling stage."""
    if not PATH.exists():
        raise RuntimeError("No registration: run `python -m review_v1.registration` before any evaluation")
    return dict(registration=read_json(PATH), sha256=sha256(PATH), path=rel(PATH), amendments=read_json(AMENDMENTS, []))


if __name__ == "__main__":
    print(register())
