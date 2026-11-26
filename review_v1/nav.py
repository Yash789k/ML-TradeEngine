"""Stable navigation contract for the review console.

The dashboard, its tests and any external capture script (for example a
screen-recording driver) read the same identifiers. Changing an ``id`` here is a
breaking change for deep links such as
``http://127.0.0.1:8510/?view=walkthrough&case=r2-2026-10-01&symbol=NVDA&step=4``.
"""

from __future__ import annotations

from review_v1.paths import OUT, RECONSTRUCTED, RECORDED, SIMULATED, write_json

PORT = 8510
BASE_URL = f"http://127.0.0.1:{PORT}"

VIEWS = [
    dict(id="overview", label="1 · Overview", title="What this system is and what the results mean"),
    dict(id="data", label="2 · Data collected", title="Data collected and how it was validated"),
    dict(id="walkthrough", label="3 · Decision walkthrough", title="One logged decision, from inputs to outcome"),
    dict(id="classifiers", label="4 · Classifier statistics", title="Classifier performance against the class-frequency prior"),
    dict(id="regimes", label="5 · Market regimes (HMM)", title="Rolling hidden Markov model of market regimes"),
    dict(id="sentiment", label="6 · Sentiment link", title="Sentiment, model outputs and trade decisions"),
    dict(id="performance", label="7 · Trading performance", title="Trading performance against benchmarks"),
    dict(id="live", label="8 · Run the pipeline", title="Run the current pipeline on the latest completed session"),
    dict(id="report", label="9 · Report and evidence", title="Progress report, audit and reproduction"),
]
VIEW_IDS = [v["id"] for v in VIEWS]

# Walkthrough steps. ``provenance`` is the label of the step's primary facts;
# individual panels inside a step carry their own label.
STEPS = [
    dict(n=1, id="inputs", title="Market data collected", provenance=RECORDED),
    dict(n=2, id="features", title="Features computed from those bars", provenance=RECONSTRUCTED),
    dict(n=3, id="models", title="Classifier outputs and market regime", provenance=RECORDED),
    dict(n=4, id="decision", title="Decision rule and risk checks", provenance=RECORDED),
    dict(n=5, id="order", title="Order created", provenance=RECORDED),
    dict(n=6, id="fill", title="Fill and account after the next close", provenance=RECORDED),
    dict(n=7, id="outcome", title="Outcome and profit or loss", provenance=RECONSTRUCTED),
]

DEFAULT_CASE = "r2-2026-10-01"
DEFAULT_SYMBOL = "NVDA"

PROVENANCE_TEXT = {
    RECORDED: "RECORDED · written by the scheduled collector at the time",
    RECONSTRUCTED: "RECONSTRUCTED · recomputed now from inputs recorded at the time",
    SIMULATED: "SIMULATED · replay or demo run, never a pilot observation",
}


def contract() -> dict:
    return dict(
        schema=1,
        base_url=BASE_URL,
        port=PORT,
        launch="python -B -m streamlit run app_review.py --server.port 8510 "
        "--server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false",
        query_parameters=dict(
            view=VIEW_IDS,
            case="walkthrough case id, see data/review-v1/walkthrough/index.json",
            symbol="one of the 22 instruments",
            step=[s["n"] for s in STEPS],
        ),
        ready_marker=dict(
            selector="#review-ready",
            attributes=["data-view", "data-case", "data-symbol", "data-step"],
            meaning="Rendered last on every run; present once the selected view has finished drawing.",
        ),
        views=VIEWS,
        walkthrough_steps=STEPS,
        default_case=DEFAULT_CASE,
        default_symbol=DEFAULT_SYMBOL,
        provenance_labels=PROVENANCE_TEXT,
    )


def write_contract():
    return write_json(OUT / "ui_contract.json", contract())


if __name__ == "__main__":
    print(write_contract())
