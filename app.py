"""Current, evidence-backed local trading and research console."""

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="ML TradeEngine | Readiness", page_icon="📊", layout="wide"
)
st.title("ML TradeEngine")
st.caption("Local paper trading and reproducible strategy research")
releases = sorted((ROOT / "data/releases").glob("*/readiness.json"), reverse=True)
if not releases:
    st.info(
        "No completed release is available. Run the registered training pipeline first."
    )
    st.stop()
release = st.sidebar.selectbox(
    "Research release", [p.parent for p in releases], format_func=lambda p: p.name
)
workspace = ROOT / "data/paper/adaptive-v1"
ready = json.loads((release / "readiness.json").read_text())
st.sidebar.caption("Execution: local simulation only")
st.sidebar.caption("Capital: $10,000 · Long only · No leverage")
if st.sidebar.button("Refresh data and run daily process", type="primary"):
    with st.spinner("Validating data, models and account state…"):
        run = subprocess.run(
            [
                sys.executable,
                str(ROOT / "trader.py"),
                "daily",
                "--release",
                str(release),
                "--workspace",
                str(workspace),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
    if run.returncode:
        st.error(run.stderr[-4000:] or run.stdout[-4000:])
    else:
        st.success("Daily record saved.")
if (workspace / "last_error.json").exists():
    st.sidebar.warning("A process error was recorded. See the operations tab.")

status, account_tab, research, signals_tab, paper = st.tabs(
    ["Readiness", "Account", "Research", "Signals & operations", "Paper"]
)
with status:
    st.subheader("Decisions supported by evidence")
    if ready["combined_readiness"] == "approved":
        st.success("Combined readiness approved")
    else:
        st.warning("Combined readiness is not approved. " + ready["reason"] + ".")
    a, b, c, d = st.columns(4)
    a.metric("Latest training data", str(pd.Timestamp(ready["asof"]).date()))
    b.metric("Instruments", len(ready["symbols"]))
    c.metric("Eligible allocation", "100% cash")
    d.metric("Pilot requirement", "63 sessions")
    decisions = [
        ("Software contracts", ready["software_contract"]),
        ("Release market data", ready["market_data"]),
        ("Model bundles", ready["model_bundles"]),
        (
            "Historical trading edge",
            "approved" if ready["historical_edge"] else "not approved",
        ),
        ("Consistent profitability", "not approved"),
        ("Broker execution", "not connected"),
    ]
    st.dataframe(
        pd.DataFrame(decisions, columns=["Decision", "Status"]),
        hide_index=True,
        width="stretch",
    )
    st.write(
        "The research shadow can collect evidence for the selected candidate. The deployment account holds cash until the required research and prospective checks pass."
    )
    st.caption(
        "A completed software test or a positive historical return does not establish future profitability."
    )

latest = (
    json.loads((workspace / "latest.json").read_text())
    if (workspace / "latest.json").exists()
    else None
)
with account_tab:
    st.subheader("One account, shared cash")
    if latest:
        deployed = latest["deployment"]
        a, b, c = st.columns(3)
        a.metric("Deployment equity", f"${deployed['equity']:,.2f}")
        b.metric("Cash", f"${deployed['cash']:,.2f}")
        c.metric("Run mode", latest["mode"].replace("_", " "))
        st.caption(
            f"Observation: {latest['session']} · Decision prepared: {latest['prepared_at']}"
        )
        if "shadow" in latest:
            st.metric("Research shadow equity", f"${latest['shadow']['equity']:,.2f}")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Instrument": s, "Proxy units": q}
                    for s, q in deployed["positions"].items()
                    if q
                ]
            ),
            hide_index=True,
        )
    else:
        st.info("Run the daily process to produce the first decision record.")
    st.write(
        "Pending buys reserve cash. Signals set target holdings; repeated signals do not stack orders. Orders fill at the next completed close in the local model."
    )
    st.caption(
        "Simulation units track adjusted total returns, including the vendor's split/dividend adjustments. They are not broker share quantities."
    )

with research:
    st.subheader("The registered comparison")
    period = st.selectbox("Period", ["audit", "selection", "development"])
    st.caption(
        "Audit: 13 July–25 September 2026. Selection: 2 January–10 July 2026. Development: 2024–2025. All are historical."
    )
    metrics = pd.read_csv(release / "experiment/portfolio_metrics.csv")
    st.dataframe(
        metrics.loc[
            metrics.period == period,
            [
                "strategy",
                "total_return",
                "sharpe_ratio",
                "max_drawdown",
                "active_mean_ci_low",
                "active_mean_ci_high",
                "bonferroni_p",
            ],
        ],
        hide_index=True,
        width="stretch",
    )
    r = pd.read_parquet(release / "experiment/returns.parquet")
    bounds = {
        "audit": ("2026-07-13", "2026-09-25"),
        "selection": ("2026-01-02", "2026-07-10"),
        "development": ("2024-01-02", "2025-12-31"),
    }
    r = r.loc[bounds[period][0] : bounds[period][1]]
    eq = 10000 * (1 + r).cumprod()
    st.plotly_chart(
        px.line(
            eq, labels={"value": "Rebased account value ($)", "variable": "Policy"}
        ),
        width="stretch",
    )
    st.caption(
        "Period curves are rebased to $10,000 for comparison; the underlying accounts run continuously from January 2024."
    )
    st.subheader("Does the model predict better than a simple guess?")
    cls = pd.read_csv(release / "experiment/classification.csv")
    st.dataframe(
        cls.loc[cls.period == period].drop(columns="period"),
        hide_index=True,
        width="stretch",
    )
    st.write(
        "Always-UP predicts UP every time. The prior predicts the most common class in the trailing training sample. Balanced accuracy gives each class equal weight; log loss measures probability quality, and lower is better. Prediction accuracy and trading profitability are different tests."
    )

with signals_tab:
    if latest:
        st.dataframe(pd.DataFrame(latest["signals"]), hide_index=True, width="stretch")
        st.write("Shadow target weights")
        st.bar_chart(pd.Series(latest["shadow_weights"]))
    if (workspace / "last_error.json").exists():
        st.json(json.loads((workspace / "last_error.json").read_text()))
    st.caption(
        "Failed input validation stops the batch. Missed prospective sessions cannot be silently backfilled."
    )
    st.code("python trader.py daily\npython trader.py verify", language="bash")

with paper:
    st.subheader("Research manuscript and reproducibility")
    path = ROOT / "docs/ADAPTIVE_RESEARCH_PAPER.md"
    pdf = ROOT / "output/pdf/ML_TradeEngine_Research.pdf"
    if pdf.exists():
        st.download_button(
            "Download manuscript (PDF)", pdf.read_bytes(), pdf.name, "application/pdf"
        )
    if path.exists():
        st.download_button(
            "Download manuscript (Markdown)",
            path.read_text(),
            path.name,
            "text/markdown",
        )
        st.markdown(path.read_text())
    else:
        st.info("The manuscript is being generated from this release's results.")
    with st.expander("Registered experiment protocol"):
        st.markdown((release / "protocol.md").read_text())
