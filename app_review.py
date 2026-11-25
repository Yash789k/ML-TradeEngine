"""Progress-review console. Read-only over sealed releases and recorded receipts.

Launch:
    python -B -m streamlit run app_review.py --server.port 8510 --server.address 127.0.0.1

Deep links: ?view=<id>&case=<case id>&symbol=<ticker>&step=<1-7>, see review_v1/nav.py.

Every number on screen is read from ``data/review-v1/`` (built by
``python -m review_v1.build all``). The only thing this file computes is layout.
The one action it can take is the demo run on the "Run the pipeline" page, which
writes to an isolated workspace and never touches a pilot account.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from review_v1 import audit, explain, nav
from review_v1.paths import CLASSES, OUT, RECONSTRUCTED, RECORDED, ROOT, SIMULATED, read_json

COLORS = {RECORDED: "#0f766e", RECONSTRUCTED: "#b45309", SIMULATED: "#6d28d9"}
ARM_COLORS = {
    "equal_weight": "#1f2937", "trend": "#0f766e", "trend_regime": "#b45309", "equal_weight_regime": "#6d28d9", "cash": "#9ca3af",
}
STATUS_COLORS = {"works": "#0f766e", "diagnostic": "#2563eb", "unvalidated": "#b45309", "missing": "#b91c1c"}
CSS = """
<style>
.rv-badge {display:inline-block; padding:2px 10px; border-radius:999px; color:#fff; font-size:0.78rem; font-weight:700; letter-spacing:.04em; vertical-align:middle;}
.rv-note {color:#4b5563; font-size:0.86rem; margin-left:6px;}
.rv-headline {font-size:1.35rem; font-weight:650; line-height:1.35; margin:0 0 .4rem 0;}
.rv-caption {color:#374151; font-size:1.0rem; line-height:1.5; margin:0 0 .8rem 0;}
.rv-pill {display:inline-block; padding:1px 8px; border-radius:6px; color:#fff; font-size:0.78rem; font-weight:600;}
.rv-action {font-size:2.0rem; font-weight:800; letter-spacing:.02em;}
[data-testid="stMainBlockContainer"] {padding-left:2.5rem; padding-right:2.5rem; max-width:1320px;}
</style>
"""


# --------------------------------------------------------------------------
# Loading (cached on file modification time, so a rebuild shows up on refresh)
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _json(path: str, stamp: float):
    return read_json(path)


@st.cache_data(show_spinner=False)
def _csv(path: str, stamp: float):
    return pd.read_csv(path)


def J(name: str):
    path = OUT / name
    return _json(str(path), path.stat().st_mtime) if path.exists() else None


def T(name: str):
    path = OUT / name
    return _csv(str(path), path.stat().st_mtime) if path.exists() else None


# --------------------------------------------------------------------------
# Small presentation helpers
# --------------------------------------------------------------------------

def badge(kind: str, note: str = ""):
    label = nav.PROVENANCE_TEXT[kind].split(" · ")
    st.html(
        f'<span class="rv-badge" data-provenance="{kind}" style="background:{COLORS[kind]}">{label[0]}</span>'
        f'<span class="rv-note">{note or label[1]}</span>'
    )


def headline(view: str, caption: str | None = None):
    """Title and one-paragraph summary of a view. ``caption`` replaces the saved summary where the page has fresher facts."""
    narration = J("narration.json")
    title = next(v["title"] for v in nav.VIEWS if v["id"] == view)
    st.header(title)
    if narration and view in narration["views"]:
        item = narration["views"][view]
        st.html(f'<p class="rv-headline">{item["headline"]}</p><p class="rv-caption">{caption or item["caption"]}</p>')


def missing(name: str):
    st.info(f"This section is being built: `{name}` has not been generated yet. Run `python -m review_v1.build all`.")


def table(frame: pd.DataFrame, formats: dict | None = None, height=None, percent: dict | None = None):
    """Show a table. ``formats`` are printf formats; ``percent`` columns hold fractions and are shown as percentages."""
    config = {}
    for column, fmt in (formats or {}).items():
        if column in frame.columns:
            config[column] = st.column_config.NumberColumn(format=fmt)
    if percent:
        frame = frame.copy()
        for column, fmt in percent.items():
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce") * 100
                config[column] = st.column_config.NumberColumn(format=fmt)
    kwargs = dict(hide_index=True, column_config=config, width="stretch")
    if height:
        kwargs["height"] = height
    st.dataframe(frame, **kwargs)


def chart(spec):
    st.altair_chart(spec, width="stretch")


def money(value) -> str:
    return f"${value:,.2f}"


def usd(value) -> str:
    """A dollar amount for Markdown text, where an unescaped pair of dollar signs would be read as a formula."""
    return f"\\${value:,.2f}"


def pct(value, digits=1, signed=True) -> str:
    return "n/a" if value is None or pd.isna(value) else (f"{value:+.{digits}%}" if signed else f"{value:.{digits}%}")


def prose(text) -> str:
    """Text from a saved file for Markdown display: dollar signs are amounts here, never formulas."""
    return str(text).replace("$", "\\$")


def prose_table(columns: list, rows: list):
    """A table of sentences. Markdown wraps long text; a data grid would cut it off."""
    def cell(value):
        return prose(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)] + ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    st.markdown("\n".join(lines))


def tiny(value) -> str:
    """A numerical difference for prose: exactly zero reads as 'none', anything else in scientific notation."""
    return "none" if value == 0 else f"{value:.0e}"


def when(value, fmt="%-d %b %Y %H:%M UTC") -> str:
    return pd.Timestamp(value).strftime(fmt)


# --------------------------------------------------------------------------
# Navigation state and the ready marker
# --------------------------------------------------------------------------

def _query(name, allowed, default):
    value = st.query_params.get(name, default)
    return value if value in allowed else default


def select_view() -> str:
    labels = {v["id"]: v["label"] for v in nav.VIEWS}
    if "view" not in st.session_state:
        st.session_state["view"] = _query("view", nav.VIEW_IDS, "overview")
    st.sidebar.title("ML TradeEngine")
    st.sidebar.caption("Monthly progress review · November 2026")
    view = st.sidebar.radio("Section", nav.VIEW_IDS, format_func=labels.get, key="view")
    st.sidebar.divider()
    st.sidebar.markdown("**How to read the labels**")
    for kind in (RECORDED, RECONSTRUCTED, SIMULATED):
        label, text = nav.PROVENANCE_TEXT[kind].split(" · ")
        st.sidebar.html(f'<span class="rv-badge" style="background:{COLORS[kind]}">{label}</span><div class="rv-note" style="margin:2px 0 8px 0">{text}</div>')
    st.sidebar.divider()
    st.sidebar.caption("Local paper trading only. External orders disabled. Nothing here is investment advice or evidence of profitability.")
    manifest = J("manifest.json")
    if manifest:
        st.sidebar.caption(f"Evidence built {when(manifest['built_at'])}")
    return view


def ready_marker(view, **attributes):
    extra = "".join(f' data-{k}="{v}"' for k, v in attributes.items() if v is not None)
    st.html(f'<div id="review-ready" data-view="{view}"{extra}></div>')


# --------------------------------------------------------------------------
# 1 · Overview
# --------------------------------------------------------------------------

PIPELINE_CSS = """
<style>
.rv-flow {display:flex; align-items:stretch; gap:0; margin:.2rem 0 .6rem 0;}
.rv-box {flex:1 1 0; border:2px solid #0f766e; background:#f0fdfa; border-radius:10px; padding:10px 12px; min-width:0;}
.rv-box.input {border-color:#9ca3af; background:#f9fafb;}
.rv-box.side {border:2px dashed #b45309; background:#fffbeb;}
.rv-box.off {border:2px dashed #b91c1c; background:#fef2f2;}
.rv-box b {display:block; font-size:.95rem; margin-bottom:2px;}
.rv-box span {font-size:.82rem; color:#374151; line-height:1.3; display:block;}
.rv-box em {font-style:normal; font-size:.72rem; font-weight:700; letter-spacing:.05em; text-transform:uppercase; display:block; margin-top:6px;}
.rv-arrow {flex:0 0 26px; display:flex; align-items:center; justify-content:center; color:#6b7280; font-size:1.2rem;}
.rv-row-title {font-size:.8rem; font-weight:700; color:#6b7280; letter-spacing:.05em; text-transform:uppercase; margin:.5rem 0 .25rem 0;}
</style>
"""


def pipeline_diagram(inventory: dict) -> str:
    """The decision path as it actually is: what decides, and what is only recorded."""
    count, features = inventory["universe"]["count"], inventory.get("models", {}).get("features", 17)
    main = [
        ("input", "1 · Daily bars", f"{count} instruments, public download, hashed snapshot", "input"),
        ("input", "2 · Features", f"{features} price and volume features per instrument", "input"),
        ("", "3 · Trend rule", "price above its 200-session average and 12-month momentum positive", "decides"),
        ("", "4 · Risk limits", "10% per instrument, 35% growth group, 80% gross, 12% volatility target, 40% turnover", "decides"),
        ("", "5 · Paper order", "fills at the next close, 15 bp cost, local ledger only", "executes"),
        ("", "6 · Outcome", "account marked each session; forecasts scored after five sessions", "evaluated"),
    ]
    side = [
        ("side", "Classifier", "P(DOWN), P(FLAT), P(UP) from the same features. Saved with every decision and scored against the prior.", "recorded, not used"),
        ("side", "HMM market regime", "Calm or turbulent, from SPY. Written for this review.", "diagnostic"),
        ("off", "News sentiment", "FinBERT scores of headlines. A pipeline exists; it has no input into the features or the rule.", "not connected"),
    ]
    box = lambda kind, title, text, role: f'<div class="rv-box {kind}"><b>{title}</b><span>{text}</span><em>{role}</em></div>'
    arrow = '<div class="rv-arrow">→</div>'
    return (
        PIPELINE_CSS
        + '<div class="rv-row-title">What decides and executes</div><div class="rv-flow">' + arrow.join(box(*b) for b in main) + "</div>"
        + '<div class="rv-row-title">Computed alongside, with no influence on any recorded order</div><div class="rv-flow" style="gap:14px">' + "".join(box(*b) for b in side) + "</div>"
    )


def view_overview():
    headline("overview")
    inventory, manifest = J("inventory.json"), J("manifest.json")
    if not inventory or not manifest:
        return missing("inventory.json")
    pilot, counts = inventory["pilot"], manifest["audit_summary"]
    a, b, c, d = st.columns(4)
    a.metric("Daily bars collected", f"{inventory['market_history']['rows']:,}", help=f"{inventory['universe']['count']} instruments, {inventory['market_history']['first']} to {inventory['market_history']['last']}")
    b.metric("Decision sessions recorded", f"{pilot.get('decisions', 0)} of {pilot.get('scheduled_decision_sessions', 0)}",
             help="Scheduled sessions that produced a valid receipt: US sessions " + ", ".join(pd.Timestamp(s).strftime("%-d %b") for s in pilot.get("decision_sessions", [])) + " 2026")
    c.metric("Recorded paper fills", pilot.get("fills", 0), help="Local paper ledger. No external order was ever sent.")
    d.metric("Verification checks passed", f"{sum(x['passed'] for x in manifest['checks'])} of {len(manifest['checks'])}", help="Listed in section 9")

    st.subheader("How a decision is made")
    st.html(pipeline_diagram(inventory))

    st.subheader("Where the project stands")
    cols = st.columns(4)
    text = {"works": "Works, verified", "diagnostic": "Diagnostic only", "unvalidated": "Exists, not validated", "missing": "Missing"}
    for column, status in zip(cols, text, strict=True):
        column.html(f'<span class="rv-pill" style="background:{STATUS_COLORS[status]}">{text[status]}</span>')
        column.metric(text[status], counts[status], label_visibility="collapsed")
        for row in [r for r in manifest["audit"] if r["status"] == status]:
            column.caption(f"• {row['component']}")
    st.caption("Full audit with evidence and how to verify each row: section 9.")

    st.subheader("What the classifier result means")
    st.markdown(f"> {explain.SENTENCE}")
    for title, body in explain.sections():
        with st.expander(title, expanded=title in ("What this tells us", "What this does not tell us")):
            st.markdown(body)

    st.subheader("How to use this dashboard")
    st.markdown(
        "- **Section 3** replays one recorded decision in seven steps, from the bars that were downloaded to what the position did afterwards.\n"
        "- **Sections 4 to 6** show how the forecasts, the regime model and sentiment are evaluated, and which of them matter to a decision.\n"
        "- **Section 7** reports trading results against a benchmark, with costs and sample-size limits stated.\n"
        "- **Section 8** runs the current pipeline on the latest completed session."
    )


# --------------------------------------------------------------------------
# 2 · Data collected
# --------------------------------------------------------------------------

def view_data():
    headline("data")
    inventory = J("inventory.json")
    if not inventory:
        return missing("inventory.json")
    history, pilot, news = inventory["market_history"], inventory["pilot"], inventory["news"]
    a, b, c, d = st.columns(4)
    a.metric("Price history", f"{history['first'][:4]} to {history['last'][:4]}", help=f"{history['first']} to {history['last']}")
    b.metric("Daily bars", f"{history['rows']:,}")
    c.metric("Scheduler snapshots", pilot.get("snapshots", 0))
    d.metric("Cached headlines", f"{sum(n['headlines'] for n in news['history']):,}")

    st.subheader("Market data")
    badge(RECORDED, "downloaded and hashed at the time; hashes are re-checked on every load")
    st.markdown(
        f"- **Source:** {history['provider']}; {history['convention']}.\n"
        f"- **Research history:** {history['instruments']} instruments, {history['first']} to {history['last']}, sealed on {when(history['retrieved_at'], '%-d %B %Y')}.\n"
        f"- **Universe:** {inventory['universe']['description']}: {', '.join(inventory['universe']['symbols'])}.\n"
        "- **Known limits:** a single public source; prices are adjusted with today's dividend history, so a later download of the same day can differ slightly; "
        "the fixed list of today's large companies carries survivorship bias."
    )

    if pilot.get("available"):
        first, last = (pd.Timestamp(d).strftime("%-d %B") for d in pilot["collection_window"])
        st.subheader(f"Collection calendar, {first} to {last} 2026")
        badge(RECORDED, "scheduler receipts")
        st.markdown(
            f"**{pilot['decisions']} of {pilot['scheduled_decision_sessions']} scheduled decision sessions were recorded**, across {len(pilot['attempts'])} registered attempts. "
            f"{len(pilot['sessions_not_run'])} decision windows passed with no run and {len(pilot['sessions_rejected'])} download was rejected by validation. "
            "After a miss the pilot halts by design rather than backfilling, and a new attempt has to be registered."
        )
        calendar = pd.DataFrame(pilot["calendar"])
        status = {"complete": "Decision recorded", "missed_deadline": "Missed (no run)", "failed": "Failed (download rejected)", "halted": "Halted", "not_scheduled": "Not scheduled"}
        dates = pd.to_datetime(calendar.session)
        calendar["status"] = calendar.decision_status.map(status)
        calendar["week"] = dates.dt.to_period("W").dt.start_time.dt.strftime("%-d %b")
        calendar["day"] = dates.dt.strftime("%-d")
        base = alt.Chart(calendar).encode(
            x=alt.X("weekday:N", sort=["Mon", "Tue", "Wed", "Thu", "Fri"], title=None, axis=alt.Axis(orient="top", labelAngle=0)),
            y=alt.Y("week:N", sort=calendar.week.unique().tolist(), title="Week starting"),
        )
        cells = base.mark_rect(stroke="white", strokeWidth=3, cornerRadius=4).encode(
            color=alt.Color("status:N", title=None, scale=alt.Scale(domain=list(status.values()), range=["#0f766e", "#b91c1c", "#7c2d12", "#f59e0b", "#e5e7eb"])),
            tooltip=["session", "segment", "decision", "outcome_read"],
        )
        labels = base.mark_text(fontSize=13, fontWeight="bold").encode(text="day:N", color=alt.condition(alt.datum.status == "Not scheduled", alt.value("#6b7280"), alt.value("white")))
        chart((cells + labels).properties(height=250))

        st.markdown("**Every registered attempt**")
        attempts = pd.DataFrame(pilot["attempts"])
        attempts["Stopped by"] = attempts.apply(
            lambda r: "download rejected by validation" if r.rejected else "decision window passed with no run" if r.not_run else "completed", axis=1)
        attempts["First failed session"] = attempts.apply(lambda r: (r.rejected + r.not_run + [""])[0], axis=1)
        table(attempts[["attempt", "start", "end", "required_sessions", "valid_sessions", "Stopped by", "First failed session"]].rename(
            columns={"attempt": "Attempt", "start": "Start", "end": "End", "required_sessions": "Sessions scheduled", "valid_sessions": "Sessions recorded"}))
        st.caption(
            f"Why windows were missed is not in the receipts. The collector's own status file lists the limitation \"{pilot['attempts'][0]['host_requirement']}\". "
            "The rejected download (21 October) had one instrument's open above its high; validation refused the whole batch, as designed."
        )
        recovery = pilot.get("outcome_recovery") or {}
        if recovery.get("required_outcome_reads"):
            st.caption(
                f"Outcome prices were then collected separately: {recovery['completed_outcome_reads']} of {recovery['required_outcome_reads']} scheduled reads completed, "
                f"giving outcomes for {recovery['mature_decision_dates']} of {recovery['required_mature_decision_dates']} decision dates "
                f"({pilot['outcome_rows']} forecast rows). The 22 October decision has no recorded outcome."
            )
        with st.expander("Session by session"):
            table(calendar[["session", "weekday", "segment", "decision", "outcome_read"]].rename(
                columns={"session": "Session", "weekday": "Day", "segment": "Attempt", "decision": "Daily decision", "outcome_read": "Outcome read"}), height=420)

        st.subheader("Paper accounts of the three attempts that recorded decisions")
        badge(RECORDED)
        segments = pd.DataFrame(pilot["segments"])
        table(segments[["segment", "start", "valid_sessions", "orders", "fills", "fees", "last_equity"]].rename(
            columns={"segment": "Attempt", "start": "Start", "valid_sessions": "Sessions recorded", "orders": "Orders", "fills": "Fills", "fees": "Costs paid", "last_equity": "Last recorded equity"}),
            {"Costs paid": "$%.2f", "Last recorded equity": "$%.2f"})
        st.caption("Each account starts from \\$10,000 of paper capital and is kept separate. No external order was sent.")

        with st.expander(f"All {len(pilot['snapshot_list'])} recorded snapshots"):
            table(pd.DataFrame(pilot["snapshot_list"]).rename(columns={"kind": "Purpose", "session": "Session", "retrieved_at": "Downloaded (UTC)", "instruments": "Instruments", "rows": "Bars", "path": "Location"}))
    else:
        st.warning("Recorded receipts are local evidence and are not on this machine. The pages built from them were generated where they exist.")

    st.subheader("News headlines")
    badge(RECONSTRUCTED, "headlines are historical; they were fetched for analysis, not at decision time")
    left, right = st.columns(2)
    left.markdown(f"**Cached history** ({len(news['history'])} tickers)")
    left.dataframe(pd.DataFrame(news["history"]).rename(columns={"symbol": "Ticker", "headlines": "Headlines", "first": "First", "last": "Last"})[["Ticker", "Headlines", "First", "Last"]], hide_index=True, width="stretch")
    left.caption("Source: " + "; ".join(sorted({str(n["source"]) for n in news["history"]})) + ".")
    right.markdown(f"**Pilot weeks, fetched for this review** ({news['pilot_window_total']} headlines)")
    if news["pilot_window"]:
        right.dataframe(pd.DataFrame(news["pilot_window"]).rename(columns={"symbol": "Ticker", "headlines": "Headlines"}).sort_values("Headlines", ascending=False)[["Ticker", "Headlines"]], hide_index=True, width="stretch", height=180)
        right.caption(f"No headlines returned for: {', '.join(news['pilot_window_without_headlines']) or 'none'}.")
    st.caption(news["note"])

    fresh = inventory.get("fresh_download") or {}
    if fresh:
        st.subheader("Fresh download used for the replay and the latest prices")
        badge(SIMULATED, "used only for the simulated replay, later mark-to-market and the demo run")
        st.markdown(f"{fresh['provider']}, retrieved {when(fresh['retrieved_at'])}, last session {fresh['expected_session']}.")


# --------------------------------------------------------------------------
# 3 · Decision walkthrough
# --------------------------------------------------------------------------

def probability_chart(rows: list, height=210):
    frame = pd.DataFrame(rows)
    order = list(dict.fromkeys(r["source"] for r in rows))
    base = alt.Chart(frame).encode(
        y=alt.Y("source:N", sort=order, title=None, axis=alt.Axis(labelLimit=300, labelFontWeight="bold")),
        yOffset=alt.YOffset("class:N", sort=CLASSES),
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, max(0.7, frame.probability.max() + 0.08)]), axis=alt.Axis(format="%"), title="Probability"),
        tooltip=["source", "class", alt.Tooltip("probability:Q", format=".1%")],
    )
    bars = base.mark_bar().encode(color=alt.Color("class:N", title=None, scale=alt.Scale(domain=CLASSES, range=["#b91c1c", "#9ca3af", "#0f766e"]), legend=alt.Legend(orient="top")))
    labels = base.mark_text(align="left", dx=4, fontSize=11, color="#374151").encode(text=alt.Text("probability:Q", format=".0%"))
    return (bars + labels).properties(height=alt.Step(17))


def step_inputs(case, item):
    collection, inputs = case["collection"], item["inputs"]
    badge(RECORDED, "bars the collector downloaded and hashed that evening")
    st.markdown(
        f"At **{when(collection['retrieved_at'])}**, half an hour after the US close, the scheduled collector downloaded daily bars for all 22 instruments "
        f"from **{collection['provider']}** ({collection['requests']} requests). Nothing after the **{when(collection['information_cutoff'])}** close was available to the decision."
    )
    a, b, c, d = st.columns(4)
    a.metric(f"{item['symbol']} bars", f"{inputs['rows']:,}")
    b.metric("History", f"{inputs['first'][:4]} to {inputs['last'][:4]}", help=f"{inputs['first']} to {inputs['last']}")
    c.metric("Last close", money(inputs["last_bars"][-1]["close"]))
    d.metric("Dividends / splits in file", f"{inputs['corporate_actions']['dividends']} / {inputs['corporate_actions']['splits']}")
    frame = pd.DataFrame(inputs["chart"]).melt("dates", var_name="series", value_name="price").dropna()
    frame["series"] = frame.series.map({"close": "Adjusted close", "average_200": "200-session average"})
    chart(alt.Chart(frame).mark_line().encode(
        x=alt.X("dates:T", title=None), y=alt.Y("price:Q", scale=alt.Scale(zero=False), title="Price (USD, adjusted)"),
        color=alt.Color("series:N", title=None, scale=alt.Scale(domain=["Adjusted close", "200-session average"], range=["#111827", "#0f766e"])),
        strokeDash=alt.condition(alt.datum.series == "200-session average", alt.value([5, 4]), alt.value([1, 0])),
    ).properties(height=260, title=f"{item['symbol']}: last 260 sessions up to the decision"))
    st.markdown("**Last five bars in the snapshot**")
    table(pd.DataFrame(inputs["last_bars"])[["date", "open", "high", "low", "close", "volume"]].rename(columns=str.title),
          {"Open": "%.2f", "High": "%.2f", "Low": "%.2f", "Close": "%.2f", "Volume": "localized"})
    st.caption(f"File hash (SHA-256): `{inputs['sha256']}` · snapshot `{collection['snapshot']}` · {collection['convention']}")

    st.markdown("**Sentiment data**")
    sentiment = inputs["sentiment"]
    if sentiment.get("available"):
        badge(RECONSTRUCTED, "fetched for this review from headlines published before the cutoff; not collected at the time and not an input")
        a, b, c = st.columns(3)
        a.metric("Headlines for this session", sentiment["headlines"])
        b.metric("FinBERT tone (positive − negative)", f"{sentiment['tone']:+.3f}" if sentiment["headlines"] else "no news")
        c.metric("Negative tone, 5-session mean", f"{sentiment['negative_5d']:.3f}")
    else:
        st.info("The collector did not gather news. " + sentiment.get("reason", ""))


def step_features(case, item):
    badge(RECONSTRUCTED, "recomputed now from the recorded bars with the pilot's own feature code")
    st.markdown(
        "Seventeen numbers are computed for each instrument from bars up to the decision close only. "
        "The two highlighted rows are the only ones the decision rule reads; all seventeen go to the classifier."
    )
    frame = pd.DataFrame(item["features"]["rows"])
    frame["Used by"] = frame.used_by_trend_rule.map({True: "Trend rule + classifier", False: "Classifier only"})
    frame["Value"] = frame.value.map(lambda v: f"{v:+.4f}")
    shown = frame[["text", "name", "Value", "Used by"]].rename(columns={"text": "Feature", "name": "Column"})
    st.dataframe(shown.style.apply(lambda row: ["background-color: #ccfbf1" if row["Used by"].startswith("Trend") else "" for _ in row], axis=1), hide_index=True, width="stretch", height=640)
    st.caption("Check: the trend flag recomputed from these values equals the flag in the recorded receipt for every instrument (build stops otherwise).")


def step_models(case, item):
    models, regime = item["models"], case["regime"]
    recorded, rebuilt = models["recorded"], models["reconstructed"]
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Classifier forecast**")
        badge(RECORDED, "the probabilities saved in the receipt")
        rows = []
        for source, values in [("Recorded forecast (calibrated mixture)", recorded["probabilities"]), ("Class-frequency prior", rebuilt["prior"]),
                               ("Logistic regression", rebuilt["linear"]), ("Boosted trees", rebuilt["tree"])]:
            rows += [dict(source=source, **{"class": CLASSES[i], "probability": values[i]}) for i in range(3)]
        chart(probability_chart(rows))
        st.markdown(
            f"Most likely class: **{recorded['top_class']}** at {max(recorded['probabilities']):.1%}. "
            f"The two expert models and the prior are recomputed from the saved model file "
            f"(difference from the recorded forecast: {tiny(rebuilt['max_abs_difference_from_recorded'])})."
        )
        model = models["model"]
        st.markdown(
            f"- Model fitted on **{model['fitted_at']}** using {model['n_fit']} rows ({model['fit_start']} to {model['fit_end']}), "
            f"calibrated on {model['n_calibration']} later rows. The last label it saw ended {model['latest_label_end']}, before the decision.\n"
            f"- On its calibration window the mixture scored log loss {model['calibration_log_loss']:.4f} against {model['prior_log_loss']:.4f} for the prior, "
            f"so its gate flag is **{'on' if model['gate'] else 'off'}** (on means it beat the prior by at least 0.01 there)."
        )
        st.warning("**Not used in this decision.** The pilot records these probabilities next to every decision so they can be scored later. The order came from the trend rule alone.")
    with right:
        st.markdown("**Market regime (HMM)**")
        badge(RECONSTRUCTED, "computed now from the SPY bars in the recorded snapshot")
        a, b = st.columns(2)
        a.metric("Regime", regime["state"].title())
        b.metric("P(turbulent)", f"{regime['p_turbulent']:.3f}")
        frame = pd.DataFrame(regime["recent"])
        chart(alt.Chart(frame).mark_area(color="#b45309", opacity=0.6, line={"color": "#b45309"}).encode(
            x=alt.X("dates:T", title=None, axis=alt.Axis(format="%-d %b", labelOverlap=True)), y=alt.Y("p_turbulent:Q", scale=alt.Scale(domain=[0, 1]), title="P(turbulent)")).properties(height=170, title="Previous 60 sessions"))
        states = pd.DataFrame(regime["states"])
        states["Volatility"] = states.annualised_volatility.map(lambda v: f"{v:.0%}")
        states["Typical length"] = states.expected_duration_sessions.map(lambda v: f"{v:.0f} sessions")
        st.dataframe(states[["name", "Volatility", "Typical length"]].rename(columns={"name": "State"}), hide_index=True, width="stretch")
        st.warning("**Not used in this decision.** " + regime["role"])


def step_decision(case, item):
    decision, rule, risk = item["decision"], case["rule"], case["risk"]
    badge(RECORDED, "target weights saved in the receipt; the arithmetic below reproduces them exactly")
    left, right = st.columns([2, 3])
    with left:
        st.html(f'<div class="rv-action" style="color:{ {"BUY": "#0f766e", "SELL": "#b91c1c"}.get(item["action"], "#4b5563")}">{item["action"]} {item["symbol"]}</div>')
        st.markdown(item["reason"])
        st.markdown("**The rule**")
        st.markdown(rule["text"])
        inputs = decision["trend_inputs"]
        prose_table(["Condition", "Value", "Passed"], [
            ["Price above its 200-session average", f"{inputs['average_200']:+.2%}", "yes" if inputs["average_200"] > 0 else "no"],
            ["12-month return, skipping the last month, is positive", f"{inputs['momentum_12_1']:+.2%}", "yes" if inputs["momentum_12_1"] > 0 else "no"],
        ])
        st.markdown(f"Result: trend **{'on' if item['trend'] else 'off'}** → raw weight {decision['raw_weight']:.2%} → final weight **{decision['final_weight']:.2%}**.")
        st.markdown("**What did not enter the decision**")
        st.markdown("\n".join(f"- {text}" for text in rule["not_used"]))
    with right:
        st.markdown("**Portfolio risk checks**")
        badge(RECONSTRUCTED, "each constraint recomputed; the result equals the recorded weights")
        prose_table(["Check", "Value", "Limit", "Effect", "What is measured"], [
            (s["rule"], f"{s['value']:.1%}", f"{s['limit']:.0%}", f"**scaled to {s['scale']:.1%}**" if s["binding"] else "within limit", s["detail"]) for s in risk["steps"]])
        account = risk["account"]
        st.markdown("**Ledger checks when the orders are created**")
        prose_table(["Check", "Value", "Limit", "Effect"], [
            ("Turnover cap per rebalance", f"{account['wanted_turnover_share']:.1%} of equity wanted", f"{account['turnover_limit']:.0%}",
             f"**every order scaled to {account['scale']:.1%}**" if account["binding"] else "within limit"),
            ("Cash reserved for buys", money(account["reserved_cash"]), "cash available", f"reserved at close + {account['reserve_buffer']:.0%}, plus {account['cost_rate'] * 1e4:.0f} bp cost"),
            ("Drawdown stop", f"{account['drawdown']:.2%} from the account's peak", f"-{account['drawdown_limit']:.0%}",
             "**TRIGGERED: all targets set to zero**" if account["drawdown_stop_triggered"] else "not triggered"),
        ])
        size = decision["sizing"]
        st.markdown(
            f"For {item['symbol']}: target value {usd(size['target_value'])}, held {usd(size['held_value'])}, wanted trade {usd(size['wanted_trade_value'])}, "
            f"after the turnover cap **{usd(size['scaled_trade_value'])}**."
        )
    with st.expander(f"All 22 instruments on this date ({rule['trend_count']} pass the trend rule; {rule['final_gross']:.1%} target gross exposure)"):
        rows = []
        for symbol, other in case["symbols"].items():
            p = other["models"]["recorded"]["probabilities"]
            rows.append(dict(Instrument=symbol, Trend="on" if other["trend"] else "off", **{"Target weight": other["weight"]}, Action=other["action"],
                             **{"Classifier top class": other["models"]["recorded"]["top_class"], "P(DOWN)": p[0], "P(FLAT)": p[1], "P(UP)": p[2]}))
        table(pd.DataFrame(rows), height=420, percent={"Target weight": "%.2f%%", "P(DOWN)": "%.0f%%", "P(FLAT)": "%.0f%%", "P(UP)": "%.0f%%"})
        st.caption("Compare the Trend and Action columns with the classifier columns: orders follow the trend flag, whatever the classifier says.")


def step_order(case, item):
    order, orders = item["order"], case["orders"]
    badge(RECORDED, "order events in the paper ledger")
    if order["created"]:
        a, b, c, d = st.columns(4)
        a.metric("Side", order["side"].upper())
        b.metric("Quantity (units)", f"{order['qty']:.6f}")
        c.metric("Reserve price", money(order["limit_price"]), help="Decision close plus 3%. A fill above this price is rejected.")
        d.metric("Cash reserved", money(order["reserved_cash"]))
        st.markdown(f"Order id `{order['id']}`, created at the {when(order['decision_at'])} information cutoff.")
    else:
        st.info(f"No order was created for {item['symbol']} on this date. {item['reason']}")
    st.markdown(
        f"- **This decision created {orders['count']} orders** ({orders['buys']} buys, {orders['sells']} sells) and reserved {usd(orders['reserved_cash'])} of cash.\n"
        f"- **Deployment account:** {orders['deployment_account']}. The trend portfolio runs only as a *shadow* account for evidence gathering.\n"
        "- **External orders:** none. No broker is connected; every order exists only in a local SQLite ledger.\n"
        "- **Units:** fractional quantities of a dividend-adjusted price series, not broker shares."
    )
    st.markdown("**Execution assumptions**")
    st.markdown(
        "- An order can fill only at a close *after* its decision (here, the next session's close).\n"
        "- The fill price is that close; cost is 10 bp commission plus 5 bp slippage on the traded value.\n"
        "- If the next close is more than 3% above the decision close, or cash or exposure limits would be breached, the fill is rejected.\n"
        "- No market impact, no partial fills, no borrowing, long only."
    )


def step_fill(case, item):
    fill, fills, account = item["fill"], case["fills"], case["account"]
    badge(RECORDED, "fill and account marks written by the collector on the next session")
    if fill["filled"]:
        a, b, c, d = st.columns(4)
        a.metric("Filled at", when(fill["asof"], "%-d %b %Y close"))
        b.metric("Fill price", money(fill["price"]), delta=f"{fill['slippage_vs_decision_close']:+.2%} vs decision close", delta_color="off")
        c.metric("Quantity", f"{fill['qty']:.6f}")
        d.metric("Cost paid", money(fill["fee"]))
        st.markdown(f"Position opened: {fill['qty']:.6f} units worth {usd(fill['qty'] * fill['price'])}.")
    elif item["order"]["created"]:
        st.error(fills["reason_if_none"] or "This order was not filled.")
    else:
        st.info("No order, so no fill.")
    st.markdown("**Shadow account**")
    marks = pd.DataFrame(account["marks"])
    marks["asof"] = marks["asof"].map(lambda v: when(v, "%-d %b %Y close"))
    table(marks.rename(columns={"asof": "Marked at", "equity": "Equity", "cash": "Cash", "fees": "Costs paid", "gross_exposure": "Invested"}),
          {"Equity": "$%.2f", "Cash": "$%.2f", "Costs paid": "$%.2f"}, percent={"Invested": "%.1f%%"})
    if fills["count"]:
        after, before, orders = account["after_fill"], account["at_decision"], case["orders"]
        change, costs = after["equity"] - before["equity"], after["fees"] - before["fees"]
        st.markdown(
            f"{fills['count']} of the {orders['count']} orders from this decision filled on {fills['session']}: {usd(fills['notional'])} {'bought' if not orders['sells'] else 'traded'} for {usd(fills['fees'])} in costs. "
            f"Equity went from {usd(before['equity'])} to {usd(after['equity'])}, of which {after['gross_exposure']:.1%} is invested. "
            + ("The whole change is the cost of trading: positions are marked at the same close they were filled at." if abs(change + costs) < 0.005
               else f"Of the {change:+,.2f} USD change, {-costs:+,.2f} is trading cost and the rest is price movement on positions already held.")
        )


def step_outcome(case, item):
    outcome, position = item["outcome"], item["position"]
    st.markdown("**Was the forecast right?**")
    if outcome["available"]:
        badge(outcome["provenance"], outcome["source"])
        a, b, c, d = st.columns(4)
        a.metric("Five-session return", pct(outcome["forward_return"], 2), help=f"Close {outcome['entry_session']} ({usd(outcome['entry_price'])}) to close {outcome['exit_session']} ({usd(outcome['exit_price'])})")
        b.metric("Outcome class", outcome["label"])
        c.metric("Classifier's top class", outcome["predicted"], delta="correct" if outcome["correct"] else "wrong", delta_color="normal" if outcome["correct"] else "inverse")
        d.metric("Log loss, forecast vs prior", f"{outcome['hybrid_log_loss']:.3f} vs {outcome['prior_log_loss']:.3f}", help="Lower is better. Probability given to what happened: "
                 f"{outcome['hybrid_probability_of_outcome']:.1%} vs {outcome['prior_probability_of_outcome']:.1%}")
        st.caption("One row. The classifier is judged on thousands of rows in section 4, where it does not beat the prior.")
    else:
        st.info("The five-session outcome for this decision is not available.")

    st.markdown("**What happened to the position?**")
    if position["opened"]:
        badge(RECONSTRUCTED, position["note"])
        latest = position["latest"]
        a, b, c = st.columns(3)
        a.metric("Cost basis", money(position["cost_basis"]))
        b.metric(f"Value on {latest['session']}", money(latest["value"]))
        c.metric("Profit or loss if held", f"{latest['pnl']:+.2f} USD", delta=pct(latest["pnl_percent"]), delta_color="normal")
        frame = pd.DataFrame(dict(date=position["dates"], pnl=position["pnl"]))
        chart(alt.Chart(frame).mark_area(line={"color": "#111827"}, color=alt.Gradient(gradient="linear", stops=[alt.GradientStop(color="#fecaca", offset=0), alt.GradientStop(color="#ccfbf1", offset=1)], x1=1, x2=1, y1=1, y2=0), opacity=0.7).encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%-d %b", labelOverlap=True)), y=alt.Y("pnl:Q", title="Unrealised P&L (USD)")).properties(height=200, title=f"{item['symbol']} position, marked with later public prices"))
    else:
        st.info("No position was opened for this instrument from this decision.")

    st.markdown("**What happened to the account?**")
    badge(RECORDED, "the collector's own record")
    after, last = case["after"], case["account"]["last_recorded"]
    missed = ", ".join(after["missing_sessions"]) or "none"
    st.markdown(
        f"- Last recorded account state: {when(last['asof'], '%-d %b %Y close')}, equity **{usd(last['equity'])}**, costs paid {usd(last['fees'])}, realised P&L {usd(last['realized_pnl'])}.\n"
        f"- Sessions recorded in this pilot segment: {after['valid_observed_sessions']} of {after['required_sessions']} required. Missed: {missed}.\n"
        + (f"- The segment **halted** on {when(after['halt']['at'], '%-d %b %Y')}: \"{after['halt']['reason']}\". Nothing was backfilled, so the recorded account stops there." if after["halt"] else "- The segment ended without a halt.")
    )
    held = case["held_forward"]
    badge(RECONSTRUCTED, held["note"])
    frame = pd.DataFrame(dict(date=held["dates"], equity=held["equity"]))
    rule = alt.Chart(pd.DataFrame(dict(y=[10000]))).mark_rule(color="#9ca3af", strokeDash=[4, 4]).encode(y="y:Q")
    chart((alt.Chart(frame).mark_line(color="#b45309").encode(x=alt.X("date:T", title=None, axis=alt.Axis(format="%-d %b", labelOverlap=True)), y=alt.Y("equity:Q", scale=alt.Scale(zero=False), title="Equity (USD)")) + rule)
          .properties(height=200, title="Last recorded positions, simply held to the latest session"))
    st.markdown(f"Held to {held['latest']['session']}: **{usd(held['latest']['equity'])}** ({pct(held['latest']['net_return'], 2)}).")
    st.caption("For what the full policy, with its weekly rebalances, would have done over the same weeks, see the simulated replay in section 7. That replay is never shown as a recorded trade.")


STEP_RENDERERS = [step_inputs, step_features, step_models, step_decision, step_order, step_fill, step_outcome]


def view_walkthrough():
    index = J("walkthrough/index.json")
    if not index:
        st.header(next(v["title"] for v in nav.VIEWS if v["id"] == "walkthrough"))
        missing("walkthrough/index.json")
        return {}
    case_ids = [c["case_id"] for c in index["cases"]]
    if "case" not in st.session_state:
        st.session_state["case"] = _query("case", case_ids, index["default_case"])
    if "symbol" not in st.session_state:
        st.session_state["symbol"] = _query("symbol", index["cases"][0]["symbols"], index["default_symbol"])
    if "step" not in st.session_state:
        st.session_state["step"] = int(_query("step", [str(s["n"]) for s in nav.STEPS], "1"))
    headline("walkthrough")
    titles = {c["case_id"]: c["title"] + ("" if c["filled"] else " · orders never filled" if c["rebalance"] else " · hold day") for c in index["cases"]}
    a, b = st.columns([3, 1])
    case_id = a.selectbox("Recorded decision", case_ids, format_func=titles.get, key="case")
    symbol = b.selectbox("Instrument", index["cases"][0]["symbols"], key="symbol")
    case = J(f"walkthrough/{case_id}.json")
    item = case["symbols"][symbol]
    step_titles = {s["n"]: f"{s['n']} · {s['title']}" for s in nav.STEPS}
    step = st.radio("Step", [s["n"] for s in nav.STEPS], format_func=step_titles.get, horizontal=True, key="step")
    st.query_params.update(view="walkthrough", case=case_id, symbol=symbol, step=str(step))

    verification = case["verification"]
    with st.expander("Is this replay faithful? Five checks, all passed", expanded=False):
        st.markdown(
            f"- The receipt file matches the SHA-256 hash registered on 27 October 2026 (`{verification['receipt_sha256'][:16]}…`).\n"
            "- The receipt binds to its plan, and the plan to the market snapshot manifest.\n"
            f"- Recomputing the classifier from the recorded bars and the saved model reproduces every recorded probability (maximum difference {tiny(verification['max_abs_probability_difference'])}).\n"
            "- Recomputing the trend rule and risk constraints reproduces every recorded target weight.\n"
            "- Recomputing the ledger's sizing reproduces every recorded order quantity."
        )
    narration = J("narration.json")
    info = next(s for s in nav.STEPS if s["n"] == step)
    st.subheader(f"Step {step} of 7 · {info['title']}")
    if narration and case_id == narration["default_case"] and symbol == narration["default_symbol"]:
        st.html(f'<p class="rv-caption">{narration["walkthrough"][str(step)]["caption"]}</p>')
    STEP_RENDERERS[step - 1](case, item)
    return dict(case=case_id, symbol=symbol, step=step)


# --------------------------------------------------------------------------
# 4 · Classifier statistics
# --------------------------------------------------------------------------

MODEL_LABELS = {"hybrid": "Calibrated mixture (recorded forecast)", "prior": "Class-frequency prior", "linear": "Logistic regression", "tree": "Boosted trees", "raw_mix": "Mixture before calibration"}


def confusion_chart(frame: pd.DataFrame, title: str):
    base = alt.Chart(frame).encode(x=alt.X("predicted:N", sort=CLASSES, title="Predicted"), y=alt.Y("actual:N", sort=CLASSES, title="Actual"))
    return (base.mark_rect().encode(color=alt.Color("count:Q", scale=alt.Scale(scheme="teals"), legend=None))
            + base.mark_text(fontSize=14).encode(text="count:Q", color=alt.condition(alt.datum.count > frame["count"].max() * 0.6, alt.value("white"), alt.value("#111827")))).properties(height=220, title=title)


def view_classifiers():
    headline("classifiers")
    summary, by_period, paired = J("classifier/summary.json"), T("classifier/by_period.csv"), T("classifier/paired_vs_prior.csv")
    if summary is None or by_period is None:
        return missing("classifier/summary.json")
    contract = summary["label_contract"]
    st.info(
        f"**What is scored.** Decision at the close of day t. Entry at the {contract['entry']}; exit at the {contract['exit']}. "
        f"UP if the return is above +1%, DOWN if below −1%, FLAT otherwise. Primary metric: log loss (lower is better). "
        f"Baseline: the class-frequency prior. {contract['note']}"
    )
    st.subheader("1. Saved out-of-fold forecasts, 2024 to September 2026")
    st.caption(f"{summary['history']['n']:,} instrument-days on {summary['history']['sessions']} dates. Every model was fitted only on data before the dates it is scored on. "
               "All of these dates were inspected in earlier work, so this is exploratory evidence, not a holdout.")
    names = {"all": "All dates", "development": "Development (2024 to 2025)", "selection": "Selection (Jan to Jul 2026)", "audit": "Audit (Jul to Sep 2026)"}
    period = st.segmented_control("Period", list(names), format_func=names.get, default="all", key="classifier_period") or "all"
    frame = by_period.loc[by_period.period == period].copy()
    frame["Model"] = frame.model.map(MODEL_LABELS)
    table(frame[["Model", "sessions", "n", "log_loss", "accuracy", "balanced_accuracy", "macro_f1", "brier", "predicted_down", "predicted_flat", "predicted_up"]].rename(
        columns={"sessions": "Dates", "n": "Forecasts", "log_loss": "Log loss ↓", "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "macro_f1": "Macro F1", "brier": "Brier ↓",
                 "predicted_down": "Predicted DOWN", "predicted_flat": "Predicted FLAT", "predicted_up": "Predicted UP"}),
        {"Log loss ↓": "%.4f", "Macro F1": "%.3f", "Brier ↓": "%.4f"}, percent={"Accuracy": "%.1f%%", "Balanced accuracy": "%.1f%%"})
    pair = frame.set_index("model")
    measures = [("log loss", "log_loss", True), ("Brier score", "brier", True), ("accuracy", "accuracy", False), ("balanced accuracy", "balanced_accuracy", False), ("macro F1", "macro_f1", False)]
    prior_better = [name for name, column, lower in measures if (pair.loc["prior", column] < pair.loc["hybrid", column]) == lower]
    mixture_better = [name for name, _, _ in measures if name not in prior_better]
    st.caption(
        f"Uniform guessing scores log loss 1.0986. In this period the prior scores better than the calibrated mixture on {', '.join(prior_better) or 'nothing'}; "
        f"the mixture scores better on {', '.join(mixture_better) or 'nothing'}. The prior names DOWN as the likeliest class on {pair.loc['prior', 'predicted_down'] / pair.loc['prior', 'n']:.0%} of rows "
        f"and UP on {pair.loc['prior', 'predicted_up'] / pair.loc['prior', 'n']:.0%}, so it scores poorly on the two class-balanced measures whatever its log loss."
    )

    st.markdown("**Difference in log loss from the prior, with 95% intervals** (right of zero = worse than the prior)")
    gaps = paired.copy()
    gaps["Model"] = gaps.model.map(MODEL_LABELS)
    gaps["Period"] = gaps.period.map(names)
    points = alt.Chart(gaps).encode(y=alt.Y("Model:N", title=None, sort=list(MODEL_LABELS.values()), axis=alt.Axis(labelLimit=280)), color=alt.Color("verdict:N", title=None, legend=alt.Legend(orient="top", labelLimit=320), scale=alt.Scale(
        domain=["worse than the prior", "not distinguishable from the prior", "better than the prior"], range=["#b91c1c", "#6b7280", "#0f766e"])))
    bars = points.mark_rule(strokeWidth=3).encode(x=alt.X("log_loss_ci_low:Q", title="Log loss minus prior"), x2="log_loss_ci_high:Q")
    dots = points.mark_point(filled=True, size=70).encode(x="log_loss_difference:Q", tooltip=["Period", "Model", alt.Tooltip("log_loss_difference:Q", format="+.4f"), alt.Tooltip("log_loss_ci_low:Q", format="+.4f"), alt.Tooltip("log_loss_ci_high:Q", format="+.4f"), "sessions"])
    zero = alt.Chart(gaps).mark_rule(color="#111827", strokeDash=[4, 4]).encode(x=alt.datum(0))
    chart(alt.layer(bars, dots, zero).properties(height=110).facet(row=alt.Row("Period:N", sort=list(names.values()), title=None, header=alt.Header(labelAngle=0, labelAlign="left"))))
    better = int((gaps.verdict == "better than the prior").sum())
    st.caption(summary["inference"] + (". No model is better than the prior in any period." if not better else f". {better} of {len(gaps)} comparisons favour a model."))

    left, right = st.columns(2)
    confusion = T("classifier/confusion.csv")
    selected = confusion.loc[confusion.period == period]
    left.altair_chart(confusion_chart(selected.loc[selected.model == "hybrid"], "Calibrated mixture"), width="stretch")
    right.altair_chart(confusion_chart(selected.loc[selected.model == "prior"], "Class-frequency prior"), width="stretch")
    report = T("classifier/class_report.csv")
    classes = report.loc[(report.period == period) & report.model.isin(["hybrid", "prior"])].copy()
    classes["Model"] = classes.model.map(MODEL_LABELS)
    with st.expander("Precision, recall and F1 by class"):
        table(classes[["Model", "class_name", "precision", "recall", "f1", "support", "predicted"]].rename(
            columns={"class_name": "Class", "precision": "Precision", "recall": "Recall", "f1": "F1", "support": "Actual count", "predicted": "Predicted count"}),
            {"Precision": "%.3f", "Recall": "%.3f", "F1": "%.3f"})
    with st.expander("By instrument"):
        symbols = T("classifier/by_symbol.csv")
        chart(alt.Chart(symbols).mark_bar().encode(
            x=alt.X("symbol:N", sort=None, title=None), y=alt.Y("log_loss_difference:Q", title="Log loss minus prior"),
            color=alt.condition(alt.datum.log_loss_difference > 0, alt.value("#b91c1c"), alt.value("#0f766e")),
            tooltip=["symbol", alt.Tooltip("hybrid_log_loss:Q", format=".4f"), alt.Tooltip("prior_log_loss:Q", format=".4f"), "n"]).properties(height=240))
        st.caption("Per-instrument differences are descriptive; with 22 instruments some will fall below zero by chance.")

    st.subheader("2. Split by market regime")
    regimes, contrasts = T("classifier/by_regime.csv"), T("classifier/regime_contrasts.csv")
    if regimes is not None:
        st.caption("The same forecasts, grouped by the HMM regime known at each decision close (section 5). Registered as two contrasts with a Holm adjustment.")
        shares = summary["regime_shares"]
        mix = pd.DataFrame([dict(Regime=k.title(), Dates=v["sessions"], Forecasts=v["n"], **{f"Actual {c}": v["label_share"][c] for c in CLASSES}) for k, v in shares.items()])
        table(mix, percent={f"Actual {c}": "%.1f%%" for c in CLASSES})
        flat = {k: v["label_share"]["FLAT"] for k, v in shares.items()}
        worse = contrasts.loc[contrasts.verdict == "worse than the prior", "regime"].tolist()
        st.caption(
            f"With a fixed ±1% band, FLAT outcomes made up {flat['calm']:.0%} of calm rows and {flat['turbulent']:.0%} of turbulent rows. "
            + ("The mixture is worse than the prior in both regimes after the Holm adjustment (last table)." if len(worse) == 2 else
               f"After the Holm adjustment the mixture is worse than the prior in: {', '.join(worse) or 'neither regime'} (last table).")
        )
        view = regimes.loc[regimes.model.isin(["hybrid", "prior"])].copy()
        view["Model"] = view.model.map(MODEL_LABELS)
        view["Regime"] = view.regime.str.title()
        table(view[["Regime", "Model", "sessions", "n", "log_loss", "accuracy", "balanced_accuracy", "macro_f1"]].rename(
            columns={"sessions": "Dates", "n": "Forecasts", "log_loss": "Log loss ↓", "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "macro_f1": "Macro F1"}),
            {"Log loss ↓": "%.4f", "Macro F1": "%.3f"}, percent={"Accuracy": "%.1f%%", "Balanced accuracy": "%.1f%%"})
        shown = contrasts.assign(Regime=contrasts.regime.str.title())[["Regime", "sessions", "log_loss_difference", "log_loss_ci_low", "log_loss_ci_high", "holm_p", "verdict"]]
        table(shown.rename(columns={"sessions": "Dates", "log_loss_difference": "Mixture − prior", "log_loss_ci_low": "95% low", "log_loss_ci_high": "95% high", "holm_p": "Holm p", "verdict": "Reading"}),
              {"Mixture − prior": "%+.4f", "95% low": "%+.4f", "95% high": "%+.4f", "Holm p": "%.3f"})

    st.subheader("3. Forecasts recorded by the live collector")
    prospective = summary["prospective"]
    if prospective.get("available"):
        badge(RECORDED, "forecasts saved at decision time, outcomes read later by the collector")
        rows = pd.DataFrame([dict(Model=MODEL_LABELS[m], **{"Log loss ↓": prospective[m]["log_loss"], "Accuracy": prospective[m]["accuracy"], "Brier ↓": prospective[m]["brier"], "Macro F1": prospective[m]["macro_f1"]}) for m in ["hybrid", "prior"]])
        a, b = st.columns([3, 2])
        with a:
            table(rows, {"Log loss ↓": "%.4f", "Brier ↓": "%.4f", "Macro F1": "%.3f"}, percent={"Accuracy": "%.1f%%"})
            st.caption(f"{prospective['n']} rows: {prospective['symbols']} instruments on {len(prospective['decision_sessions'])} decision dates ({', '.join(prospective['decision_sessions'])}). {prospective['reading']}")
        with b:
            daily = pd.DataFrame([dict(Date=k, Difference=v) for k, v in prospective["daily_log_loss_difference"].items()])
            table(daily.rename(columns={"Difference": "Mixture − prior, that date"}), {"Mixture − prior, that date": "%+.4f"})
    else:
        st.info("Recorded outcome rows are not available on this machine.")

    st.subheader("4. Simulated replay since the release")
    replay = J("replay/summary.json")
    if replay and replay["classification"].get("available"):
        result = replay["classification"]
        badge(SIMULATED, "the same code replayed on a November download; not a pilot observation")
        replayed = {"hybrid": "Calibrated mixture (replayed)", "prior": "Class-frequency prior (replayed)"}
        rows = pd.DataFrame([dict(Model=replayed[m], **{"Log loss ↓": result[m]["log_loss"], "Accuracy": result[m]["accuracy"], "Balanced accuracy": result[m]["balanced_accuracy"], "Macro F1": result[m]["macro_f1"]}) for m in ["hybrid", "prior"]])
        table(rows, {"Log loss ↓": "%.4f", "Macro F1": "%.3f"}, percent={"Accuracy": "%.1f%%", "Balanced accuracy": "%.1f%%"})
        gap = result["hybrid_minus_prior_log_loss"]
        st.caption(f"{result['n']} forecasts on {result['sessions']} dates, {result['first']} to {result['last']}. Mixture − prior: {gap['effect']:+.4f} (95% interval {gap['ci_low']:+.4f} to {gap['ci_high']:+.4f}). {result['reading']}")

    st.subheader("5. The two earlier studies quoted in the summary")
    studies = summary["earlier_studies"]
    study, screen = studies["study_848"], studies["controller_screen"]
    rows = [
        dict(Study=study["name"], Model="Logistic/tree mixture, calibrated", Forecasts=study["observations"], **{"Log loss ↓": study["mixture"]["log_loss"], "Prior": study["prior"]["log_loss"],
             "Model − prior": study["difference"]["effect"], "95% low": study["difference"]["ci_low"], "95% high": study["difference"]["ci_high"]}),
    ] + [
        dict(Study=screen["name"], Model=label, Forecasts=screen["observations"], **{"Log loss ↓": screen["models"][key]["log_loss"], "Prior": screen["models"]["prior"]["log_loss"],
             "Model − prior": screen["versus_prior"][key]["effect"], "95% low": screen["versus_prior"][key]["ci_low"], "95% high": screen["versus_prior"][key]["ci_high"]})
        for key, label in [("hedge", "Hedge controller"), ("stack", "Stacking controller")]
    ]
    table(pd.DataFrame(rows), {"Log loss ↓": "%.4f", "Prior": "%.4f", "Model − prior": "%+.4f", "95% low": "%+.4f", "95% high": "%+.4f"})
    st.caption(f"Sources: `{study['source']}`, `{screen['source']}`. A plain-language reading is on the overview page.")

    check = summary["cross_check"]
    st.success(f"Check: these statistics were recomputed independently of the research tables and agree with `{check['reference']}` on {check['rows_compared']} rows to within {tiny(check['max_abs_metric_difference'])}.")


# --------------------------------------------------------------------------
# 5 · Market regimes
# --------------------------------------------------------------------------

def view_regimes():
    headline("regimes")
    summary, history, recent = J("regime/summary.json"), T("regime/posterior_history.csv"), T("regime/posterior_recent.csv")
    if summary is None or history is None:
        return missing("regime/summary.json")
    badge(RECONSTRUCTED, "written for this review; computed from recorded prices")
    st.warning("**Role.** " + summary["role"])
    config = summary["config"]
    st.markdown(
        f"A Gaussian hidden Markov model with **{config['n_states']} states** reads two numbers per session from SPY: the daily log return and the log of {config['vol_window']}-session realised volatility. "
        f"It is refitted every {config['refit_every']} sessions on the previous {config['window']} sessions. Between refits the parameters are frozen, and the probability shown for a day uses data up to that day's close only."
    )
    full = pd.concat([history, recent.loc[recent.date > history.date.max()]]) if recent is not None else history
    full = full.assign(date=pd.to_datetime(full.date))
    spans = full.assign(block=(full.state != full.state.shift()).cumsum()).groupby("block").agg(state=("state", "first"), start=("date", "first"), end=("date", "last")).query("state == 'turbulent'")
    bands = alt.Chart(spans).mark_rect(color="#f59e0b", opacity=0.25).encode(x="start:T", x2="end:T")
    price = alt.Chart(full).mark_line(color="#111827", strokeWidth=1.2).encode(x=alt.X("date:T", title=None), y=alt.Y("close:Q", scale=alt.Scale(type="log", zero=False), title="SPY adjusted close (log scale)"))
    chart((bands + price).properties(height=280, title="SPY with turbulent sessions shaded"))
    chart(alt.Chart(full).mark_area(color="#b45309", opacity=0.65).encode(x=alt.X("date:T", title=None), y=alt.Y("p_turbulent:Q", title="P(turbulent)", scale=alt.Scale(domain=[0, 1]))).properties(height=130))
    st.caption(f"Sealed September history to {summary['series']['history']['last']}; the {summary['series']['recent']['new_sessions']} sessions after that come from the November download.")

    left, right = st.columns(2)
    with left:
        st.markdown("**The two states** (median across refits)")
        states = pd.DataFrame(summary["states"])
        table(states[["state", "median_annualised_volatility", "median_annualised_return", "median_expected_duration_sessions"]].rename(
            columns={"state": "State", "median_annualised_volatility": "Volatility / yr", "median_annualised_return": "Mean return / yr", "median_expected_duration_sessions": "Typical spell (sessions)"}),
            {"Typical spell (sessions)": "%.0f"}, percent={"Volatility / yr": "%.1f%%", "Mean return / yr": "%+.1f%%"})
        st.markdown("**Known episodes** (share of sessions marked turbulent)")
        episodes = pd.DataFrame(summary["episodes"])
        episodes["Period"] = episodes.start.str.slice(0, 7) + " to " + episodes.end.str.slice(0, 7)
        table(episodes[["name", "turbulent_share", "sessions", "Period"]].rename(columns={"name": "Episode", "sessions": "Sessions", "turbulent_share": "Marked turbulent"}), percent={"Marked turbulent": "%.0f%%"})
    with right:
        st.markdown("**Turbulent share by year**")
        years = pd.DataFrame(summary["occupancy_by_year"])
        chart(alt.Chart(years).mark_bar(color="#b45309").encode(x=alt.X("year:O", title=None), y=alt.Y("turbulent_share:Q", axis=alt.Axis(format="%"), title="Sessions marked turbulent"), tooltip=["year", "sessions", alt.Tooltip("turbulent_share:Q", format=".0%")]).properties(height=250))
        latest, first_new = summary["occupancy_recent"], summary["series"]["recent"]["first_new"]
        turned = full.loc[(full.date >= first_new) & (full.state == "turbulent"), "date"]
        st.caption(f"Since {when(first_new, '%-d %B %Y')}: {latest['turbulent_share']:.0%} of {latest['sessions']} sessions turbulent"
                   + (f" (first turbulent close: {when(turned.min(), '%-d %B')})." if len(turned) else "."))

    st.subheader("Does the regime say anything about what follows?")
    forward = summary["forward_validation"]
    rows = pd.DataFrame(forward["rows"])
    table(rows[["state", "sessions", "mean_next_day_abs_return", "share_next_day_move_over_1pct", "mean_next_21_session_volatility", "mean_next_21_session_return"]].rename(
        columns={"state": "Regime at the close", "sessions": "Sessions", "mean_next_day_abs_return": "Mean |next-day SPY return|", "share_next_day_move_over_1pct": "Next day moves > 1%",
                 "mean_next_21_session_volatility": "Volatility, next 21 sessions", "mean_next_21_session_return": "Mean return, next 21 sessions"}),
        percent={"Mean |next-day SPY return|": "%.2f%%", "Next day moves > 1%": "%.1f%%", "Volatility, next 21 sessions": "%.1f%%", "Mean return, next 21 sessions": "%+.2f%%"})
    st.caption(forward["note"] + f" Rank correlation of P(turbulent) with the next 21 sessions' volatility: {forward['spearman_p_turbulent_vs_next_21_volatility']:+.2f}; with their return: {forward['spearman_p_turbulent_vs_next_21_return']:+.2f}.")

    st.subheader("How to verify it")
    checks = summary["checks"]
    st.markdown(
        f"- **Causality.** The model was rerun three times with all data after {', '.join(c['cut'] for c in checks['truncation'])} removed. The largest change in any earlier probability was "
        f"{max(c['max_abs_difference'] for c in checks['truncation']):.2g} across {sum(c['rows_compared'] for c in checks['truncation']):,} compared sessions: {'passed' if checks['truncation_passed'] else 'FAILED'}.\n"
        f"- **Convergence.** {summary['refits']['converged']} of {summary['refits']['count']} refits converged.\n"
        f"- **Stability across downloads.** September and November downloads agree on the regime label on {checks['vintage']['label_agreement']:.0%} of {checks['vintage']['sessions']} shared sessions.\n"
        "- **Recovery on synthetic data.** `pytest tests/test_review_v1_regime.py` generates data from a known two-state process and checks the states are recovered."
    )
    st.markdown("**Assumptions**")
    st.markdown("\n".join(f"- {a}" for a in summary["assumptions"]))


# --------------------------------------------------------------------------
# 6 · Sentiment link
# --------------------------------------------------------------------------

def view_sentiment():
    headline("sentiment")
    summary = J("sentiment/summary.json")
    if summary is None:
        return missing("sentiment/summary.json")
    badge(RECONSTRUCTED, "headlines are historical; scores were computed for this review")
    st.warning("**Role.** " + summary["role"])
    st.markdown(
        f"The pilot's feature set has {len(summary['pilot_feature_columns'])} columns, all from price and volume: "
        f"`{', '.join(summary['pilot_feature_columns'])}`. A separate FinBERT pipeline exists in `src/features/sentiment.py` and was never wired in. "
        "What follows therefore asks a narrower question: *is sentiment related to what the models and the trend rule said, or to what happened next?*"
    )
    st.caption(f"Scorer: {summary['scorer']}. Timing rule: {summary['attribution']}.")

    st.subheader("1. At the five recorded decision dates")
    decisions, frame = summary["decision_dates"], T("sentiment/decision_dates.csv")
    if decisions.get("available") and frame is not None:
        st.caption(f"{decisions['headlines_fetched']} headlines from {decisions['source']} for {decisions['window'][0]} to {decisions['window'][1]}. {decisions['note']} "
                   f"No headlines were returned for {', '.join(decisions['instruments_without_headlines'])}.")
        session = st.segmented_control("Decision date", sorted(frame.session.unique()), default=sorted(frame.session.unique())[0], key="sentiment_session") or sorted(frame.session.unique())[0]
        day = frame.loc[(frame.session == session) & frame.has_news].copy()
        day["In portfolio"] = day.in_portfolio.map({True: "Trend rule on (held)", False: "Trend rule off"})
        left, right = st.columns([3, 2])
        with left:
            base = alt.Chart(day).encode(x=alt.X("tone:Q", title="FinBERT tone (positive − negative)"), y=alt.Y("model_score:Q", title="Classifier score: P(UP) − P(DOWN)"))
            chart((base.mark_point(filled=True, size=110).encode(color=alt.Color("In portfolio:N", title=None, scale=alt.Scale(domain=["Trend rule on (held)", "Trend rule off"], range=["#0f766e", "#9ca3af"])),
                                                                 tooltip=["symbol", "headlines", alt.Tooltip("tone:Q", format="+.3f"), alt.Tooltip("model_score:Q", format="+.3f")])
                   + base.mark_text(dy=-12, fontSize=11).encode(text="symbol:N")).properties(height=320, title=f"{session}: instruments with at least one headline"))
        with right:
            info = next(r for r in decisions["by_date"] if r["session"] == session)
            st.metric("Instruments with news", f"{info['instruments_with_news']} of 22")
            st.metric("Rank correlation, tone vs classifier score", f"{info['spearman_tone_vs_model_score']:+.2f}" if info["spearman_tone_vs_model_score"] is not None else "n/a")
            st.metric("Mean tone: held vs not held", f"{info['mean_tone_in_portfolio'] or 0:+.3f} vs {info['mean_tone_not_in_portfolio'] or 0:+.3f}")
            st.caption(f"{info['instruments_with_news']} points on one date. Read as a picture, not a test.")
        with st.expander("All rows for this date"):
            shown = frame.loc[frame.session == session, ["symbol", "headlines", "tone", "negative_5d", "p_down", "p_flat", "p_up", "trend", "target_weight"]]
            table(shown.rename(columns={"symbol": "Instrument", "headlines": "Headlines", "tone": "Tone", "negative_5d": "Negative, 5-session", "p_down": "P(DOWN)", "p_flat": "P(FLAT)", "p_up": "P(UP)", "trend": "Trend on", "target_weight": "Target weight"}),
                  {"Tone": "%+.3f", "Negative, 5-session": "%.3f"}, percent={"P(DOWN)": "%.0f%%", "P(FLAT)": "%.0f%%", "P(UP)": "%.0f%%", "Target weight": "%.2f%%"})
    else:
        st.info("Decision-date sentiment is not available: " + decisions.get("reason", "not built"))

    st.subheader("2. Over the cached news history (three tickers)")
    history = summary["history"]
    rows = pd.DataFrame(history["rows"])
    points = []
    for _, row in rows.iterrows():
        for key, label in [("model_score", "Classifier score"), ("trend_flag", "Trend rule on"), ("forward_return", "Next five-session return")]:
            points.append(dict(Ticker=row.symbol, Against=label, rho=row[f"tone_vs_{key}_rho"], low=row[f"tone_vs_{key}_ci_low"], high=row[f"tone_vs_{key}_ci_high"], sessions=row.sessions_with_news))
    points = pd.DataFrame(points)
    base = alt.Chart(points).encode(y=alt.Y("Ticker:N", title=None), color=alt.Color("Ticker:N", legend=None))
    zero = alt.Chart(points).mark_rule(color="#111827", strokeDash=[4, 4]).encode(x=alt.datum(0))
    chart(alt.layer(base.mark_rule(strokeWidth=3).encode(x=alt.X("low:Q", title="Rank correlation with daily FinBERT tone", scale=alt.Scale(domain=[-0.4, 0.4])), x2="high:Q"),
                    base.mark_point(filled=True, size=80).encode(x="rho:Q", tooltip=["Ticker", "Against", alt.Tooltip("rho:Q", format="+.3f"), "sessions"]), zero)
          .properties(height=90).facet(row=alt.Row("Against:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left"))))
    table(rows[["symbol", "first", "last", "sessions_with_news", "median_headlines_per_session", "tone_vs_model_score_rho", "tone_vs_trend_flag_rho", "tone_vs_forward_return_rho"]].rename(
        columns={"symbol": "Ticker", "first": "From", "last": "To", "sessions_with_news": "Sessions with news", "median_headlines_per_session": "Median headlines",
                 "tone_vs_model_score_rho": "ρ with classifier score", "tone_vs_trend_flag_rho": "ρ with trend flag", "tone_vs_forward_return_rho": "ρ with next 5-session return"}),
        {"ρ with classifier score": "%+.3f", "ρ with trend flag": "%+.3f", "ρ with next 5-session return": "%+.3f"})
    st.markdown("\n".join(f"- Tone {text}." for text in audit.sentiment_link_text(history["rows"])))
    st.caption(history["note"] + " Only the last relationship, with the return that follows, would matter for trading.")

    st.subheader("3. Does adding sentiment improve the classifier?")
    ablation = summary["ablation"]
    if ablation.get("status") == "run":
        rows = pd.DataFrame(ablation["rows"])
        table(rows[["symbol", "sessions", "price_only_log_loss", "with_sentiment_log_loss", "prior_log_loss", "difference", "ci_low", "ci_high"]].rename(
            columns={"symbol": "Ticker", "sessions": "Dates", "price_only_log_loss": "Price only ↓", "with_sentiment_log_loss": "Price + sentiment ↓", "prior_log_loss": "Prior ↓",
                     "difference": "With − without", "ci_low": "95% low", "ci_high": "95% high"}),
            {"Price only ↓": "%.4f", "Price + sentiment ↓": "%.4f", "Prior ↓": "%.4f", "With − without": "%+.4f", "95% low": "%+.4f", "95% high": "%+.4f"})
        verdict = audit.interval_verdict(rows.ci_low.iloc[-1], rows.ci_high.iloc[-1])
        (st.error if verdict == "worse" else st.info)(
            f"**{audit.sentiment_ablation_text(ablation)}** This was the registered optional ablation: the same walk-forward schedule and models as the pilot, with and without the sentiment columns."
        )
        st.caption(f"{ablation['reading']} {ablation['caveat']}")
        st.success(f"Check: the price-only arm reproduces the sealed forecasts (maximum difference {tiny(max(ablation['parity_with_sealed_price_only'].values()))}), so the only thing that differs between the two arms is the six sentiment columns.")
    else:
        st.info("The ablation was not run in this build.")


# --------------------------------------------------------------------------
# 7 · Trading performance
# --------------------------------------------------------------------------

ARMS = ["equal_weight", "trend", "equal_weight_regime", "trend_regime"]
ARM_LABELS = {
    "equal_weight": "Equal weight (benchmark)", "trend": "Trend rule (pilot shadow)",
    "equal_weight_regime": "Equal weight × regime overlay", "trend_regime": "Trend rule × regime overlay", "cash": "Cash",
}


def curve_chart(frame: pd.DataFrame, title: str, height=300):
    data = frame.loc[frame.arm.isin(ARMS)].assign(date=lambda d: pd.to_datetime(d.date), Portfolio=lambda d: d.arm.map(ARM_LABELS))
    return alt.Chart(data).mark_line(strokeWidth=1.6).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%-d %b", labelOverlap=True) if data.date.nunique() < 120 else alt.Axis()), y=alt.Y("equity:Q", scale=alt.Scale(zero=False), title="Equity (USD)"),
        color=alt.Color("Portfolio:N", title=None, scale=alt.Scale(domain=[ARM_LABELS[a] for a in ARMS], range=[ARM_COLORS[a] for a in ARMS]), legend=alt.Legend(orient="top", labelLimit=260, columns=2)),
        tooltip=["date:T", "Portfolio", alt.Tooltip("equity:Q", format=",.2f")],
    ).properties(height=height, title=title)


def metric_table(metrics: pd.DataFrame, period: str, sharpe: bool = True, details: bool = True):
    """Headline trading metrics for the four invested portfolios; the rest goes in an expander."""
    frame = metrics.loc[(metrics.period == period) & metrics.arm.isin(ARMS)].set_index("arm").loc[ARMS].reset_index()
    frame["Portfolio"] = frame.arm.map(ARM_LABELS)
    frame["P&L (USD)"] = frame.pnl_usd.map(lambda v: f"{v:+,.0f}")
    if sharpe:
        frame["Sharpe (95% interval)"] = frame.apply(lambda r: f"{r.sharpe_zero_rf:.2f}  ({r.sharpe_ci_low:.2f} to {r.sharpe_ci_high:.2f})", axis=1)
    columns = {"net_return": "Net return", "max_drawdown": "Max drawdown", "fills": "Fills (trades)", "combined_fees_usd": "Costs paid", "mean_gross_exposure": "Mean invested"}
    order = ["Portfolio", "Net return", "P&L (USD)"] + (["Sharpe (95% interval)"] if sharpe else []) + ["Max drawdown", "Fills (trades)", "Costs paid", "Mean invested"]
    table(frame.rename(columns=columns)[order], {"Costs paid": "$%.2f"}, percent={"Net return": "%+.1f%%", "Max drawdown": "%.1f%%", "Mean invested": "%.0f%%"})
    if details:
        with st.expander("More metrics: annualised return, volatility, turnover, rejected fills, sell statistics"):
            more = {"cagr": "Annualised return", "annualised_volatility": "Volatility", "sortino_zero_mar": "Sortino", "turnover_initial_capital": "Turnover (× capital)",
                    "rejections": "Rejected fills", "max_drawdown_duration_sessions": "Longest drawdown (sessions)", "sell_fills": "Sell fills",
                    "sell_fill_win_rate": "Sell fills closed at a profit", "net_realized_sell_pnl_usd": "Realised P&L"}
            if not sharpe:
                more = {k: v for k, v in more.items() if k not in ("cagr", "sortino_zero_mar")}
            table(frame.rename(columns=more)[["Portfolio"] + list(more.values())], {"Sortino": "%.2f", "Turnover (× capital)": "%.2f", "Realised P&L": "$%.2f"},
                  percent={"Annualised return": "%+.1f%%", "Volatility": "%.1f%%", "Sell fills closed at a profit": "%.0f%%"})
            st.caption("Rejected fills: the ledger cancels a buy at the next close if the price is more than 3% above the decision close, or if filling it would breach the cash or "
                       "exposure limits (for example a portfolio already at the 80% cap once costs are deducted). The order is not retried; the next weekly rebalance issues a new one. "
                       "This is the pilot ledger's behaviour and applies to every portfolio, including the benchmark.")


def view_performance():
    headline("performance")
    summary, metrics = J("sandbox/summary.json"), T("sandbox/metrics.csv")
    if summary is None or metrics is None:
        return missing("sandbox/summary.json")
    assumptions = summary["assumptions"]
    with st.container(border=True):
        st.markdown("**Assumptions that apply to every number on this page**")
        st.markdown(
            f"- **Capital:** {usd(assumptions['capital_usd'])} paper capital per portfolio. **Data:** {assumptions['data']}.\n"
            f"- **Execution:** {assumptions['fills']}. **Costs:** {assumptions['costs']}. Idle cash earns {assumptions['cash_return'].split(' ')[0]}.\n"
            f"- **Limits:** {assumptions['limits']}.\n"
            f"- **Benchmark:** {assumptions['benchmark']}.\n"
            f"- **Sharpe ratio:** {assumptions['sharpe']}; intervals from 2,000 date-block resamples."
        )
    st.caption("Classification performance (section 4) and trading performance (this page) are separate questions with separate baselines: the prior there, equal weight here.")

    st.subheader("A. Recorded paper trading (the only recorded results)")
    inventory = J("inventory.json")
    if inventory and inventory["pilot"].get("available"):
        badge(RECORDED, "scheduler ledgers, October 2026")
        segments = pd.DataFrame(inventory["pilot"]["segments"])
        segments["pnl"] = (segments.last_equity - assumptions["capital_usd"]).map(lambda v: f"{v:+,.2f}")
        table(segments[["segment", "start", "valid_sessions", "orders", "fills", "fees", "last_equity", "pnl"]].rename(
            columns={"segment": "Attempt", "start": "Start", "valid_sessions": "Sessions recorded", "orders": "Orders", "fills": "Fills (trades)", "fees": "Costs paid", "last_equity": "Last recorded equity", "pnl": "Recorded P&L (USD)"}),
            {"Costs paid": "$%.2f", "Last recorded equity": "$%.2f"})
        st.caption("No account was marked on more than two sessions, so the recorded P&L is just the cost of the first purchases. "
                   "**There is no recorded sample on which a return, let alone a Sharpe ratio, can be judged.**")

    st.subheader("B. Registered sandbox comparison on history")
    history = summary["history"]
    st.markdown(f"**Question.** {summary['question']}")
    st.caption(f"{history['first']} to {history['last']}, {history['sessions']} sessions, {history['rebalances']} weekly rebalances. Status: {history['status']}. "
               f"Registered before any result (`data/review-v1/registration.json`, hash {summary['registration_sha256'][:12]}…).")
    metric_table(metrics, "history")
    curves = T("sandbox/curves_history.csv")
    chart(curve_chart(curves, "Equity, 2024 to September 2026, after costs"))
    st.caption("The regime overlay multiplies each weekly target by (1 − P(turbulent)) from the HMM in section 5, using only data up to the decision close.")
    spy = history["spy_reference"]
    st.caption(f"Reference only: buy-and-hold SPY returned {pct(spy['net_return'])} with Sharpe {spy['sharpe_zero_rf']:.2f} and a {spy['max_drawdown']:.1%} maximum drawdown. {spy['note']}")

    contrasts = T("sandbox/contrasts.csv")
    row = metrics.set_index(["period", "arm"])
    st.markdown("**Registered contrasts: overlay minus its parent**")
    shown = contrasts.assign(Contrast=contrasts.arm.map(ARM_LABELS) + "  minus  " + contrasts.parent.map(ARM_LABELS))[["Contrast", "sharpe_difference", "sharpe_ci_low", "sharpe_ci_high", "sharpe_holm_p", "annualised_return_difference", "verdict"]]
    table(shown.rename(columns={"sharpe_difference": "Sharpe difference", "sharpe_ci_low": "95% low", "sharpe_ci_high": "95% high", "sharpe_holm_p": "Holm p", "annualised_return_difference": "Return difference / yr", "verdict": "Verdict"}),
          {"Sharpe difference": "%+.2f", "95% low": "%+.2f", "95% high": "%+.2f", "Holm p": "%.2f"}, percent={"Return difference / yr": "%+.1f%%"})
    trend, equal = contrasts.set_index("arm").loc["trend_regime"], contrasts.set_index("arm").loc["equal_weight_regime"]
    established = (contrasts.verdict != "not established").any()
    (st.success if established else st.error)(
        f"**Result: {'see verdicts above' if established else 'not established'}.** With the overlay the trend rule's Sharpe ratio changed by {trend.sharpe_difference:+.2f} and equal weight's by {equal.sharpe_difference:+.2f}; "
        f"{'both intervals include zero' if not established else 'see the intervals'}. For the trend rule the overlay moved the maximum drawdown from {row.loc[('history', 'trend'), 'max_drawdown']:.1%} to "
        f"{row.loc[('history', 'trend_regime'), 'max_drawdown']:.1%} and the net return from {pct(row.loc[('history', 'trend'), 'net_return'])} to {pct(row.loc[('history', 'trend_regime'), 'net_return'])}. "
        + summary["decision_rule"]
    )
    benchmark = T("sandbox/versus_benchmark.csv")
    st.markdown("**Against the benchmark (equal weight, same ledger and costs)**")
    table(benchmark.assign(arm=benchmark.arm.map(ARM_LABELS))[["arm", "sharpe_difference", "sharpe_ci_low", "sharpe_ci_high", "annualised_return_difference", "return_ci_low", "return_ci_high"]].rename(
        columns={"arm": "Portfolio", "sharpe_difference": "Sharpe − benchmark", "sharpe_ci_low": "95% low", "sharpe_ci_high": "95% high", "annualised_return_difference": "Return − benchmark / yr", "return_ci_low": "Return 95% low", "return_ci_high": "Return 95% high"}),
        {c: "%+.2f" for c in ["Sharpe − benchmark", "95% low", "95% high"]},
        percent={c: "%+.1f%%" for c in ["Return − benchmark / yr", "Return 95% low", "Return 95% high"]})
    versus = benchmark.set_index("arm").loc["trend"]
    reading = audit.interval_verdict(versus.sharpe_ci_low, versus.sharpe_ci_high, lower_is_better=False)
    st.markdown(
        f"The trend rule made money on this history ({pct(row.loc[('history', 'trend'), 'net_return'])}), and so did holding everything ({pct(row.loc[('history', 'equal_weight'), 'net_return'])}). "
        f"On average it had {row.loc[('history', 'trend'), 'mean_gross_exposure']:.0%} of its capital invested against {row.loc[('history', 'equal_weight'), 'mean_gross_exposure']:.0%}, paid "
        f"{usd(row.loc[('history', 'trend'), 'combined_fees_usd'])} in costs against {usd(row.loc[('history', 'equal_weight'), 'combined_fees_usd'])}, and its Sharpe ratio is "
        f"{'not distinguishable from' if reading == 'not distinguishable' else reading + ' than'} the benchmark's. "
        "**A positive return in a rising market is not evidence of skill.**"
    )
    deflated = history["deflated"]
    with st.expander("How much of the Sharpe ratio could be luck? (deflated Sharpe ratio)"):
        rows = pd.DataFrame([dict(Portfolio=ARM_LABELS.get(k, f"Sealed September trial: {k.replace('_', ' ')}"), **{"Deflated Sharpe ratio": v}) for k, v in deflated["deflated_sharpe_ratio"].items()])
        table(rows, {"Deflated Sharpe ratio": "%.3f"})
        st.caption(f"{deflated['note']} Trials counted: {deflated['n_trials']}; best Sharpe expected from luck alone among them: {deflated['expected_best_sharpe_with_no_skill_annualised']:.2f}.")
    with st.expander("By sub-period (one continuous account, sliced)"):
        names = {"development": "Development, 2024 to 2025", "selection": "Selection, January to July 2026", "audit": "Audit, July to September 2026"}
        for key, label in names.items():
            st.markdown(f"**{label}** ({int(row.loc[(key, 'trend'), 'sessions'])} sessions)")
            metric_table(metrics, key, details=False)
        st.caption("The shorter the slice, the wider the Sharpe interval: the 2026 slices cannot rank these portfolios.")

    st.subheader("C. Simulated replay since the release")
    post = summary["post_release"]
    badge(SIMULATED, "the pilot's code replayed day by day on a November download; fills assumed at the next close")
    st.caption(f"{post['first']} to {post['last']}, {post['sessions']} sessions, {post['rebalances']} rebalances. Status: {post['status']}.")
    metric_table(metrics, "post_release", sharpe=False)
    chart(curve_chart(T("sandbox/curves_post_release.csv"), "Equity since 29 September 2026 (simulated), after costs", height=260))
    spy = post["spy_reference"]
    returns = {a: row.loc[("post_release", a), "net_return"] for a in ARMS}
    lead = "**Every invested portfolio lost money**" if max(returns.values()) < 0 else "**Results were mixed**"
    st.warning(
        f"{lead} in this window (SPY buy-and-hold: {pct(spy['net_return'])}): trend rule {pct(returns['trend'])}, equal weight {pct(returns['equal_weight'])}, "
        f"trend with overlay {pct(returns['trend_regime'])}, equal weight with overlay {pct(returns['equal_weight_regime'])}. "
        f"The overlays held less ({row.loc[('post_release', 'trend_regime'), 'mean_gross_exposure']:.0%} invested on average against {row.loc[('post_release', 'trend'), 'mean_gross_exposure']:.0%} for the trend rule) "
        "after the HMM turned turbulent in November. That is one episode, not a test. "
        f"**{post['sessions']} sessions is far too few to interpret a Sharpe ratio**, so none is shown: two months of daily returns containing a single market move cannot separate skill from noise."
    )
    replay = J("replay/summary.json")
    quality = (replay or {}).get("data_quality")
    if quality:
        second = quality["second_source"]
        with st.expander("Data quality of the November download: largest one-day moves and a second-source check"):
            moves = pd.DataFrame(quality["largest_moves"])
            columns = {"symbol": "Instrument", "session": "Session", "return": "One-day return", "second_source_return": "Second source"}
            table(moves[[c for c in columns if c in moves.columns]].rename(columns=columns), percent={"One-day return": "%+.2f%%", "Second source": "%+.2f%%"})
            if second.get("available"):
                st.caption(
                    f"All {second['rows_compared']} daily returns in the window ({second['symbols']} instruments) were compared with {second['source']}. "
                    f"Largest difference: {second['max_abs_return_difference']:.2%} of return; rows over the {second['tolerance']:.1%} tolerance: {second['rows_over_tolerance']}. "
                    "The large moves above are in both sources."
                )
            else:
                st.caption(f"A second-source comparison was not available in this build ({second.get('reason')}). The moves above come from a single public source.")

    st.subheader("What to run next")
    exposure = row.loc[("history", "trend"), "mean_gross_exposure"] / row.loc[("history", "equal_weight"), "mean_gross_exposure"]
    st.markdown(
        f"1. **The proposal's label.** Rebuild the classifier target with next-open entry and a volatility-scaled threshold, refit, then rerun exactly the comparison in section 4.\n"
        f"2. **An uninterrupted prospective run.** Move the collector to an always-on machine and log all five portfolios in shadow every session until at least 63 sessions exist. Nothing above substitutes for this.\n"
        f"3. **Exposure-matched benchmark.** The trend rule held about {exposure:.0%} of the benchmark's average exposure. Compare it with equal weight scaled to the same exposure, to separate *being less invested* from *choosing better instruments*.\n"
        f"4. **The regime as a gate input, not an exposure dial.** {'The overlay was supported on history only' if established else 'The overlay was not shown to help'}; the proposal's use of regime probabilities as inputs to a learned gate is still untested.\n\n"
        "Any further portfolio variant needs a new registration first: the budget of five portfolios and two contrasts for this review is spent."
    )
    checks = summary["history"]["parity"]
    st.success(f"Checks: this harness reproduces the sealed September trend and equal-weight accounts exactly (largest equity difference {usd(max(v['max_abs_equity_difference_usd'] for v in checks['arms'].values()))}), "
               f"and its trend portfolio equals the pilot's own reference replay over the simulated window (difference {usd(post['parity']['max_abs_equity_difference_usd'])}).")


# --------------------------------------------------------------------------
# 8 · Run the pipeline
# --------------------------------------------------------------------------

def show_run(run: dict):
    a, b, c, d = st.columns(4)
    a.metric("Session", run["session"])
    b.metric("Pass the trend rule", f"{run['decision']['trend_count']} of 22")
    c.metric("Pending orders", run["account"]["orders"])
    d.metric("Regime (diagnostic)", run["regime"]["state"].title(), delta=f"P(turbulent) {run['regime']['p_turbulent']:.3f}", delta_color="off")
    st.markdown(f"Run `{run['run_id']}` finished {when(run['completed_at'])} in {run['seconds']:.0f} s. Data: {run['data']['provider']}, retrieved {when(run['data']['retrieved_at'])}. Models: {run['models']}.")
    steps = pd.DataFrame(run["steps"])
    steps["Result"] = steps.status.map({"ok": "✓ ok", "failed": "✗ failed"})
    table(steps[["title", "Result", "seconds", "detail"]].rename(columns={"title": "Step", "seconds": "Seconds", "detail": "What happened"}), {"Seconds": "%.1f"})
    st.markdown("**Signals and orders**")
    signals = pd.DataFrame(run["signals"])
    signals["Trend"] = signals.trend.map({True: "on", False: "off"})
    signals["Order"] = signals.order_side.map(lambda v: "none" if pd.isna(v) else str(v))
    signals["Quantity"] = signals.order_quantity.map(lambda v: "" if pd.isna(v) else f"{v:.6f}")
    signals["Reserve price"] = signals.order_limit_price.map(lambda v: "" if pd.isna(v) else f"{v:,.2f}")
    table(signals[["symbol", "close", "Trend", "target_weight", "Order", "Quantity", "Reserve price", "top_class", "p_down", "p_flat", "p_up", "overlay_weight"]].rename(
        columns={"symbol": "Instrument", "close": "Close", "target_weight": "Target weight",
                 "top_class": "Classifier top class", "p_down": "P(DOWN)", "p_flat": "P(FLAT)", "p_up": "P(UP)", "overlay_weight": "Overlay weight (not executed)"}),
        {"Close": "%.2f"}, height=420,
        percent={"Target weight": "%.2f%%", "P(DOWN)": "%.0f%%", "P(FLAT)": "%.0f%%", "P(UP)": "%.0f%%", "Overlay weight (not executed)": "%.2f%%"})
    st.caption("The order columns come from the trend rule and the risk limits. The classifier columns are shown beside them for comparison only: no order depends on them.")
    shadow = run["account"]["shadow"]
    st.markdown(
        f"- Demo shadow account: equity {usd(shadow['equity'])}, cash reserved for the pending buys {usd(shadow['reserved_cash'])}, invested {shadow['gross_exposure']:.0%} (orders have not filled).\n"
        f"- Target gross exposure {run['decision']['target_gross']:.1%}. With the sandbox regime overlay it would be {run['decision']['overlay_gross']:.1%} today (shown for comparison, not executed).\n"
        f"- {run['account']['note']}\n"
        f"- Deployment account: cash. External orders: none."
    )


def view_live():
    from review_v1 import live

    top = st.empty()  # filled after the button, so the summary always describes the run shown below
    badge(SIMULATED, "a demo run in an isolated account; it stops at pending orders")
    st.markdown(
        "The button runs the same code path as the scheduled pilot on the latest completed session: download, validate, features, models, trend rule, risk limits, ledger. "
        "It writes only to `data/paper/review-v1/live/`, starts from cash each time, simulates no fill and sends nothing to a broker. It takes about half a minute."
    )
    if st.button("Run the pipeline now", type="primary"):
        with st.status("Running the pipeline…", expanded=True) as status:
            def progress(identifier, title, state, detail):
                if state == "ok":
                    st.write(f"✓ **{title}**: {detail}")
                elif state == "failed":
                    st.write(f"✗ **{title}**: {detail}")
            try:
                live.run(progress=progress)
                status.update(label="Pipeline run complete", state="complete")
            except Exception as exc:  # shown, never hidden
                status.update(label=f"Pipeline run failed: {type(exc).__name__}", state="error")
                st.error(f"{type(exc).__name__}: {exc}. The last saved run below is unchanged.")
        _json.clear()
    run = live.last_run()
    with top.container():
        headline("live", live.caption(run))
    st.subheader("Last saved run")
    if run:
        show_run(run)
    else:
        st.info("No run has been saved yet. Press the button above.")


# --------------------------------------------------------------------------
# 9 · Report and evidence
# --------------------------------------------------------------------------

def view_report():
    headline("report")
    manifest = J("manifest.json")
    if manifest is None:
        return missing("manifest.json")
    tabs = st.tabs(["Audit", "Checks", "Progress report", "Reproduce", "Registration"])
    with tabs[0]:
        text = {"works": "Works, verified", "diagnostic": "Diagnostic only", "unvalidated": "Exists, not validated", "missing": "Missing"}
        counts = manifest["audit_summary"]
        cols = st.columns(4)
        for column, status in zip(cols, text, strict=True):
            column.metric(text[status], counts[status])
        for status in text:
            st.markdown(f"#### {text[status]}")
            prose_table(["Component", "Finding", "How to verify"],
                        [(f"**{r['component']}**", r["finding"], r["verify"]) for r in manifest["audit"] if r["status"] == status])
    with tabs[1]:
        failed = [c for c in manifest["checks"] if not c["passed"]]
        (st.error if failed else st.success)(f"{len(manifest['checks']) - len(failed)} of {len(manifest['checks'])} checks passed" + (f"; failed: {', '.join(c['check'] for c in failed)}." if failed else "."))
        prose_table(["Check", "Result", "Detail"], [(c["check"], "✓\u00a0passed" if c["passed"] else "✗\u00a0FAILED", c["detail"]) for c in manifest["checks"]])
        st.caption("These are recomputed by `python -m review_v1.build manifest`; the build exits non-zero if any fails.")
    with tabs[2]:
        path = ROOT / "docs/review-v1/PROGRESS_REPORT.md"
        st.markdown(prose(path.read_text()) if path.exists() else "The report has not been generated yet: run `python -m review_v1.build report`.")
    with tabs[3]:
        path = ROOT / "docs/review-v1/REPRODUCE.md"
        st.markdown(path.read_text() if path.exists() else "See `docs/review-v1/REPRODUCE.md`.")
    with tabs[4]:
        registration = J("registration.json")
        st.markdown(f"Registered {when(registration['registered_at'])} before any new evaluation. SHA-256 `{manifest['registration_sha256']}`.")
        amendments = J("registration_amendments.json") or []
        for item in amendments:
            st.markdown(f"**Amendment {item['id']}** ({when(item['recorded_at'])}, made before {item['made_before']}). {item['change']} *Reason:* {item['reason']}")
        st.json(registration, expanded=1)


VIEWS = dict(overview=view_overview, data=view_data, walkthrough=view_walkthrough, classifiers=view_classifiers, regimes=view_regimes,
             sentiment=view_sentiment, performance=view_performance, live=view_live, report=view_report)


def main():
    st.set_page_config(page_title="ML TradeEngine | Progress review", layout="wide")
    st.html(CSS)
    view = select_view()
    if view != "walkthrough":
        st.query_params.from_dict(dict(view=view))
    state = VIEWS[view]() or {}
    ready_marker(view, **state)


if __name__ == "__main__":
    main()
