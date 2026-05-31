"""
Phase 07 — ML Trade Engine Dashboard

Interactive Streamlit app covering:
  1. Account & model overview — key metrics at a glance
  2. Equity curves — strategy vs buy-and-hold, per ticker
  3. Rolling performance — rolling Sharpe and max drawdown windows
  4. Live signals — most recent signal log from Phase 06E SQLite
  5. Strategy ranking — Phase 06B unified scorecard
  6. Feature importance — XGBoost feature importances per ticker
  7. Trade log — per-trade P&L, win/loss streak

Run locally:
  streamlit run dashboard.py

Deploy to Streamlit Community Cloud:
  1. Push repo to GitHub (data/ excluded via .gitignore except JSON summaries)
  2. Connect repo at share.streamlit.io → select dashboard.py

Requirements:
  pip install streamlit plotly pandas pyarrow
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT         = Path(__file__).resolve().parent
_BACKTEST_DIR = _ROOT / "data" / "backtest"
_RESEARCH_DIR = _ROOT / "data" / "research"
_MODELS_DIR   = _ROOT / "data" / "models"
_RISK_DIR     = _ROOT / "data" / "risk"
_LIVE_DB      = _ROOT / "data" / "live" / "signal_log.db"

_PALETTE = {
    "strategy":  "#00C9A7",
    "benchmark": "#8884D8",
    "up":        "#10B981",
    "down":      "#EF4444",
    "flat":      "#6B7280",
    "accent":    "#3B82F6",
    "warn":      "#F59E0B",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title  = "ML Trade Engine",
    page_icon   = "📈",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# Minimal CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    section[data-testid="stSidebar"] { background: #0F172A; }
    section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    div[data-testid="metric-container"] {
        background: #1E293B; border-radius: 8px; padding: 12px 16px;
    }
    h1, h2, h3 { color: #E2E8F0; }
    .stDataFrame { border-radius: 8px; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loaders (all cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def available_tickers() -> list[str]:
    if not _BACKTEST_DIR.exists():
        return []
    return sorted(p.name for p in _BACKTEST_DIR.iterdir() if p.is_dir())


@st.cache_data(ttl=300)
def load_equity_curves(ticker: str) -> Optional[pd.DataFrame]:
    path = _BACKTEST_DIR / ticker / "equity_curves.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


@st.cache_data(ttl=300)
def load_trade_log(ticker: str, source: str = "backtest") -> Optional[pd.DataFrame]:
    root = _BACKTEST_DIR if source == "backtest" else _RISK_DIR
    path = root / ticker / "trade_log.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


@st.cache_data(ttl=300)
def load_backtest_summary(ticker: str) -> dict:
    path = _BACKTEST_DIR / ticker / "backtest_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_data(ttl=300)
def load_risk_summary(ticker: str) -> dict:
    path = _RISK_DIR / ticker / "risk_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_data(ttl=300)
def load_scorecard() -> Optional[pd.DataFrame]:
    path = _RESEARCH_DIR / "scorecard.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data(ttl=300)
def load_feature_importance(ticker: str) -> Optional[pd.DataFrame]:
    """
    Load XGBoost feature importances from the final model.
    Falls back to feature_cols.json ordering if model weight extraction fails.
    """
    try:
        import xgboost as xgb
        model_path = _MODELS_DIR / ticker / "xgb_final.json"
        feat_path  = _MODELS_DIR / ticker / "feature_cols.json"
        if not model_path.exists() or not feat_path.exists():
            return None

        booster = xgb.Booster()
        booster.load_model(str(model_path))
        feat_cols = json.loads(feat_path.read_text())
        scores    = booster.get_score(importance_type="gain")

        rows = [
            {"feature": f, "importance": scores.get(f, 0.0)}
            for f in feat_cols
        ]
        df = pd.DataFrame(rows).sort_values("importance", ascending=False)
        return df.head(20)
    except Exception:
        return None


@st.cache_data(ttl=60)
def load_live_signals(n: int = 50) -> Optional[pd.DataFrame]:
    if not _LIVE_DB.exists():
        return None
    try:
        conn = sqlite3.connect(_LIVE_DB)
        df = pd.read_sql(
            f"SELECT * FROM signals ORDER BY id DESC LIMIT {n}", conn
        )
        conn.close()
        return df
    except Exception:
        return None


@st.cache_data(ttl=60)
def load_live_orders(n: int = 50) -> Optional[pd.DataFrame]:
    if not _LIVE_DB.exists():
        return None
    try:
        conn = sqlite3.connect(_LIVE_DB)
        df = pd.read_sql(
            f"SELECT * FROM orders ORDER BY id DESC LIMIT {n}", conn
        )
        conn.close()
        return df
    except Exception:
        return None


@st.cache_data(ttl=60)
def load_equity_history() -> Optional[pd.DataFrame]:
    if not _LIVE_DB.exists():
        return None
    try:
        conn = sqlite3.connect(_LIVE_DB)
        df = pd.read_sql("SELECT * FROM equity ORDER BY id ASC", conn)
        conn.close()
        if not df.empty:
            df["run_ts"] = pd.to_datetime(df["run_ts"], utc=True)
        return df
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _rolling_sharpe(returns: pd.Series, window: int = 63, rfr: float = 0.05) -> pd.Series:
    excess = returns - rfr / 252
    roll   = excess.rolling(window)
    return (roll.mean() / roll.std().replace(0, float("nan"))) * (252 ** 0.5)


def _rolling_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (equity - peak) / peak


def equity_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    if "strategy" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["strategy"],
            name="ML Strategy",
            line=dict(color=_PALETTE["strategy"], width=2),
        ))
    if "buy_hold" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["buy_hold"],
            name="Buy & Hold",
            line=dict(color=_PALETTE["benchmark"], width=1.5, dash="dot"),
        ))
    fig.update_layout(
        title       = f"{ticker} — Equity Curve",
        xaxis_title = "Date",
        yaxis_title = "Portfolio Value ($)",
        legend      = dict(x=0.01, y=0.99),
        hovermode   = "x unified",
        template    = "plotly_dark",
        height      = 380,
    )
    return fig


def rolling_chart(equity: pd.Series, ticker: str) -> go.Figure:
    returns = equity.pct_change().fillna(0)
    sharpe  = _rolling_sharpe(returns)
    dd      = _rolling_drawdown(equity) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sharpe.index, y=sharpe,
        name="Rolling Sharpe (63d)",
        line=dict(color=_PALETTE["accent"], width=2),
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd,
        name="Drawdown (%)",
        line=dict(color=_PALETTE["down"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.15)",
        yaxis="y2",
    ))
    fig.add_hline(y=1.0, line=dict(color=_PALETTE["up"], dash="dash", width=1),
                  annotation_text="Sharpe=1", yref="y")
    fig.update_layout(
        title   = f"{ticker} — Rolling Sharpe & Drawdown",
        xaxis   = dict(title="Date"),
        yaxis   = dict(title="Rolling Sharpe"),
        yaxis2  = dict(title="Drawdown (%)", overlaying="y", side="right"),
        hovermode  = "x unified",
        legend     = dict(x=0.01, y=0.99),
        template   = "plotly_dark",
        height     = 340,
    )
    return fig


def trade_pnl_chart(trade_df: pd.DataFrame) -> go.Figure:
    exits = trade_df[trade_df["type"] == "exit"].copy()
    if exits.empty:
        return go.Figure()
    exits = exits.dropna(subset=["pnl"])
    exits["color"] = exits["pnl"].apply(
        lambda x: _PALETTE["up"] if x >= 0 else _PALETTE["down"]
    )
    exits["cum_pnl"] = exits["pnl"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=exits.get("date", exits.index),
        y=exits["pnl"],
        name="Trade P&L",
        marker_color=exits["color"],
    ))
    fig.add_trace(go.Scatter(
        x=exits.get("date", exits.index),
        y=exits["cum_pnl"],
        name="Cumulative P&L",
        line=dict(color=_PALETTE["accent"], width=2),
        yaxis="y2",
    ))
    fig.update_layout(
        title    = "Trade P&L",
        xaxis    = dict(title="Date"),
        yaxis    = dict(title="P&L per Trade ($)"),
        yaxis2   = dict(title="Cumulative P&L ($)", overlaying="y", side="right"),
        template = "plotly_dark",
        height   = 320,
        barmode  = "relative",
    )
    return fig


def shap_chart(feat_imp: pd.DataFrame) -> go.Figure:
    top = feat_imp.head(20)
    fig = px.bar(
        top.iloc[::-1],
        x       = "importance",
        y       = "feature",
        orientation = "h",
        title   = "Top 20 Feature Importances (XGBoost gain)",
        color   = "importance",
        color_continuous_scale = "Teal",
        template = "plotly_dark",
        height  = 480,
    )
    fig.update_coloraxes(showscale=False)
    fig.update_layout(yaxis_title="", xaxis_title="Gain")
    return fig


def signal_color(label: str) -> str:
    return {"UP": "🟢", "DOWN": "🔴", "FLAT": "⚪"}.get(label, "⚫")


# ---------------------------------------------------------------------------
# Metric formatters
# ---------------------------------------------------------------------------

def fmt_pct(v) -> str:
    try:
        return f"{float(v):.1%}"
    except Exception:
        return "—"


def fmt_f2(v) -> str:
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "—"


def fmt_dollar(v) -> str:
    try:
        return f"${float(v):,.0f}"
    except Exception:
        return "—"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("📈 ML Trade Engine")
st.sidebar.markdown("---")

pages = [
    "🏠 Overview",
    "📊 Equity Curves",
    "🔄 Rolling Performance",
    "⚡ Live Signals",
    "🏆 Strategy Ranking",
    "🧬 Feature Importance",
    "📋 Trade Log",
]
page = st.sidebar.radio("Navigate", pages, index=0)

tickers = available_tickers()
if not tickers:
    tickers = ["AAPL"]

ticker = st.sidebar.selectbox("Ticker", tickers, index=0) if tickers else "AAPL"

st.sidebar.markdown("---")
st.sidebar.caption("Phase 07 — Dashboard  |  ML Trade Engine")


# ===========================================================================
# PAGE: Overview
# ===========================================================================

if page == "🏠 Overview":
    st.title("ML Trade Engine — Overview")

    summary = load_backtest_summary(ticker)
    risk    = load_risk_summary(ticker)

    sm = summary.get("strategy_metrics", {})
    rm = risk.get("risk_metrics", {})

    # ── Top metrics row ─────────────────────────────────────────────────
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Sharpe Ratio",  fmt_f2(sm.get("sharpe_ratio")),
                delta=fmt_f2(rm.get("sharpe_ratio", 0) - sm.get("sharpe_ratio", 0)) + " (risk)")
    col2.metric("CAGR",         fmt_pct(sm.get("cagr")))
    col3.metric("Max Drawdown", fmt_pct(sm.get("max_drawdown")))
    col4.metric("Win Rate",     fmt_pct(sm.get("win_rate")))
    col5.metric("Calmar Ratio", fmt_f2(sm.get("calmar_ratio")))
    col6.metric("# Trades",     str(int(sm.get("n_trades", 0)) if sm.get("n_trades") else "—"))

    st.markdown("---")

    # ── Mini equity chart ────────────────────────────────────────────────
    eq_df = load_equity_curves(ticker)
    if eq_df is not None:
        st.plotly_chart(equity_chart(eq_df, ticker), use_container_width=True)
    else:
        st.info(f"No equity curve found for {ticker}. Run `python3 backtest.py` first.")

    # ── Phase target status ──────────────────────────────────────────────
    st.subheader("Performance Targets")
    targets = [
        ("Sharpe Ratio",   sm.get("sharpe_ratio", 0),  "> 1.0",  lambda v: v > 1.0),
        ("Max Drawdown",   abs(sm.get("max_drawdown", 1)), "< 20%", lambda v: v < 0.20),
        ("Win Rate",       sm.get("win_rate", 0),       "> 55%",  lambda v: v > 0.55),
        ("CAGR",           sm.get("cagr", 0),           "> 15%",  lambda v: v > 0.15),
        ("Calmar Ratio",   sm.get("calmar_ratio", 0),   "> 0.5",  lambda v: v > 0.5),
    ]
    t_cols = st.columns(5)
    for col, (name, val, target_str, check) in zip(t_cols, targets):
        ok = check(val) if val else False
        icon = "✅" if ok else "⏳"
        col.metric(f"{icon} {name}", fmt_f2(val) if isinstance(val, float) else "—",
                   delta=target_str)

    st.markdown("---")

    # ── Monte Carlo summary ──────────────────────────────────────────────
    mc = summary.get("monte_carlo", {})
    if mc:
        st.subheader("Monte Carlo (1000 paths)")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Sharpe p5",     fmt_f2(mc.get("sharpe_p5")))
        mc2.metric("Sharpe median", fmt_f2(mc.get("sharpe_median")))
        mc3.metric("Sharpe p95",    fmt_f2(mc.get("sharpe_p95")))
        mc4.metric("Prob(ruin)",    fmt_pct(mc.get("prob_ruin")))

    # ── Risk engine metrics ──────────────────────────────────────────────
    if rm:
        st.subheader(f"Risk Engine Metrics  ({ticker})")
        var_rep = risk.get("var_report", {})
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("VaR 95%",    fmt_pct(var_rep.get("var_95_pct")))
        r2.metric("CVaR 99%",   fmt_pct(var_rep.get("cvar_99_pct")))
        r3.metric("Sharpe (w/ risk)", fmt_f2(rm.get("sharpe_ratio")))
        r4.metric("CAGR (w/ risk)",   fmt_pct(rm.get("cagr")))

    # ── Live account equity ──────────────────────────────────────────────
    eq_hist = load_equity_history()
    if eq_hist is not None and not eq_hist.empty:
        st.subheader("Paper Account Equity")
        fig_eq = px.line(
            eq_hist, x="run_ts", y="equity",
            title="Alpaca Paper Account Equity",
            template="plotly_dark",
            color_discrete_sequence=[_PALETTE["accent"]],
            height=280,
        )
        st.plotly_chart(fig_eq, use_container_width=True)


# ===========================================================================
# PAGE: Equity Curves
# ===========================================================================

elif page == "📊 Equity Curves":
    st.title(f"Equity Curves — {ticker}")

    eq_df = load_equity_curves(ticker)
    if eq_df is None:
        st.warning(f"No equity curve data for {ticker}. Run `python3 backtest.py` first.")
    else:
        # Multi-ticker overlay
        multi = st.checkbox("Compare multiple tickers", value=False)
        if multi:
            selected = st.multiselect("Select tickers", tickers, default=tickers[:3])
            fig = go.Figure()
            for t in selected:
                df = load_equity_curves(t)
                if df is not None and "strategy" in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df["strategy"] / df["strategy"].iloc[0] * 100,
                        name=t, mode="lines",
                    ))
            fig.update_layout(
                title    = "Normalised Equity — Strategy (base=100)",
                xaxis_title = "Date",
                yaxis_title = "Indexed Equity",
                template = "plotly_dark",
                height   = 450,
                hovermode = "x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.plotly_chart(equity_chart(eq_df, ticker), use_container_width=True)

            # Return distribution
            if "strategy" in eq_df.columns:
                rets = eq_df["strategy"].pct_change().dropna() * 100
                fig_hist = px.histogram(
                    rets, nbins=60,
                    title   = f"{ticker} — Daily Return Distribution (%)",
                    labels  = {"value": "Daily Return (%)"},
                    template = "plotly_dark",
                    color_discrete_sequence = [_PALETTE["strategy"]],
                    height  = 260,
                )
                fig_hist.add_vline(x=0, line=dict(color="white", dash="dash", width=1))
                st.plotly_chart(fig_hist, use_container_width=True)

    # ── MC percentile chart ──────────────────────────────────────────────
    mc_path = _BACKTEST_DIR / ticker / "mc_percentiles.parquet"
    if mc_path.exists():
        mc_df = pd.read_parquet(mc_path)
        st.subheader("Monte Carlo Paths (1000×)")
        fig_mc = go.Figure()
        for col, color, name in [
            ("p5",   "#EF4444", "p5"),
            ("p50",  "#3B82F6", "Median"),
            ("p95",  "#10B981", "p95"),
        ]:
            if col in mc_df.columns:
                fig_mc.add_trace(go.Scatter(
                    x=mc_df.index, y=mc_df[col],
                    name=name, line=dict(color=color, width=2),
                ))
        if "p5" in mc_df.columns and "p95" in mc_df.columns:
            fig_mc.add_trace(go.Scatter(
                x=pd.concat([mc_df.index.to_series(), mc_df.index.to_series()[::-1]]),
                y=pd.concat([mc_df["p95"], mc_df["p5"][::-1]]),
                fill="toself",
                fillcolor="rgba(59,130,246,0.1)",
                line=dict(color="rgba(0,0,0,0)"),
                name="p5–p95 band",
                showlegend=True,
            ))
        fig_mc.update_layout(
            title    = f"{ticker} — Monte Carlo Equity Projection",
            xaxis_title = "Trading Days Forward",
            yaxis_title = "Portfolio Value ($)",
            template = "plotly_dark",
            height   = 350,
        )
        st.plotly_chart(fig_mc, use_container_width=True)


# ===========================================================================
# PAGE: Rolling Performance
# ===========================================================================

elif page == "🔄 Rolling Performance":
    st.title(f"Rolling Performance — {ticker}")

    eq_df = load_equity_curves(ticker)
    if eq_df is None:
        st.warning("No equity data found.")
    else:
        if "strategy" in eq_df.columns:
            window = st.slider("Rolling window (days)", 21, 252, 63, step=21)
            returns = eq_df["strategy"].pct_change().fillna(0)

            sharpe = _rolling_sharpe(returns, window=window)
            dd     = _rolling_drawdown(eq_df["strategy"]) * 100

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sharpe.index, y=sharpe.round(3),
                name=f"Rolling Sharpe ({window}d)",
                line=dict(color=_PALETTE["accent"], width=2),
                yaxis="y",
            ))
            fig.add_trace(go.Scatter(
                x=dd.index, y=dd.round(2),
                name="Drawdown (%)",
                line=dict(color=_PALETTE["down"], width=1.5),
                fill="tozeroy",
                fillcolor="rgba(239,68,68,0.15)",
                yaxis="y2",
            ))
            fig.add_hline(y=1.0, line=dict(color=_PALETTE["up"], dash="dash", width=1),
                          annotation_text="Sharpe=1.0", yref="y")
            fig.add_hline(y=-15, line=dict(color=_PALETTE["warn"], dash="dash", width=1),
                          annotation_text="-15% CB", yref="y2")
            fig.update_layout(
                xaxis   = dict(title="Date"),
                yaxis   = dict(title=f"Rolling Sharpe ({window}d)"),
                yaxis2  = dict(title="Drawdown (%)", overlaying="y", side="right"),
                hovermode  = "x unified",
                legend     = dict(x=0.01, y=0.99),
                template   = "plotly_dark",
                height     = 400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Rolling volatility
            roll_vol = returns.rolling(window).std() * (252 ** 0.5) * 100
            fig_vol = px.area(
                x=roll_vol.index, y=roll_vol.round(2),
                title=f"{ticker} — Rolling Annualised Volatility ({window}d, %)",
                labels={"x": "Date", "y": "Vol (%)"},
                template="plotly_dark",
                color_discrete_sequence=[_PALETTE["warn"]],
                height=250,
            )
            st.plotly_chart(fig_vol, use_container_width=True)


# ===========================================================================
# PAGE: Live Signals
# ===========================================================================

elif page == "⚡ Live Signals":
    st.title("Live Signals — Phase 06E")

    if not _LIVE_DB.exists():
        st.info(
            "No live signal log found. Run `python3 live.py run --dry-run` to generate signals."
        )
    else:
        signals = load_live_signals(100)
        orders  = load_live_orders(50)

        if signals is not None and not signals.empty:
            # ── Signal summary cards ─────────────────────────────────────
            latest = signals.groupby("ticker").first().reset_index()
            st.subheader(f"Latest Signal per Ticker  ({len(latest)} tickers)")

            n_cols = min(len(latest), 5)
            cols   = st.columns(n_cols)
            for i, (_, row) in enumerate(latest.iterrows()):
                icon  = signal_color(row.get("label", ""))
                col   = cols[i % n_cols]
                col.metric(
                    label = f"{icon} {row['ticker']}",
                    value = row.get("label", "—"),
                    delta = f"conf={row.get('confidence', 0):.1%}",
                )

            st.markdown("---")

            # ── Signal log table ─────────────────────────────────────────
            st.subheader("Signal Log")
            show_cols = ["ticker", "date", "label", "confidence", "kelly_frac",
                         "close", "stop_loss", "atr", "run_ts"]
            display   = signals[[c for c in show_cols if c in signals.columns]]

            def _color_label(val):
                colors = {"UP": "color: #10B981", "DOWN": "color: #EF4444",
                          "FLAT": "color: #6B7280"}
                return colors.get(val, "")

            if "label" in display.columns:
                styled = display.style.applymap(_color_label, subset=["label"])
                st.dataframe(styled, use_container_width=True, height=400)
            else:
                st.dataframe(display, use_container_width=True, height=400)

            # ── Signal direction bar chart ────────────────────────────────
            if "label" in signals.columns:
                dist = signals["label"].value_counts().reset_index()
                dist.columns = ["Direction", "Count"]
                fig_dist = px.bar(
                    dist, x="Direction", y="Count",
                    color="Direction",
                    color_discrete_map={
                        "UP": _PALETTE["up"],
                        "DOWN": _PALETTE["down"],
                        "FLAT": _PALETTE["flat"],
                    },
                    title="Signal Direction Distribution",
                    template="plotly_dark",
                    height=260,
                )
                st.plotly_chart(fig_dist, use_container_width=True)

        else:
            st.info("No signals logged yet.")

        # ── Orders table ─────────────────────────────────────────────────
        if orders is not None and not orders.empty:
            st.subheader("Order Log")
            order_cols = ["ticker", "side", "qty", "status", "stop_price",
                          "take_profit", "error", "run_ts"]
            st.dataframe(
                orders[[c for c in order_cols if c in orders.columns]],
                use_container_width=True,
                height=300,
            )

        # ── Live account equity chart ─────────────────────────────────────
        eq_hist = load_equity_history()
        if eq_hist is not None and not eq_hist.empty:
            st.subheader("Paper Account Equity")
            fig = px.line(
                eq_hist, x="run_ts", y="equity",
                title="Paper Account Equity Over Time",
                template="plotly_dark",
                color_discrete_sequence=[_PALETTE["accent"]],
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE: Strategy Ranking
# ===========================================================================

elif page == "🏆 Strategy Ranking":
    st.title("Strategy Ranking — Phase 06B")

    scorecard = load_scorecard()
    if scorecard is None:
        st.warning("No scorecard found. Run `python3 research.py zoo && python3 research.py rank` first.")
    else:
        scorecard = scorecard.reset_index()

        # ── Filter controls ───────────────────────────────────────────────
        fcol1, fcol2 = st.columns([1, 3])
        with fcol1:
            min_sharpe = st.number_input("Min Sharpe", value=0.0, step=0.1)
        with fcol2:
            sort_col = st.selectbox(
                "Sort by",
                [c for c in ["sharpe_ratio", "cagr", "calmar_ratio", "win_rate", "max_drawdown"]
                 if c in scorecard.columns],
            )

        filtered = scorecard
        if "sharpe_ratio" in scorecard.columns:
            filtered = scorecard[scorecard["sharpe_ratio"] >= min_sharpe]
        filtered = filtered.sort_values(sort_col, ascending=(sort_col == "max_drawdown"))

        # ── Metrics columns ───────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Strategies",   len(filtered))
        m2.metric("Avg Sharpe",   fmt_f2(filtered["sharpe_ratio"].mean()) if "sharpe_ratio" in filtered.columns else "—")
        m3.metric("Best Sharpe",  fmt_f2(filtered["sharpe_ratio"].max()) if "sharpe_ratio" in filtered.columns else "—")
        m4.metric("Best CAGR",    fmt_pct(filtered["cagr"].max()) if "cagr" in filtered.columns else "—")

        st.markdown("---")

        # ── Ranked table ──────────────────────────────────────────────────
        display_cols = ["ticker", "strategy", "sharpe_ratio", "cagr", "max_drawdown",
                        "calmar_ratio", "win_rate", "n_trades"]
        show = [c for c in display_cols if c in filtered.columns]
        st.dataframe(
            filtered[show].style.format({
                "sharpe_ratio":  "{:.2f}",
                "cagr":          "{:.1%}",
                "max_drawdown":  "{:.1%}",
                "calmar_ratio":  "{:.2f}",
                "win_rate":      "{:.1%}",
            }),
            use_container_width=True,
            height=500,
        )

        # ── Sharpe distribution ───────────────────────────────────────────
        if "sharpe_ratio" in scorecard.columns and "strategy" in scorecard.columns:
            fig_box = px.box(
                scorecard, x="strategy", y="sharpe_ratio",
                title="Sharpe Ratio Distribution by Strategy",
                template="plotly_dark",
                color="strategy",
                height=380,
            )
            fig_box.add_hline(y=1.0, line=dict(color=_PALETTE["up"], dash="dash"),
                              annotation_text="Target Sharpe=1.0")
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        # ── CAGR vs Sharpe scatter ────────────────────────────────────────
        if {"sharpe_ratio", "cagr", "max_drawdown"}.issubset(scorecard.columns):
            fig_sc = px.scatter(
                scorecard, x="sharpe_ratio", y="cagr",
                color="strategy",
                size=scorecard["max_drawdown"].abs().clip(0.01, 0.5),
                hover_data=["ticker", "win_rate", "calmar_ratio"],
                title="CAGR vs Sharpe Ratio  (bubble = |Max Drawdown|)",
                template="plotly_dark",
                height=420,
                labels={"sharpe_ratio": "Sharpe", "cagr": "CAGR"},
            )
            fig_sc.add_vline(x=1.0, line=dict(color=_PALETTE["up"], dash="dash"))
            fig_sc.add_hline(y=0.15, line=dict(color=_PALETTE["warn"], dash="dash"))
            st.plotly_chart(fig_sc, use_container_width=True)


# ===========================================================================
# PAGE: Feature Importance
# ===========================================================================

elif page == "🧬 Feature Importance":
    st.title(f"Feature Importance — {ticker}")

    feat_imp = load_feature_importance(ticker)
    if feat_imp is None:
        st.warning(
            f"Could not load feature importance for {ticker}. "
            "Ensure XGBoost models are trained (`python3 train.py`)."
        )
    else:
        st.plotly_chart(shap_chart(feat_imp), use_container_width=True)

        st.subheader("Feature Importance Table")
        st.dataframe(
            feat_imp.style.format({"importance": "{:.1f}"}),
            use_container_width=True,
            height=500,
        )

    # ── Model info ────────────────────────────────────────────────────────
    train_results_path = _MODELS_DIR / ticker / "training_results.json"
    if train_results_path.exists():
        tr = json.loads(train_results_path.read_text())
        oos = tr.get("oos_metrics", {})
        st.subheader("OOS Training Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("OOS Accuracy",   fmt_pct(oos.get("accuracy")))
        c2.metric("OOS F1",         fmt_f2(oos.get("f1_macro")))
        c3.metric("N Folds",        str(tr.get("n_folds", "—")))


# ===========================================================================
# PAGE: Trade Log
# ===========================================================================

elif page == "📋 Trade Log":
    st.title(f"Trade Log — {ticker}")

    source = st.radio("Data source", ["Backtest (Phase 04)", "Risk Engine (Phase 05)"],
                      horizontal=True)
    src_key = "backtest" if "Backtest" in source else "risk"

    trade_df = load_trade_log(ticker, source=src_key)

    if trade_df is None:
        st.warning(f"No trade log found for {ticker} ({src_key}). Run the pipeline first.")
    else:
        entries = trade_df[trade_df["type"] == "entry"]
        exits   = trade_df[trade_df["type"] == "exit"]
        closed  = exits.dropna(subset=["pnl"])

        # ── Summary metrics ──────────────────────────────────────────────
        n_trades   = len(entries)
        wins       = (closed["pnl"] > 0).sum()
        losses     = (closed["pnl"] <= 0).sum()
        win_rate   = wins / len(closed) if len(closed) > 0 else 0
        avg_win    = closed.loc[closed["pnl"] > 0, "pnl"].mean() if wins else 0
        avg_loss   = closed.loc[closed["pnl"] <= 0, "pnl"].mean() if losses else 0
        total_pnl  = closed["pnl"].sum()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Trades",  str(n_trades))
        c2.metric("Win Rate",      fmt_pct(win_rate))
        c3.metric("Avg Win",       fmt_dollar(avg_win))
        c4.metric("Avg Loss",      fmt_dollar(avg_loss))
        c5.metric("Total P&L",     fmt_dollar(total_pnl),
                  delta="profit" if total_pnl >= 0 else "loss")

        st.markdown("---")

        # ── P&L chart ─────────────────────────────────────────────────────
        st.plotly_chart(trade_pnl_chart(trade_df), use_container_width=True)

        # ── Win/loss streak ───────────────────────────────────────────────
        if len(closed) > 0:
            closed_sorted = closed.sort_values("date") if "date" in closed.columns else closed
            is_win        = (closed_sorted["pnl"] > 0).astype(int)
            streak        = is_win.groupby((is_win != is_win.shift()).cumsum()).cumcount() + 1
            closed_sorted["streak"] = streak.values
            closed_sorted["result"] = is_win.map({1: "Win", 0: "Loss"})

            fig_streak = px.bar(
                closed_sorted.tail(50),
                x=closed_sorted.tail(50).get("date", closed_sorted.tail(50).index),
                y="pnl",
                color="result",
                color_discrete_map={"Win": _PALETTE["up"], "Loss": _PALETTE["down"]},
                title="Last 50 Trades — Win/Loss Streak",
                template="plotly_dark",
                height=280,
            )
            st.plotly_chart(fig_streak, use_container_width=True)

        # ── Full trade log table ──────────────────────────────────────────
        st.subheader("Full Trade Log")
        show_cols = ["date", "type", "reason", "price", "entry_price",
                     "stop_level", "pnl"]
        table_cols = [c for c in show_cols if c in trade_df.columns]
        fmt_map    = {"price": "{:.2f}", "entry_price": "{:.2f}",
                      "pnl": "{:.2f}", "stop_level": "{:.2f}"}

        st.dataframe(
            trade_df[table_cols].style.format(
                {k: v for k, v in fmt_map.items() if k in table_cols}
            ),
            use_container_width=True,
            height=500,
        )
