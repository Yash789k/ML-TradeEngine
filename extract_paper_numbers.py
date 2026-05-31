"""
Paper Number Extractor — Phase 06D

Reads every artifact produced by the pipeline and prints all [FILL] values
needed for PHASE_06D_PAPER_DRAFT.md in a structured, copy-paste-ready format.

Run AFTER the full pipeline has completed:
    python3 extract_paper_numbers.py

Outputs are grouped by paper section.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT         = Path(__file__).resolve().parent
BACKTEST_DIR = ROOT / "data" / "backtest"
RESEARCH_DIR = ROOT / "data" / "research"
RISK_DIR     = ROOT / "data" / "risk"
MODELS_DIR   = ROOT / "data" / "models"
LIVE_DB      = ROOT / "data" / "live" / "signal_log.db"
ENV_DIR      = RESEARCH_DIR / "env"

SPY_RF = 0.05   # risk-free rate used throughout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{float(v):.1%}"


def _f2(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{float(v):.2f}"


def _f3(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{float(v):.3f}"


def _dollar(v) -> str:
    try:
        return f"${float(v):,.0f}"
    except Exception:
        return "N/A"


def load_scorecard() -> pd.DataFrame | None:
    p = RESEARCH_DIR / "scorecard.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def load_ranked() -> pd.DataFrame | None:
    p = RESEARCH_DIR / "ranked_scorecard.parquet"
    if not p.exists():
        return load_scorecard()
    return pd.read_parquet(p)


def load_backtest_summary(ticker: str) -> dict:
    p = BACKTEST_DIR / ticker / "backtest_summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_risk_summary(ticker: str) -> dict:
    p = RISK_DIR / ticker / "risk_summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_training_results(ticker: str) -> dict:
    p = MODELS_DIR / ticker / "training_results.json"
    return json.loads(p.read_text()) if p.exists() else {}


def t_stat_on_returns(ticker: str, source: str = "backtest") -> float | None:
    """Compute t-statistic on daily mean returns from equity curve."""
    from scipy import stats
    if source == "backtest":
        eq_path = BACKTEST_DIR / ticker / "equity_curves.parquet"
        col = "strategy"
    else:
        eq_path = RISK_DIR / ticker / "equity_curve.parquet"
        col = "equity"
    if not eq_path.exists():
        return None
    eq = pd.read_parquet(eq_path)
    if col not in eq.columns:
        col = eq.columns[0]
    rets = eq[col].pct_change().dropna()
    t, p = stats.ttest_1samp(rets, 0.0)
    return float(t)


def compute_alpha_beta(ticker: str) -> tuple[float | None, float | None]:
    """Jensen's alpha and beta vs SPY via OLS."""
    try:
        from scipy import stats
        eq_path  = BACKTEST_DIR / ticker / "equity_curves.parquet"
        spy_path = BACKTEST_DIR / "SPY" / "equity_curves.parquet"
        if not eq_path.exists() or not spy_path.exists():
            return None, None
        eq  = pd.read_parquet(eq_path)
        spy = pd.read_parquet(spy_path)
        strat_rets = eq["strategy"].pct_change().dropna() if "strategy" in eq.columns else eq.iloc[:, 0].pct_change().dropna()
        spy_rets   = spy["strategy"].pct_change().dropna() if "strategy" in spy.columns else spy["buy_hold"].pct_change().dropna()
        common = strat_rets.index.intersection(spy_rets.index)
        if len(common) < 50:
            return None, None
        r_s = strat_rets.loc[common].values
        r_m = spy_rets.loc[common].values
        slope, intercept, *_ = stats.linregress(r_m, r_s)
        alpha_annual = intercept * 252
        return float(alpha_annual), float(slope)
    except Exception:
        return None, None


def section(title: str) -> None:
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def subsection(title: str) -> None:
    print(f"\n── {title} ──────────────────────────────────────────────")


# ===========================================================================
# SECTION 0 — Available data inventory
# ===========================================================================

def print_inventory() -> None:
    section("0. DATA INVENTORY")

    # Trained tickers
    trained = sorted(p.name for p in MODELS_DIR.iterdir() if p.is_dir()) if MODELS_DIR.exists() else []
    backtested = sorted(p.name for p in BACKTEST_DIR.iterdir() if p.is_dir()) if BACKTEST_DIR.exists() else []
    risk_done = sorted(p.name for p in RISK_DIR.iterdir() if p.is_dir()) if RISK_DIR.exists() else []

    print(f"  Trained models : {len(trained):2d}  {trained}")
    print(f"  Backtested     : {len(backtested):2d}  {backtested}")
    print(f"  Risk engine    : {len(risk_done):2d}  {risk_done}")
    sc = load_scorecard()
    if sc is not None:
        n_strat = len(sc.index.get_level_values("strategy").unique()) if "strategy" in sc.index.names else sc.shape[0]
        print(f"  Scorecard rows : {len(sc):3d}  ({n_strat} unique strategies)")
    else:
        print("  Scorecard      : NOT FOUND (run research.py zoo)")

    for name, path in [
        ("Regime breakdown", ENV_DIR / "regime_breakdown.parquet"),
        ("Cost sensitivity", ENV_DIR / "cost_sensitivity.parquet"),
        ("Signal decay",     ENV_DIR / "signal_decay_summary.parquet"),
        ("Factor attribution", ENV_DIR / "factor_attribution.parquet"),
    ]:
        status = "✓" if path.exists() else "✗ missing"
        print(f"  06C {name:20s}: {status}")


# ===========================================================================
# SECTION 1 — Abstract numbers
# ===========================================================================

def print_abstract() -> None:
    section("1. ABSTRACT — [FILL] values")

    tickers = sorted(p.name for p in BACKTEST_DIR.iterdir() if p.is_dir()) if BACKTEST_DIR.exists() else []
    if not tickers:
        print("  No backtest data found.")
        return

    # Portfolio-level stats: aggregate across all default tickers (5-ticker average as MVP proxy)
    sharpes, cagrs, dds = [], [], []
    for t in tickers:
        sm = load_backtest_summary(t).get("strategy_metrics", {})
        if sm:
            sharpes.append(sm.get("sharpe_ratio", 0))
            cagrs.append(sm.get("cagr", 0))
            dds.append(abs(sm.get("max_drawdown", 0)))

    if sharpes:
        print(f"\n  Portfolio-level (mean across {len(sharpes)} backtested tickers):")
        print(f"    Sharpe ratio     : {_f2(np.mean(sharpes))}  (range {_f2(min(sharpes))} – {_f2(max(sharpes))})")
        print(f"    CAGR             : {_pct(np.mean(cagrs))}  (range {_pct(min(cagrs))} – {_pct(max(cagrs))})")
        print(f"    Max Drawdown     : {_pct(np.mean(dds))}  (range {_pct(min(dds))} – {_pct(max(dds))})")

    # t-stats
    print("\n  t-statistics on mean daily returns:")
    for t in tickers:
        tstat = t_stat_on_returns(t)
        sig = "✓ sig" if tstat and abs(tstat) > 2.0 else "✗ insig"
        print(f"    {t:<8}  t = {_f2(tstat)}  {sig}")

    # Cost sensitivity threshold
    cost_path = ENV_DIR / "cost_sensitivity.parquet"
    if cost_path.exists():
        cs = pd.read_parquet(cost_path)
        print(f"\n  Cost sensitivity data shape: {cs.shape}")
        print(f"  Columns: {cs.columns.tolist()}")


# ===========================================================================
# SECTION 3 — Data / Methodology
# ===========================================================================

def print_methodology() -> None:
    section("3. METHODOLOGY [FILL] values")

    # Label construction params from training results
    tickers = sorted(p.name for p in MODELS_DIR.iterdir() if p.is_dir()) if MODELS_DIR.exists() else []
    if tickers:
        tr = load_training_results(tickers[0])
        subsection("Label parameters (from first trained ticker)")
        print(f"    horizon     = {tr.get('horizon', '[not saved]')} bars")
        print(f"    threshold   = ±{tr.get('threshold', '[not saved]')} (UP/DOWN label)")
        print(f"    n_folds     = {tr.get('n_folds', '[not saved]')}")

    # Date ranges from equity curves
    subsection("OOS period (from backtest equity curves)")
    for t in (tickers or []):
        eq_path = BACKTEST_DIR / t / "equity_curves.parquet"
        if eq_path.exists():
            eq = pd.read_parquet(eq_path)
            eq.index = pd.to_datetime(eq.index, utc=True)
            print(f"    {t:<8}  {eq.index.min().date()} → {eq.index.max().date()}  ({len(eq)} trading days)")
            break   # all tickers share same calendar; just show one


# ===========================================================================
# SECTION 4 — Strategy Zoo Results
# ===========================================================================

def print_zoo_results() -> None:
    section("4. STRATEGY ZOO RESULTS [FILL] values")

    sc = load_ranked()
    if sc is None:
        print("  Scorecard not found. Run: python3 research.py zoo && python3 research.py rank")
        return

    sc = sc.reset_index() if not isinstance(sc.index, pd.RangeIndex) else sc

    subsection("Per-strategy aggregate (mean across all tickers)")
    if "strategy" in sc.columns:
        agg = sc.groupby("strategy")[
            [c for c in ["sharpe_ratio", "cagr", "max_drawdown", "calmar_ratio", "win_rate"]
             if c in sc.columns]
        ].mean().sort_values("sharpe_ratio", ascending=False)
        print(agg.to_string())

    subsection("Best strategy per metric (paper table values)")
    for col, label in [
        ("sharpe_ratio",  "Highest Sharpe"),
        ("cagr",          "Highest CAGR"),
        ("max_drawdown",  "Shallowest MaxDD"),
        ("calmar_ratio",  "Highest Calmar"),
        ("win_rate",      "Highest Win Rate"),
    ]:
        if col not in sc.columns:
            continue
        if col == "max_drawdown":
            idx = sc[col].abs().idxmin()
        else:
            idx = sc[col].idxmax()
        row = sc.loc[idx]
        strat = row.get("strategy", "?")
        tick  = row.get("ticker", "?")
        val   = _pct(row[col]) if "rate" in col or "drawdown" in col or "cagr" in col else _f2(row[col])
        print(f"    {label:<22}: {strat} × {tick}  =  {val}")

    subsection("Alpha Trends strategy performance")
    if "strategy" in sc.columns:
        at = sc[sc["strategy"].str.contains("Alpha", case=False, na=False)]
        if not at.empty:
            print(at[["ticker", "strategy", "sharpe_ratio", "cagr", "max_drawdown", "win_rate"]].to_string())
        else:
            print("    Alpha_Trends not found in scorecard.")

    subsection("t-statistics from ranker (if available)")
    for col in ["t_stat", "t_statistic", "alpha", "information_ratio", "beta"]:
        if col in sc.columns:
            print(f"\n  {col}:")
            print(sc[["ticker", "strategy", col]].dropna().sort_values(col, ascending=False).head(10).to_string())


# ===========================================================================
# SECTION 5 — Regime-Gated Meta-Strategy
# ===========================================================================

def print_meta_strategy() -> None:
    section("5. META-STRATEGY [FILL] values")

    subsection("Regime breakdown (Phase 06C.1)")
    regime_path = ENV_DIR / "regime_breakdown.parquet"
    if regime_path.exists():
        rb = pd.read_parquet(regime_path)
        print(f"  Shape: {rb.shape}")
        print(f"  Columns: {rb.columns.tolist()}")
        print(f"\n  Per-strategy mean Sharpe by regime:")
        try:
            if "hmm_regime" in rb.columns and "strategy" in rb.columns and "sharpe_ratio" in rb.columns:
                pivot = rb.pivot_table(values="sharpe_ratio", index="strategy", columns="hmm_regime", aggfunc="mean")
                print(pivot.round(3).to_string())
            else:
                print(rb.head(10).to_string())
        except Exception as e:
            print(f"  (Could not pivot: {e})")
            print(rb.head(10).to_string())
    else:
        print("  Not found — run: python3 research.py env --regime")

    subsection("Best strategy per regime state")
    if regime_path.exists():
        rb = pd.read_parquet(regime_path)
        if "hmm_regime" in rb.columns and "sharpe_ratio" in rb.columns and "strategy" in rb.columns:
            for regime in sorted(rb["hmm_regime"].unique()):
                sub = rb[rb["hmm_regime"] == regime]
                best = sub.loc[sub["sharpe_ratio"].idxmax()]
                print(f"    Regime {regime}: best = {best.get('strategy','?')} × {best.get('ticker','?')}  Sharpe={_f2(best['sharpe_ratio'])}")

    subsection("ML Ensemble vs individual strategies (backtest tickers)")
    tickers = sorted(p.name for p in BACKTEST_DIR.iterdir() if p.is_dir()) if BACKTEST_DIR.exists() else []
    print(f"\n  {'Ticker':<8}  {'Sharpe':>7}  {'CAGR':>8}  {'MaxDD':>7}  {'t-stat':>7}  {'Alpha':>8}  {'Beta':>6}")
    print("  " + "-" * 65)
    for t in tickers:
        sm = load_backtest_summary(t).get("strategy_metrics", {})
        tstat = t_stat_on_returns(t)
        alpha, beta = compute_alpha_beta(t)
        print(
            f"  {t:<8}  {_f2(sm.get('sharpe_ratio')):>7}  "
            f"{_pct(sm.get('cagr')):>8}  "
            f"{_pct(sm.get('max_drawdown')):>7}  "
            f"{_f2(tstat):>7}  "
            f"{_pct(alpha):>8}  "
            f"{_f2(beta):>6}"
        )


# ===========================================================================
# SECTION 6 — Environment Characterisation
# ===========================================================================

def print_env_characterisation() -> None:
    section("6. ENVIRONMENT CHARACTERISATION [FILL] values")

    # ── 6.1 Cost Sensitivity ──────────────────────────────────────────────
    subsection("6.1 Cost Sensitivity")
    cost_path = ENV_DIR / "cost_sensitivity.parquet"
    if cost_path.exists():
        cs = pd.read_parquet(cost_path).reset_index()
        print(f"  Shape: {cs.shape}  Columns: {cs.columns.tolist()}")

        # Find edge-survival threshold (commission where Sharpe drops below 1.0)
        if "commission" in cs.columns and "sharpe_ratio" in cs.columns:
            above = cs[cs["sharpe_ratio"] >= 1.0]
            below = cs[cs["sharpe_ratio"] <  1.0]
            if not above.empty and not below.empty:
                max_comm = above["commission"].max()
                print(f"\n  Edge survival threshold: Sharpe ≥ 1.0 up to commission = {max_comm:.4f} ({max_comm*100:.0f} bps)")

            # Table: mean Sharpe by commission level
            print("\n  Mean Sharpe by commission level:")
            print(cs.groupby("commission")["sharpe_ratio"].mean().round(3).to_string())
    else:
        print("  Not found — run: python3 research.py env --cost")

    # ── 6.2 Signal Decay ─────────────────────────────────────────────────
    subsection("6.2 Signal Decay")
    decay_path = ENV_DIR / "signal_decay_summary.parquet"
    if decay_path.exists():
        sd = pd.read_parquet(decay_path).reset_index()
        print(f"  Shape: {sd.shape}  Columns: {sd.columns.tolist()}")
        print("\n  Signal decay summary (first 20 rows):")
        print(sd.head(20).to_string())
    else:
        print("  Not found — run: python3 research.py env --decay")

    # ── 6.3 Factor Attribution ───────────────────────────────────────────
    subsection("6.3 Fama-French Factor Attribution")
    factor_path = ENV_DIR / "factor_attribution.parquet"
    if factor_path.exists():
        fa = pd.read_parquet(factor_path).reset_index()
        print(f"  Shape: {fa.shape}  Columns: {fa.columns.tolist()}")
        print("\n  Factor attribution table:")
        # Show key columns: alpha, beta_mkt, t_alpha, r_squared
        key_cols = [c for c in ["ticker", "strategy", "alpha_annual", "alpha", "beta_mkt",
                                 "beta", "t_alpha", "t_stat_alpha", "r_squared"]
                    if c in fa.columns]
        if key_cols:
            print(fa[key_cols].sort_values(
                next((c for c in ["alpha_annual", "alpha"] if c in fa.columns), key_cols[0]),
                ascending=False
            ).head(20).to_string())
        else:
            print(fa.head(10).to_string())

        # Extract the headline alpha for the meta-strategy / ML ensemble
        alpha_col = next((c for c in ["alpha_annual", "alpha"] if c in fa.columns), None)
        t_col     = next((c for c in ["t_alpha", "t_stat_alpha", "t_stat"] if c in fa.columns), None)
        if alpha_col:
            best = fa.loc[fa[alpha_col].idxmax()]
            print(f"\n  Best factor alpha:  {_pct(best[alpha_col])} annualised"
                  f"  (t={_f2(best[t_col]) if t_col else 'N/A'})"
                  f"  strategy={best.get('strategy','?')}")
    else:
        print("  Not found — run: python3 research.py env --factors")


# ===========================================================================
# SECTION 7 — HMM Regime Statistics (for Appendix C)
# ===========================================================================

def print_hmm_stats() -> None:
    section("APPENDIX C — HMM Regime Statistics")

    regime_path = ENV_DIR / "regime_breakdown.parquet"
    if not regime_path.exists():
        print("  Not found — run: python3 research.py env --regime")
        return

    rb = pd.read_parquet(regime_path).reset_index()
    if "hmm_regime" in rb.columns:
        print("\n  Regime state distribution (all tickers):")
        dist = rb["hmm_regime"].value_counts().sort_index()
        total = dist.sum()
        for state, count in dist.items():
            label = {0: "Bear", 1: "Ranging", 2: "Bull"}.get(state, str(state))
            print(f"    State {state} ({label}): {count:5d} bars  ({count/total:.1%})")

    if "sharpe_ratio" in rb.columns and "hmm_regime" in rb.columns:
        print("\n  Mean Sharpe by regime state:")
        by_regime = rb.groupby("hmm_regime")["sharpe_ratio"].agg(["mean", "std", "count"])
        for state, row in by_regime.iterrows():
            label = {0: "Bear", 1: "Ranging", 2: "Bull"}.get(state, str(state))
            print(f"    State {state} ({label}): mean={_f2(row['mean'])}  std={_f2(row['std'])}  n={int(row['count'])}")


# ===========================================================================
# SECTION 8 — OOS Training Metrics (for Section 3 / Appendix)
# ===========================================================================

def print_oos_metrics() -> None:
    section("ML ENSEMBLE OOS METRICS (Section 3.5)")

    tickers = sorted(p.name for p in MODELS_DIR.iterdir() if p.is_dir()) if MODELS_DIR.exists() else []
    print(f"\n  {'Ticker':<8}  {'Acc':>7}  {'F1_macro':>9}  {'F1_down':>8}  {'F1_flat':>8}  {'F1_up':>8}  {'n_folds':>7}")
    print("  " + "-" * 60)
    for t in tickers:
        tr  = load_training_results(t)
        oos = tr.get("oos_metrics", {})
        print(
            f"  {t:<8}  "
            f"{_f3(oos.get('accuracy')):>7}  "
            f"{_f3(oos.get('f1_macro')):>9}  "
            f"{_f3(oos.get('f1_down')):>8}  "
            f"{_f3(oos.get('f1_flat')):>8}  "
            f"{_f3(oos.get('f1_up')):>8}  "
            f"{str(tr.get('n_folds','?')):>7}"
        )


# ===========================================================================
# SECTION 9 — Performance Targets Gate Check
# ===========================================================================

def print_targets() -> None:
    section("PERFORMANCE TARGETS (MVP gate check)")

    print(f"\n  {'Ticker':<8}  {'Sharpe':>7}  {'MaxDD':>7}  {'WinRate':>8}  {'CAGR':>7}  {'Calmar':>7}  {'t-stat':>7}  Gates")
    print("  " + "-" * 75)

    tickers = sorted(p.name for p in BACKTEST_DIR.iterdir() if p.is_dir()) if BACKTEST_DIR.exists() else []
    for t in tickers:
        sm    = load_backtest_summary(t).get("strategy_metrics", {})
        tstat = t_stat_on_returns(t)
        s     = sm.get("sharpe_ratio", 0)
        dd    = abs(sm.get("max_drawdown", 1))
        wr    = sm.get("win_rate", 0)
        cagr  = sm.get("cagr", 0)
        cal   = sm.get("calmar_ratio", 0)

        gates = []
        if s     > 1.0:  gates.append("Sharpe✓")
        if dd    < 0.20: gates.append("MaxDD✓")
        if wr    > 0.55: gates.append("WinRate✓")
        if cagr  > 0.15: gates.append("CAGR✓")
        if cal   > 0.5:  gates.append("Calmar✓")
        if tstat and abs(tstat) > 2.0: gates.append("t✓")

        print(
            f"  {t:<8}  {_f2(s):>7}  {_pct(dd):>7}  {_pct(wr):>8}  "
            f"{_pct(cagr):>7}  {_f2(cal):>7}  {_f2(tstat):>7}  {' '.join(gates)}"
        )


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print_inventory()
    print_oos_metrics()
    print_abstract()
    print_methodology()
    print_zoo_results()
    print_meta_strategy()
    print_env_characterisation()
    print_hmm_stats()
    print_targets()

    print("\n" + "═" * 70)
    print("  EXTRACTION COMPLETE")
    print("  Copy values above into docs/PHASE_06D_PAPER_DRAFT.md")
    print("═" * 70)
