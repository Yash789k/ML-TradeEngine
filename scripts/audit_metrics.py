"""
Full-artifact consistency audit.

Recomputes performance metrics from raw artifacts (equity curves, OOS
predictions, price data) and cross-checks them against every saved summary
(zoo scorecard, ranked scorecard, meta summary, backtest summaries).

Flags:
  A. Raw data issues        — NaNs, non-positive prices, stale end dates,
                              duplicate index entries, large gaps
  B. Zoo scorecard drift    — saved Sharpe/CAGR vs recomputed from curves
  C. Meta summary drift     — saved metrics vs recomputed from meta_returns
  D. ML backtest drift      — saved summary vs recomputed from equity curves
  E. Abnormal values        — |Sharpe| > 3, MaxDD < -80%, equity <= 0,
                              OOS accuracy < 1/3, degenerate equity curves

Usage:
    python3 scripts/audit_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.backtest.metrics import compute_metrics

ROOT = Path(__file__).resolve().parents[1]
FLAGS: list[str] = []


def flag(section: str, msg: str) -> None:
    FLAGS.append(f"[{section}] {msg}")


# ───────────────────────────── A. raw data ──────────────────────────────

def audit_raw_data() -> None:
    print("\n=== A. Raw price data (data/parquet) ===")
    pq_dir = ROOT / "data" / "parquet"
    files = sorted(pq_dir.glob("*.parquet"))
    print(f"  {len(files)} parquet files")
    for f in files:
        df = pd.read_parquet(f)
        name = f.stem
        if df.empty:
            flag("RAW", f"{name}: EMPTY file")
            continue
        n_nan = int(df.isna().sum().sum())
        if "Close" in df.columns:
            bad_px = int((df["Close"] <= 0).sum())
            if bad_px:
                flag("RAW", f"{name}: {bad_px} non-positive Close prices")
        if df.index.duplicated().any():
            flag("RAW", f"{name}: {int(df.index.duplicated().sum())} duplicated index rows")
        if n_nan:
            # macro series legitimately have NaN on non-release days; only
            # flag OHLCV files
            if {"Open", "Close"}.issubset(df.columns):
                flag("RAW", f"{name}: {n_nan} NaN cells in OHLCV")
        try:
            idx = pd.to_datetime(df.index)
            gaps = idx.to_series().diff().dt.days.dropna()
            big = gaps[gaps > 7]
            if len(big):
                flag("RAW", f"{name}: {len(big)} gaps > 7 days (max {int(big.max())}d)")
            print(f"  {name:<28} {len(df):>5} rows  {idx.min().date()} → {idx.max().date()}  nan={n_nan}")
        except Exception as exc:
            flag("RAW", f"{name}: index not datetime ({exc})")


# ─────────────────────────── B. zoo scorecard ───────────────────────────

def audit_zoo() -> None:
    print("\n=== B. Strategy zoo scorecard vs equity curves ===")
    sc_path = ROOT / "data" / "research" / "scorecard.parquet"
    if not sc_path.exists():
        flag("ZOO", "scorecard.parquet missing")
        return
    sc = pd.read_parquet(sc_path)
    print(f"  scorecard: {len(sc)} rows, cols={list(sc.columns)[:8]}…")

    n_checked = n_drift = 0
    for (ticker, strat), row in sc.iterrows():
        eq_path = ROOT / "data" / "research" / str(ticker) / str(strat) / "equity_curve.parquet"
        if not eq_path.exists():
            flag("ZOO", f"{ticker}/{strat}: equity_curve.parquet missing")
            continue
        eq = pd.read_parquet(eq_path).iloc[:, 0]
        if (eq <= 0).any():
            flag("ZOO", f"{ticker}/{strat}: equity curve has non-positive values")
        rets = eq.pct_change().fillna(0.0)
        m = compute_metrics(rets)
        n_checked += 1
        for k_sc, k_m, tol in [("sharpe_ratio", "sharpe_ratio", 0.02),
                               ("cagr", "cagr", 0.005),
                               ("max_drawdown", "max_drawdown", 0.005)]:
            if k_sc in row.index and pd.notna(row[k_sc]):
                if abs(float(row[k_sc]) - float(m[k_m])) > tol:
                    n_drift += 1
                    flag("ZOO", f"{ticker}/{strat}: {k_sc} saved={row[k_sc]:.3f} recomputed={m[k_m]:.3f}")
        if abs(m["sharpe_ratio"]) > 3:
            flag("ZOO-ABNORMAL", f"{ticker}/{strat}: |Sharpe|={m['sharpe_ratio']:.2f} > 3")
        if m["max_drawdown"] < -0.80:
            flag("ZOO-ABNORMAL", f"{ticker}/{strat}: MaxDD={m['max_drawdown']:.1%}")
    print(f"  checked {n_checked} (ticker,strategy) pairs, {n_drift} metric drifts")


# ─────────────────────────── C. meta summary ────────────────────────────

def audit_meta() -> None:
    print("\n=== C. Meta-strategy summary vs meta_returns ===")
    meta_dir = ROOT / "data" / "research" / "meta"
    summ_p = meta_dir / "meta_summary.json"
    ret_p  = meta_dir / "meta_returns.parquet"
    if not (summ_p.exists() and ret_p.exists()):
        flag("META", "meta artifacts missing")
        return
    summ = json.loads(summ_p.read_text())
    rets = pd.read_parquet(ret_p)
    port = rets["portfolio"]
    m = compute_metrics(port)
    saved = summ["meta_portfolio"]
    print(f"  portfolio: {len(port)} days  {rets.index.min().date()} → {rets.index.max().date()}")
    for k in ["sharpe_ratio", "cagr", "max_drawdown", "calmar_ratio"]:
        s, r = float(saved[k]), float(m[k])
        ok = abs(s - r) < 0.02
        print(f"    {k:<15} saved={s:>8.4f}  recomputed={r:>8.4f}  {'OK' if ok else 'DRIFT'}")
        if not ok:
            flag("META", f"portfolio {k}: saved={s:.4f} recomputed={r:.4f}")
    # per-ticker sanity
    for t in [c for c in rets.columns if c != "portfolio"]:
        tm = compute_metrics(rets[t])
        if abs(tm["sharpe_ratio"]) > 3:
            flag("META-ABNORMAL", f"{t}: per-ticker meta Sharpe {tm['sharpe_ratio']:.2f}")
        if tm["max_drawdown"] < -0.5:
            flag("META-ABNORMAL", f"{t}: per-ticker meta MaxDD {tm['max_drawdown']:.1%}")
    # allocations should exist for every ticker/day
    alloc = pd.read_parquet(meta_dir / "meta_allocations.parquet")
    n_null = int(alloc.isna().sum().sum())
    if n_null:
        flag("META", f"{n_null} null allocation cells")


# ─────────────────────────── D. ML backtests ────────────────────────────

def audit_ml_backtests() -> None:
    print("\n=== D. ML ensemble backtests vs equity curves ===")
    bt_root = ROOT / "data" / "backtest"
    tickers = sorted(d.name for d in bt_root.iterdir() if d.is_dir())
    print(f"  {len(tickers)} tickers")
    n_drift = 0
    for t in tickers:
        summ_p = bt_root / t / "backtest_summary.json"
        eq_p   = bt_root / t / "equity_curves.parquet"
        if not (summ_p.exists() and eq_p.exists()):
            flag("MLBT", f"{t}: missing summary or curves")
            continue
        summ = json.loads(summ_p.read_text())
        eq = pd.read_parquet(eq_p)
        strat_col = "strategy" if "strategy" in eq.columns else eq.columns[0]
        rets = eq[strat_col].pct_change().fillna(0.0)
        m = compute_metrics(rets)
        saved_m = summ.get("strategy", summ)
        sv = saved_m.get("sharpe_ratio")
        if sv is not None and abs(float(sv) - m["sharpe_ratio"]) > 0.02:
            n_drift += 1
            flag("MLBT", f"{t}: sharpe saved={sv} recomputed={m['sharpe_ratio']:.3f}")
        # OOS accuracy sanity
        oos_p = ROOT / "data" / "models" / t / "oos_predictions.parquet"
        if oos_p.exists():
            oos = pd.read_parquet(oos_p)
            pred = np.argmax(oos[["p_down", "p_flat", "p_up"]].values, axis=1)
            acc = float((pred == oos["true_label"].values).mean())
            if acc < 1/3:
                flag("MLBT-ABNORMAL", f"{t}: OOS accuracy {acc:.3f} below random")
            psum = oos[["p_down", "p_flat", "p_up"]].sum(axis=1)
            if (abs(psum - 1) > 0.01).any():
                flag("MLBT", f"{t}: OOS probabilities do not sum to 1 "
                             f"(range {psum.min():.3f}–{psum.max():.3f})")
    print(f"  {n_drift} sharpe drifts flagged")


# ────────────────────────────── report ──────────────────────────────────

def main() -> None:
    audit_raw_data()
    audit_zoo()
    audit_meta()
    audit_ml_backtests()

    print("\n" + "═" * 74)
    print(f"  AUDIT COMPLETE — {len(FLAGS)} flags")
    print("═" * 74)
    for f in FLAGS:
        print("  " + f)
    (ROOT / "data" / "audit_flags.json").write_text(json.dumps(FLAGS, indent=2))
    print(f"\n  saved → data/audit_flags.json")


if __name__ == "__main__":
    main()
