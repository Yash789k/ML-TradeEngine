"""Read-only forensic audit of saved research artifacts and accounting invariants.

Run with the project's Python environment:
    python scripts/repository_audit.py --output docs/audit-2026-09-27.json

No network requests, training, cache refreshes, orders, or alerts. Only the
explicit output file is written. Accounting sensitivity preserves saved trade
dates and long-only fill prices; it is NOT a leakage-free strategy backtest.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def metrics(returns: pd.Series) -> dict:
    """Independent implementation; match legacy 252-day, 5% RF convention."""
    r = returns.dropna().astype(float)
    if len(r) == 0 or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError("Invalid return series")
    nav = (1 + r).cumprod()
    std = r.std(ddof=1)
    # Include the initial capital in the running peak.
    dd = nav / nav.cummax().clip(lower=1) - 1
    daily_rf = 1.05 ** (1 / 252) - 1
    return {
        "n_days": len(r), "start": str(r.index.min()), "end": str(r.index.max()),
        "total_return": float(nav.iloc[-1] - 1),
        "cagr": float(nav.iloc[-1] ** (252 / len(r)) - 1),
        "sharpe_ratio": float((r.mean() - daily_rf) / std * np.sqrt(252)) if std > 1e-12 else None,
        "max_drawdown": float(dd.min()),
        "ending_nav_per_10000": float(nav.iloc[-1] * 10_000),
    }


def accounting_sensitivity(eq: pd.Series, trades: pd.DataFrame, cost: float = .0015):
    """Repair omitted exit-day returns, retaining every original trade decision.

    For these fully invested long-only simulations, return on an exit day must
    be shares * exit_price * (1-cost) / previous_NAV - 1. Saved shares refer to
    saved NAV, so this yields the scale-independent correct daily return. All
    other daily returns are unchanged. Compounding rescales subsequent shares.
    Both entry and exit costs are included when computing closed-trade wins.
    """
    r = eq.pct_change().fillna(0.0)
    fixed = r.copy()
    if trades.empty:
        return fixed, {"closed_trades": 0, "net_win_rate": None, "legacy_win_rate": None}
    if not (trades.direction == 1).all():
        raise ValueError("Sensitivity is only defined for long-only logs")
    exits = trades.loc[trades.type == "exit"]
    if exits.date.duplicated().any():
        raise ValueError("Multiple exits per day are unsupported")
    for row in exits.itertuples():
        loc = eq.index.get_loc(row.date)
        if loc == 0:
            raise ValueError("Exit has no preceding NAV")
        fixed.iloc[loc] = abs(row.shares) * row.price * (1 - cost) / eq.iloc[loc - 1] - 1
    net_trade_returns = exits.price / exits.entry_price * (1 - cost) ** 2 - 1
    return fixed, {
        "closed_trades": len(exits),
        "net_win_rate": float((net_trade_returns > 0).mean()) if len(exits) else None,
        "legacy_win_rate": float((exits.pnl > 0).mean()) if len(exits) else None,
        "changed_daily_returns": int((abs(fixed - r) > 1e-10).sum()),
    }


def reproductions() -> dict:
    from src.backtest.simulator import run_simulation
    from src.features.regime import decode_hmm
    from src.live.alerts import Alerts
    from src.models.gating import rolling_expert_accuracy

    idx = pd.date_range("2024-01-01", periods=4, tz="UTC")
    long = run_simulation(pd.DataFrame({"close": [100., 100., 110., 110.],
                                        "filtered_signal": [2, 1, 1, 1]}, index=idx),
                          commission=0, slippage=0)
    short = run_simulation(pd.DataFrame({"close": [100., 100., 90., 90.],
                                         "filtered_signal": [0, 0, 0, 1]}, index=idx),
                           commission=0, slippage=0, mode="long_short")
    small_gain = run_simulation(pd.DataFrame({"close": [100., 100., 100.2, 100.2],
                                              "filtered_signal": [2, 1, 1, 1]}, index=idx))

    class StubHMM:
        def predict(self, values):
            return np.array([0, 1, 2])

    decoded = decode_hmm(StubHMM(), pd.DataFrame({"log_return": [np.nan, 1., 2., 3.]}, index=idx))
    try:
        with patch.dict("os.environ", {"SMTP_PORT": ""}):
            Alerts()
        alert_error = None
    except ValueError as exc:
        alert_error = str(exc)
    # At t=20, a 5-bar label originating at 19 is not known until 24.
    probs = [np.tile([0., 0., 1.], (40, 1))]
    labels = np.full(40, 2)
    original_acc = rolling_expert_accuracy(probs, labels)
    labels[19] = 0
    changed_acc = rolling_expert_accuracy(probs, labels)
    return {
        "long_exit": {"actual_nav": long.equity_curve.tolist(), "expected_final": 11000.,
                      "trade_pnl": float(long.trade_log.iloc[-1].pnl)},
        "short_flat_price": {"actual_nav": short.equity_curve.tolist(), "expected_final": 11000.},
        "trade_costs": {"logged_pnl": float(small_gain.trade_log.iloc[-1].pnl),
                        "expected_pnl": 10000 * ((1 - .0015) ** 2 * 1.002 - 1)},
        "hmm_row_alignment": {"actual": decoded.tolist(), "expected": [-1, 0, 1, 2]},
        "empty_smtp_port": alert_error,
        "moe_unavailable_label": {"t": 20, "before": float(original_acc[20, 0]),
                                  "after": float(changed_acc[20, 0]), "label_available_at": 24},
    }


def audit() -> dict:
    report = {
        "audit_time_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "Current working tree, including pre-existing uncommitted changes",
        "caveat": "All saved strategy results remain unvalidated: lookahead, accounting, calendars, and execution parity defects. Accounting sensitivity fixes only exit-day NAV and trade cost win classification.",
        "reproductions": reproductions(),
    }
    now = pd.Timestamp.now(tz="UTC").normalize()
    report["raw_data"] = []
    for p in sorted((ROOT / "data/parquet").glob("*.parquet")):
        d = pd.read_parquet(p)
        row = {"file": str(p.relative_to(ROOT)), "rows": len(d), "columns": list(d.columns),
               "start": str(d.index.min()), "end": str(d.index.max()),
               "calendar_days_old": (now - d.index.max()).days,
               "nan_cells": int(d.isna().sum().sum()), "duplicates": int(d.index.duplicated().sum())}
        if "Volume" in d:
            row["zero_volume_rows"] = int((d.Volume == 0).sum())
        row["last_nonnull_by_column"] = {c: str(d[c].last_valid_index()) for c in d if c != "ticker"}
        report["raw_data"].append(row)
    report["features"] = []
    for p in sorted((ROOT / "data/features").glob("*.parquet")):
        d = pd.read_parquet(p)
        ticker = p.stem.removesuffix("_features")
        fp = ROOT / "data/models" / ticker / "feature_cols.json"
        cols = json.loads(fp.read_text()) if fp.exists() else []
        report["features"].append({"ticker": ticker, "rows": len(d), "start": str(d.index.min()),
                                   "end": str(d.index.max()), "missing_model_features": sorted(set(cols) - set(d.columns)),
                                   "unknown_regime_last_10": int((d.hmm_regime.tail(10) == -1).sum())})

    report["ml"] = []
    old_ml, fixed_ml, risk_rets = {}, {}, {}
    for p in sorted((ROOT / "data/backtest").glob("*/backtest_summary.json")):
        ticker = p.parent.name
        summary = json.loads(p.read_text())
        eq = pd.read_parquet(p.parent / "equity_curves.parquet")["strategy"]
        trades = pd.read_parquet(p.parent / "trade_log.parquet")
        c = summary["sim_params"]["commission"] + summary["sim_params"]["slippage"]
        fixed, trade_stats = accounting_sensitivity(eq, trades, c)
        old_ml[ticker], fixed_ml[ticker] = eq.pct_change().fillna(0), fixed
        oos = pd.read_parquet(ROOT / "data/models" / ticker / "oos_predictions.parquet")
        pred = oos[["p_down", "p_flat", "p_up"]].values.argmax(axis=1)
        up = pred == 2
        train = json.loads((ROOT / "data/models" / ticker / "training_results.json").read_text())
        row = {"ticker": ticker, "saved": summary["strategy_metrics"], "recomputed": metrics(old_ml[ticker]),
               "accounting_only": metrics(fixed), **trade_stats,
               "accuracy": float((pred == oos.true_label).mean()),
               "majority_baseline": float(oos.true_label.value_counts(normalize=True).max()),
               "always_up_accuracy": float((oos.true_label == 2).mean()),
               "up_precision": float((oos.true_label[up] == 2).mean()) if up.any() else None,
               "down_prediction_fraction": float((pred == 0).mean()),
               "probability_sum_max_error": float(abs(oos[["p_down", "p_flat", "p_up"]].sum(axis=1) - 1).max()),
               "horizon_saved": train.get("horizon"), "threshold_saved": train.get("threshold"),
               "has_lstm": (ROOT / "data/models" / ticker / "lstm_final.pt").exists()}
        rp = ROOT / "data/risk" / ticker / "risk_summary.json"
        if rp.exists():
            row["risk_saved"] = json.loads(rp.read_text())["risk_metrics"]
            ep = rp.parent / "equity_curve.parquet"
            # Phase 05 uses equity_curves.parquet (risk / phase04 / buy_hold).
            if not ep.exists():
                ep = rp.parent / "equity_curves.parquet"
            if ep.exists():
                rd = pd.read_parquet(ep)
                col = "risk" if "risk" in rd else rd.columns[0]
                risk_rets[ticker] = rd[col].pct_change().fillna(0)
        report["ml"].append(row)
    # Every portfolio here is a hypothetical daily equal-weight portfolio on
    # intersecting dates; it is not a broker account or separately traded fund.
    report["ml_ew_portfolio"] = metrics(pd.DataFrame(old_ml).dropna().mean(axis=1))
    report["ml_ew_accounting_only"] = metrics(pd.DataFrame(fixed_ml).dropna().mean(axis=1))
    if risk_rets:
        report["risk_ew_portfolio"] = metrics(pd.DataFrame(risk_rets).dropna().mean(axis=1))

    score = pd.read_parquet(ROOT / "data/research/scorecard.parquet")
    report["zoo"] = []
    report["stale_zoo_trade_logs"] = []
    old_zoo, fixed_zoo = {}, {}
    for (ticker, strategy), saved in score.iterrows():
        directory = ROOT / "data/research" / ticker / strategy
        eq = pd.read_parquet(directory / "equity_curve.parquet").iloc[:, 0]
        tr = pd.read_parquet(directory / "trade_log.parquet")
        if pd.isna(saved.get("n_trades")) and eq.nunique() == 1:
            report["stale_zoo_trade_logs"].append({"ticker": ticker, "strategy": strategy,
                                                  "stale_entries": int((tr.type == "entry").sum())})
            tr = tr.iloc[:0]
        fixed, ts = accounting_sensitivity(eq, tr)
        old = eq.pct_change().fillna(0)
        old_zoo.setdefault(ticker, {})[strategy] = old
        fixed_zoo.setdefault(ticker, {})[strategy] = fixed
        report["zoo"].append({"ticker": ticker, "strategy": strategy,
                              "saved": saved.to_dict(), "recomputed": metrics(old),
                              "accounting_only": metrics(fixed), **ts})
    report["strategy_ew_portfolios"] = {}
    for s in sorted(score.index.get_level_values(1).unique()):
        old = pd.DataFrame({t: v[s] for t, v in old_zoo.items()}).dropna().mean(axis=1)
        fixed = pd.DataFrame({t: v[s] for t, v in fixed_zoo.items()}).dropna().mean(axis=1)
        report["strategy_ew_portfolios"][s] = {"saved_curve_metrics": metrics(old), "accounting_only": metrics(fixed)}

    meta_dir = ROOT / "data/research/meta"
    meta = pd.read_parquet(meta_dir / "meta_returns.parquet")
    alloc = pd.read_parquet(meta_dir / "meta_allocations.parquet")
    fixed_meta = {}
    for t in alloc:
        old = pd.DataFrame(old_zoo[t]).loc[meta.index]
        fixed = pd.DataFrame(fixed_zoo[t]).loc[meta.index]
        adjustments = []
        for date in meta.index:
            chosen = alloc.loc[date, t]
            delta = fixed.loc[date] - old.loc[date]
            adjustments.append(delta.mean() if chosen == "EW" else delta[chosen])
        fixed_meta[t] = meta[t] + np.array(adjustments)
    report["meta_saved"] = json.loads((meta_dir / "meta_summary.json").read_text())
    report["meta_recomputed"] = metrics(meta.portfolio)
    report["meta_fixed_allocations_accounting_only"] = metrics(pd.DataFrame(fixed_meta).mean(axis=1))
    # Check the paper benchmark's actual source coverage before reindex/fill.
    spy = pd.read_parquet(ROOT / "data/backtest/SPY/equity_curves.parquet").buy_hold
    report["meta_benchmark_coverage"] = {"meta_rows": len(meta), "spy_rows": len(spy),
                                          "missing_spy_dates_filled_zero": int((~meta.index.isin(spy.index)).sum()),
                                          "spy_start": str(spy.index.min()), "spy_end": str(spy.index.max())}
    uri = (ROOT / "data/live/signal_log.db").as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as c:
        report["local_live_counts"] = {t: c.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                                        for t in ("signals", "orders", "equity", "circuit_breaker")}
        report["last_signal_time"] = c.execute("SELECT MAX(run_ts) FROM signals").fetchone()[0]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    # Normalize non-finite legacy fields to JSON null.
    def clean(obj):
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
            return None
        return obj
    args.output.write_text(json.dumps(clean(report), indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}: {len(report['ml'])} ML tickers, {len(report['zoo'])} zoo runs.")
    for key in ["ml_ew_portfolio", "ml_ew_accounting_only", "risk_ew_portfolio", "meta_recomputed", "meta_fixed_allocations_accounting_only", "meta_benchmark_coverage"]:
        if key in report:
            print(key, json.dumps(report[key]))


if __name__ == "__main__":
    main()
