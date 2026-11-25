"""
Dynamic policy layer benchmark — static constants vs PolicyEngine.

For every ticker in the 22-asset universe, runs the full Phase 04→05
pipeline twice on the identical walk-forward OOS predictions:

  static  : confidence 0.38, ATR stop 2.0×, full size, always eligible
  dynamic : PolicyEngine — threshold tightens in unstable regimes, ATR
            stop widens with the causal vol percentile, size halves in
            stress, entries blocked in vol extremes / unknown regimes

Both variants share every other parameter (Kelly filter, circuit
breaker, costs), so the delta isolates the policy layer.

Outputs
-------
  data/research/policy/policy_comparison.parquet   per-ticker table
  data/research/policy/policy_comparison.json      + aggregate summary

Usage:
    python3 scripts/compare_policy.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("compare_policy")

ROOT    = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "research" / "policy"

TICKERS = sorted(p.name for p in (ROOT / "data" / "models").iterdir() if p.is_dir())


def run_ticker(ticker: str) -> dict:
    from src.backtest.metrics import compute_metrics
    from src.backtest.signals import build_signal_df
    from src.risk.engine import RiskEngine
    from src.risk.policy_engine import PolicyEngine

    feat = pd.read_parquet(ROOT / "data" / "features" / f"{ticker}_features.parquet")
    feat.index = pd.to_datetime(feat.index, utc=True)

    row = {"ticker": ticker}

    variants = [
        ("static",  None),
        ("dynamic", PolicyEngine()),                    # full rule set
        ("dyn_nobump", PolicyEngine(unstable_bump=0.0)),  # ablation: no threshold
                                                          # tightening, keep ATR/size/
                                                          # eligibility rules
    ]
    for mode, policy in variants:
        sig = build_signal_df(ticker, feat, policy_engine=policy)
        eng = RiskEngine()
        res = eng.run(ticker, sig, feat, policy_engine=policy)
        m   = compute_metrics(res.daily_returns,
                              res.trade_log if len(res.trade_log) else None)
        n_trades = int((res.trade_log["type"] == "entry").sum()) if len(res.trade_log) else 0
        row.update({
            f"{mode}_sharpe":   round(m["sharpe_ratio"], 3),
            f"{mode}_cagr":     round(m["cagr"], 4),
            f"{mode}_max_dd":   round(m["max_drawdown"], 4),
            f"{mode}_calmar":   round(m["calmar_ratio"], 3),
            f"{mode}_n_trades": n_trades,
        })
    return row


def main() -> None:
    rows = []
    for i, ticker in enumerate(TICKERS, 1):
        print(f"[{i:2d}/{len(TICKERS)}] {ticker} …", flush=True)
        try:
            rows.append(run_ticker(ticker))
        except Exception as exc:
            print(f"    {ticker} failed: {exc}")

    df = pd.DataFrame(rows).set_index("ticker")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "policy_comparison.parquet")

    agg: dict = {"n_tickers": len(df)}
    for mode in ("static", "dynamic", "dyn_nobump"):
        agg.update({
            f"mean_sharpe_{mode}":   round(float(df[f"{mode}_sharpe"].mean()), 3),
            f"median_sharpe_{mode}": round(float(df[f"{mode}_sharpe"].median()), 3),
            f"mean_max_dd_{mode}":   round(float(df[f"{mode}_max_dd"].mean()), 4),
            f"mean_calmar_{mode}":   round(float(df[f"{mode}_calmar"].mean()), 3),
            f"mean_trades_{mode}":   round(float(df[f"{mode}_n_trades"].mean()), 1),
        })
    for mode in ("dynamic", "dyn_nobump"):
        agg[f"sharpe_improved_{mode}"] = int((df[f"{mode}_sharpe"] > df["static_sharpe"]).sum())
        agg[f"dd_improved_{mode}"]     = int((df[f"{mode}_max_dd"] > df["static_max_dd"]).sum())
    (OUT_DIR / "policy_comparison.json").write_text(
        json.dumps({"aggregate": agg, "per_ticker": df.reset_index().to_dict("records")},
                   indent=2)
    )

    pd.set_option("display.width", 160)
    print("\n", df.to_string())
    print("\nAggregate:", json.dumps(agg, indent=2))
    print(f"\nSaved → {OUT_DIR}")


if __name__ == "__main__":
    main()
