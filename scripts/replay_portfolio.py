"""Replay frozen research decisions with one constrained account and fill ledger."""

from pathlib import Path
import argparse
import hashlib
import json
import shutil
import importlib.metadata
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.backtest.portfolio import replay_portfolio
from src.backtest.metrics import compute_metrics
from src.execution.account import Limits
from src.research.protocol import evidence, Protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capital", type=float, default=10000.0)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for pattern in ["src/**/*.py", "scripts/*.py", "requirements.txt"]:
        for source in ROOT.glob(pattern):
            target = out / "source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    (out / "dependencies.json").write_text(
        json.dumps(
            {
                d.metadata["Name"]: d.version
                for d in importlib.metadata.distributions()
                if d.metadata["Name"]
            },
            indent=2,
        )
    )
    research = args.research.resolve()
    p = Protocol()
    sc = pd.read_csv(research / "scorecard.csv")
    names = [
        "Time_Series_Momentum_12_1",
        "Mean_Reversion",
        "EMA_Crossover",
        "Turtle_Breakout",
        "Relative_Value_Long_Leg",
        "SMA_200_Trend_Fallback",
        "Vol_Breakout",
        "Alpha_Trends",
    ]
    # This rule is fixed before reading evaluation metrics.
    selected = (
        sc[(sc.period == "validation") & sc.strategy.isin(names)]
        .sort_values(["sharpe_ratio", "cagr"], ascending=False)
        .groupby("ticker")
        .head(1)
    )
    selected[["ticker", "strategy"]].to_csv(out / "selection.csv", index=False)
    tickers = sorted(selected.ticker.unique())
    sig = {}
    prices = {}
    for t in tickers:
        for name in names + ["Static_Ensemble"]:
            sig[t, name] = pd.read_parquet(
                research / "results" / t / name / "signals.parquet"
            )
        prices[t] = sig[t, names[0]].close
    prices = pd.DataFrame(prices).loc[p.evaluation_start :]
    base_weight = 0.8 / len(tickers)
    targets = pd.DataFrame(0.0, index=prices.index, columns=tickers)
    for row in selected.itertuples():
        targets[row.ticker] = (
            sig[row.ticker, row.strategy].filtered_signal.reindex(prices.index) == 2
        ) * base_weight
    momentum = pd.DataFrame(
        {
            t: (sig[t, names[0]].filtered_signal.reindex(prices.index) == 2)
            * base_weight
            for t in tickers
        }
    )
    ensemble = pd.DataFrame(
        {
            t: (sig[t, "Static_Ensemble"].filtered_signal.reindex(prices.index) == 2)
            * base_weight
            for t in tickers
        }
    )
    dynamic = ensemble.copy()
    from src.risk.policy_engine import PolicyEngine

    for t in tickers:
        f = pd.read_parquet(research / "features" / f"{t}_features.parquet")
        pol = PolicyEngine().compute_policy(f).reindex(prices.index)
        dynamic[t] *= pol.size_cap * pol.eligible
    # Meta selector only sees past virtual returns; holdings come from current
    # chosen strategy signals and the shared account, not blended P&L streams.
    from src.research.meta_strategy import MetaStrategy
    from src.backtest.simulator import run_simulation

    meta = targets * 0
    for t in tickers:
        returns = pd.DataFrame(
            {name: run_simulation(sig[t, name]).daily_returns for name in names}
        )
        feat = pd.read_parquet(research / "features" / f"{t}_features.parquet")
        _, allocation = MetaStrategy().run_ticker(
            returns, feat.hmm_regime.reindex(returns.index)
        )
        for date in prices.index:
            name = allocation.loc[date]
            meta.loc[date, t] = (
                base_weight
                if name in names and sig[t, name].loc[date, "filtered_signal"] == 2
                else 0.0
            )
    groups = {
        t: "growth"
        for t in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "XLK", "QQQ"]
        if t in tickers
    }
    limits = Limits(
        max_position=0.1,
        max_gross=0.8,
        max_turnover=0.4,
        groups=groups,
        group_limits={"growth": 0.35},
    )
    configs = {
        "validation_selected": targets,
        "momentum_12_1": momentum,
        "static_ensemble": ensemble,
        "dynamic_size_policy": dynamic,
        "causal_meta": meta,
    }
    save = lambda path, data: path.write_text(json.dumps(data, indent=2, default=str))
    save(
        out / "protocol.json",
        dict(
            capital=args.capital,
            limits=limits.__dict__,
            research_manifest_sha256=hashlib.sha256(
                (research / "manifest.json").read_bytes()
            ).hexdigest(),
            selection="max validation Sharpe, CAGR tie-break, zoo only",
            status="historical_replay_not_prospective",
            rule="complete next-close target vector; no brackets; EW fallback is cash",
            period_start=str(prices.index[0]),
            period_end=str(prices.index[-1]),
        ),
    )
    rows = []
    for name, weights in configs.items():
        print("Replay", name, flush=True)
        dest = out / name
        dest.mkdir()
        weights.to_parquet(dest / "targets.parquet")
        prices.to_parquet(dest / "prices.parquet")
        history, a = replay_portfolio(
            prices, weights, args.capital, limits, dest / "account.sqlite"
        )
        history.to_parquet(dest / "history.parquet")
        returns = history.equity.pct_change().fillna(0)
        fills = pd.DataFrame([e for e in a.events if e["kind"] == "fill"])
        fills.to_parquet(dest / "fills.parquet")
        spy = prices.SPY.pct_change().fillna(0)
        ev = evidence(returns, spy, len(configs), p)
        row = dict(
            strategy=name,
            **compute_metrics(returns),
            **ev,
            final_equity=float(history.equity.iloc[-1]),
            total_fees=a.fees,
            fill_count=len(fills),
            rejection_count=sum(e["kind"] == "rejection" for e in a.events),
            mean_gross_exposure=float(history.gross_exposure.mean()),
            turnover=(
                float((fills.qty * fills.price).sum() / args.capital)
                if len(fills)
                else 0.0
            ),
            max_reconciliation_error=float(history.reconciliation_error.abs().max()),
            minimum_cash=float(history.cash.min()),
            max_gross_exposure=float(history.gross_exposure.max()),
            pending_orders_at_end=len(a.pending()),
        )
        rows.append(row)
        save(dest / "summary.json", row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "portfolio-results.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
    for name in configs:
        h = pd.read_parquet(out / name / "history.parquet")
        axes[0].plot(h.index, h.equity, label=name)
        axes[1].plot(h.index, h.equity / h.equity.cummax() - 1, label=name)
    axes[0].plot(
        prices.index,
        args.capital * prices.SPY / prices.SPY.iloc[0],
        label="SPY frictionless buy-and-hold",
        color="black",
        linestyle="--",
    )
    axes[0].set_ylabel("Simulated account equity ($)")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_ylabel("Drawdown (fraction)")
    axes[1].set_xlabel("Date (completed XNYS session)")
    axes[0].set_title(
        "Corrected historical replay — exploratory, not prospective evidence"
    )
    fig.savefig(out / "portfolio-comparison.png", dpi=160)
    save(
        out / "output-hashes.json",
        {
            str(f.relative_to(out)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in out.rglob("*")
            if f.is_file()
        },
    )
    print(
        table[
            [
                "strategy",
                "final_equity",
                "sharpe_ratio",
                "max_drawdown",
                "fill_count",
                "max_reconciliation_error",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
