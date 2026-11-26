"""Run the registered CECI2 experiment and write its figures and research ledger."""

import argparse
import json
import platform
import shutil
import os
import getpass
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from .download import fetch
from .evaluate import interval, summarize
from .panel import panel
from .train import fold, observations


def figures(frames, records, ledger, output):
    plt.rcParams.update(
        {"axes.spines.top": False, "axes.spines.right": False, "font.size": 10}
    )
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), layout="constrained")
    for symbol in ("SPY", "IWM", "TLT", "GLD", "QQQ"):
        series = frames[symbol].close
        axes[0].plot(series.index, series / series.iloc[0] * 100, label=symbol)
    axes[0].set(
        title="Selected markets since 2017", ylabel="Adjusted close (first day = 100)"
    )
    axes[0].legend(ncols=5)
    dates = pd.to_datetime(records.date.astype(str))
    axes[1].bar(dates, records.positive - records.negative, width=18, color="#41796c")
    axes[1].axhline(0, color="0.5", linewidth=0.7)
    axes[1].set(
        title="Federal Reserve statements", ylabel="FinBERT positive minus negative"
    )
    fig.savefig(output / "datasets.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    for name in (
        "SPY buy and hold",
        "Equal weight",
        "Cash",
        "Uniform vote",
        "HMM gate",
    ):
        series = ledger.loc[ledger.model == name]
        ax.plot(series.mark, series.equity, label=name, linewidth=1.6)
    ax.set(
        title="Historical portfolio value after costs", ylabel="USD, starting at 10,000"
    )
    ax.legend(ncols=3)
    fig.savefig(output / "backtest.png", dpi=150)
    plt.close(fig)


def coverage(folder, output):
    import numpy as np

    market, records = (
        pd.read_csv(folder / "company-scores.csv"),
        pd.read_csv(folder / "finbert-scores.csv"),
    )
    by_year = (
        market.assign(
            year=pd.to_datetime(market.published, utc=True, format="mixed").dt.year
        )
        .groupby(["year", "kind"])
        .size()
        .unstack(fill_value=0)
    )
    dates = pd.to_datetime(records.date.astype(str))
    by_year["Fed statements"] = dates.groupby(dates.dt.year).size()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    by_year.rename(
        columns={"filing": "Annual reports", "news": "Sampled headlines"}
    ).plot.bar(ax=axes[0], rot=0, color=["#b87543", "#41796c", "#527fb1"])
    axes[0].set(
        title="Text observations admitted by source", xlabel="", ylabel="Documents"
    )
    factors = pd.read_csv(folder / "jkp.zip", parse_dates=["date"]).pivot(
        index="date", columns="name", values="ret"
    )
    annual = (1 + factors.loc["2017":"2025"]).groupby(lambda d: d.year).prod() - 1
    colors = axes[1].imshow(annual.T, aspect="auto", cmap="RdBu", vmin=-0.35, vmax=0.35)
    axes[1].set(
        yticks=np.arange(len(annual.columns)),
        yticklabels=[c.replace("_", " ") for c in annual.columns],
        xticks=np.arange(len(annual)),
        xticklabels=annual.index,
        title="JKP US factor-theme returns by year",
    )
    fig.colorbar(colors, ax=axes[1], label="Compounded return")
    fig.savefig(output / "sources.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/raw/ceci2-v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, choices=(1, 5), default=5)
    parser.add_argument(
        "--hosted-review",
        action="store_true",
        help="Request a new paid agent proposal for each fold",
    )
    parser.add_argument(
        "--rebuild-text",
        action="store_true",
        help="Recompute the text features from source documents and open weights in a new cache",
    )
    parser.add_argument(
        "--refresh-adjustments",
        action="store_true",
        help="Use the latest vendor adjustment ratios for a new experiment",
    )
    args = parser.parse_args()
    if args.hosted_review:
        os.environ["CECI2_HOSTED_REVIEW"] = "1"
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key: ")
    args.cache.mkdir(parents=True, exist_ok=True)
    receipts = Path(__file__).with_name("receipts")
    if not args.rebuild_text:
        for filename in ("statements.json", "finbert-scores.csv", "company-scores.csv"):
            src = receipts / filename
            if src.is_file() and not (args.cache / filename).exists():
                shutil.copyfile(src, args.cache / filename)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory for this run")
    torch.set_num_threads(1)
    fetch(args.cache)
    frames, records = panel(args.cache, args.horizon, args.refresh_adjustments)
    windows, rows = observations(frames)
    protocol_path = Path(__file__).with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    blocks, reviews = [], []
    for year in protocol["years"]:
        print(f"Training and evaluating fold {year}", flush=True)
        forecasts, receipts = fold(
            windows,
            rows,
            frames["SPY"],
            year,
            args.cache,
            args.horizon,
            args.output / f"models-{year}",
        )
        blocks.append(forecasts)
        reviews.extend(receipts)
    (args.output / "adaptation.json").write_text(json.dumps(reviews, indent=2) + "\n")
    predictions = pd.concat(blocks, ignore_index=True)
    summary, ledger, orders = summarize(predictions)
    orders.to_csv(args.output / "dynamic_orders.csv.gz", index=False)
    classified = predictions.assign(
        hit=predictions[["down", "flat", "up"]].to_numpy().argmax(1)
        == predictions.label.to_numpy()
    )
    classified.groupby(["model", "symbol", "regime", "label"]).hit.agg(
        ["size", "mean"]
    ).to_csv(args.output / "classification_by_group.csv")
    yearly = ledger.assign(year=ledger.decision.dt.year).groupby(["model", "year"])
    yearly.agg(
        sessions=("net_return", "size"),
        net_return=("net_return", lambda r: (1 + r).prod() - 1),
        fees=("fees", "sum"),
        exposure=("exposure", "mean"),
    ).to_csv(args.output / "annual.csv")
    from .diagnostics import diagnostics

    diagnostics(predictions, ledger, args.output)
    summary.to_csv(args.output / "comparison.csv", index=False)
    ledger.to_csv(args.output / "portfolio.csv.gz", index=False)
    predictions.to_csv(args.output / "predictions.csv.gz", index=False)
    report = dict(
        protocol=protocol,
        horizon=args.horizon,
        adjustments="fresh"
        if args.refresh_adjustments
        else "registered vendor ratio snapshot",
        python=platform.python_version(),
        versions={
            n: version(n)
            for n in (
                "torch",
                "transformers",
                "numpy",
                "pandas",
                "scikit-learn",
                "xgboost",
                "lightgbm",
                "hmmlearn",
            )
        },
        market={
            s: dict(
                bars=len(f),
                start=str(f.index.min().date()),
                end=str(f.index.max().date()),
                source=json.loads((args.cache / f"{s}.json").read_text())["url"],
            )
            for s, f in frames.items()
        },
        statements=len(records),
        sessions=predictions.decision.nunique(),
        gate_minus_vote_95pct=interval(predictions, ledger),
    )
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    figures(frames, records, ledger, args.output)
    coverage(args.cache, args.output)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
