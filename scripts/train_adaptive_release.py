"""Run the registered rolling experiment and create a sealed local release.

Consumes an immutable download. Never writes legacy caches or sends orders.
"""

from pathlib import Path
import argparse
import importlib.metadata
import json
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.snapshot import load_snapshot, digest
from src.models.adaptive import AdaptiveModel, features, walk_forward
from src.research.adaptive_portfolio import CANDIDATES, limits, target_panels
from src.research.protocol import evidence
from src.backtest.metrics import compute_metrics
from src.backtest.portfolio import replay_portfolio

PERIODS = {
    "development": ("2024-01-02", "2025-12-31"),
    "selection": ("2026-01-02", "2026-07-10"),
    "audit": ("2026-07-13", "2026-09-25"),
}


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False))


def classification_metrics(predictions):
    rows, accuracy_days = [], []
    for period, (start, end) in PERIODS.items():
        pieces = []
        for symbol, frame in predictions.items():
            f = frame.loc[start:end].copy()
            f = f.loc[f.label >= 0]
            f["symbol"] = symbol
            pieces.append(f)
        df = pd.concat(pieces).sort_index()
        if df.empty:
            raise ValueError(f"No classification observations in {period}")
        y = df.label.to_numpy()
        for name in ["linear", "tree", "hybrid", "prior", "always_up"]:
            p = (
                df[[f"{name}_{c}" for c in range(3)]].to_numpy()
                if name != "always_up"
                else np.tile([1e-12, 1e-12, 1 - 2e-12], (len(df), 1))
            )
            pred = p.argmax(axis=1)
            dates = (
                pd.DataFrame(
                    {"accuracy_minus_up": (pred == y).astype(float) - (y == 2)},
                    index=df.index,
                )
                .groupby(level=0)
                .mean()
            )
            # Resample dates, not individual stocks: cross-sectional dependence
            # is retained, and overlapping five-session labels remain blocked.
            rng = np.random.default_rng(42)
            values = dates.accuracy_minus_up.to_numpy()
            boot = []
            for _ in range(1000):
                starts = rng.integers(0, len(values), int(np.ceil(len(values) / 10)))
                ids = ((starts[:, None] + np.arange(10)) % len(values)).ravel()[
                    : len(values)
                ]
                boot.append(values[ids].mean())
            lo, hi = np.quantile(boot, [0.025, 0.975])
            rows.append(
                dict(
                    period=period,
                    model=name,
                    n=len(y),
                    sessions=len(dates),
                    accuracy=accuracy_score(y, pred),
                    balanced_accuracy=balanced_accuracy_score(y, pred),
                    macro_f1=f1_score(
                        y, pred, labels=[0, 1, 2], average="macro", zero_division=0
                    ),
                    log_loss=log_loss(y, p, labels=[0, 1, 2]),
                    brier=float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1))),
                    accuracy_gain_over_up=float(values.mean()),
                    gain_ci_low=lo,
                    gain_ci_high=hi,
                )
            )
            dates["model"], dates["period"] = name, period
            accuracy_days.append(dates)
    return pd.DataFrame(rows), pd.concat(accuracy_days)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--release", type=Path, required=True)
    args = p.parse_args()
    out = args.release.resolve()
    if (out / "experiment").exists():
        raise FileExistsError("Refusing to overwrite an experiment")
    registration = json.loads((out / "registration.json").read_text())
    if digest(out / "protocol.md") != registration["protocol_sha256"]:
        raise ValueError("Registered protocol changed")
    data, market = load_snapshot(out / "market", require_fresh=True)
    if pd.Timestamp(registration["registered_at"]) > pd.Timestamp(
        market["retrieved_at"]
    ):
        raise ValueError("Protocol registration must precede new data retrieval")
    if any(f.index[-1] < pd.Timestamp("2026-09-25", tz="UTC") for f in data.values()):
        raise ValueError("The registered audit window is incomplete")
    # This command is a dated experiment, never an open-ended moving holdout.
    data = {s: f.loc[:"2026-09-25"] for s, f in data.items()}
    common = data["SPY"].index
    for frame in data.values():
        common = common.intersection(frame.index)
    data = {s: f.loc[common] for s, f in sorted(data.items())}
    exp = out / "experiment"
    exp.mkdir()
    (out / "models").mkdir()
    source = out / "source"
    code = {}
    for pattern in ["src/**/*.py", "scripts/*.py", "requirements.txt"]:
        for f in ROOT.glob(pattern):
            rel = f.relative_to(ROOT)
            (source / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, source / rel)
            code[str(rel)] = digest(f)
    save(
        out / "dependencies.json",
        {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        },
    )
    manifest = dict(
        status="training",
        created_at=pd.Timestamp.now(tz="UTC").isoformat(),
        source=code,
        symbols=list(data),
        asof=str(common[-1]),
        capital=10000,
        protocol_sha256=registration["protocol_sha256"],
        periods=PERIODS,
        kind="retrospective_adaptive_experiment_and_local_shadow_release",
    )
    save(out / "manifest.json", manifest)
    predictions, folds = {}, {}
    with threadpool_limits(limits=1):
        for symbol, raw in data.items():
            print(f"Rolling training {symbol}", flush=True)
            pred, fold, _ = walk_forward(raw, data["SPY"])
            predictions[symbol], folds[symbol] = pred, fold
            pred.to_parquet(exp / f"{symbol}_predictions.parquet")
            save(exp / f"{symbol}_folds.json", fold)
            deploy = AdaptiveModel().fit_at(
                features(raw, data["SPY"]), raw.Close, raw.index[-1]
            )
            deploy.save(out / "models" / symbol)
            # Loading and predicting tests serialization with the actual model.
            restored = AdaptiveModel.load(out / "models" / symbol)
            restored.predict(features(raw, data["SPY"]).iloc[[-1]])
    classification, days = classification_metrics(predictions)
    classification.to_csv(exp / "classification.csv", index=False)
    days.to_parquet(exp / "accuracy_by_date.parquet")
    prices, panels = target_panels(data, predictions)
    prices.to_parquet(exp / "prices.parquet")
    rebalance_dates = prices.index[
        (prices.index.weekday == 4) | (prices.index == prices.index[0])
    ]
    histories, accounts = {}, {}
    for name, targets in panels.items():
        print(f"Shared account replay {name}", flush=True)
        targets.to_parquet(exp / f"{name}_targets.parquet")
        hist, account = replay_portfolio(
            prices,
            targets,
            limits=limits(),
            state_path=exp / f"{name}.sqlite",
            rebalance_dates=rebalance_dates,
        )
        hist.to_parquet(exp / f"{name}_account.parquet")
        save(exp / f"{name}_events.json", account.events)
        histories[name], accounts[name] = hist, account
    returns = pd.DataFrame(
        {
            n: f.equity.pct_change(fill_method=None).fillna(0)
            for n, f in histories.items()
        }
    )
    returns["spy_reference"] = (
        data["SPY"].Close.pct_change(fill_method=None).loc[returns.index]
    )
    returns.iloc[0, returns.columns.get_loc("spy_reference")] = 0
    returns.to_parquet(exp / "returns.parquet")
    records = []
    for period, (start, end) in PERIODS.items():
        for name in returns:
            r = returns[name].loc[start:end]
            ev = evidence(r, returns.equal_weight.loc[r.index], 333)
            records.append(
                dict(period=period, strategy=name, **compute_metrics(r), **ev)
            )
    results = pd.DataFrame(records)
    results.to_csv(exp / "portfolio_metrics.csv", index=False)
    selection = results.loc[
        (results.period == "selection") & results.strategy.isin(CANDIDATES)
    ]
    winner = (
        selection.sort_values(["sharpe_ratio", "strategy"], ascending=[False, True])
        .iloc[0]
        .strategy
    )
    audit = results.loc[
        (results.period == "audit") & (results.strategy == winner)
    ].iloc[0]
    selected = selection.loc[selection.strategy == winner].iloc[0]
    historical_edge = bool(
        audit.positive_evidence and audit.total_return > 0 and selected.total_return > 0
    )
    cost_stress = []
    for multiple in [2, 4]:
        for name in CANDIDATES:
            hist, _ = replay_portfolio(
                prices,
                panels[name],
                limits=limits(multiple),
                rebalance_dates=rebalance_dates,
            )
            r = (
                hist.equity.pct_change(fill_method=None)
                .fillna(0)
                .loc["2026-07-13":"2026-09-25"]
            )
            cost_stress.append(
                dict(cost_multiplier=multiple, strategy=name, **compute_metrics(r))
            )
    save(exp / "cost_stress.json", cost_stress)
    diagnostic = dict(
        folds=sum(len(v) for v in folds.values()),
        max_reconciliation_error=max(
            float(f.reconciliation_error.abs().max()) for f in histories.values()
        ),
        minimum_cash=min(float(f.cash.min()) for f in histories.values()),
        fills={
            n: sum(e["kind"] == "fill" for e in a.events) for n, a in accounts.items()
        },
        all_label_cutoffs_precede_decisions=all(
            pd.Timestamp(f["latest_label_end"]) < pd.Timestamp(f["decision"])
            for fs in folds.values()
            for f in fs
        ),
        audit_sessions=int(audit.n_days),
        selected_candidate=winner,
    )
    save(exp / "diagnostics.json", diagnostic)
    release = dict(
        version=1,
        candidate=winner,
        approved_strategy="cash",
        historical_edge=historical_edge,
        prospective_start="2026-09-29",
        prospective_required_sessions=63,
        prospective_observed_sessions=0,
        external_orders_authorized=False,
        software_contract="pending_test_evidence",
        market_data="approved",
        model_bundles="approved",
        combined_readiness="not_approved",
        profitable_trader_claim="not_approved",
        reason=(
            "Prospective evidence is absent"
            if historical_edge
            else "Historical edge gate failed and prospective evidence is absent"
        ),
        asof=str(common[-1]),
        symbols=list(data),
        capital=10000,
        price_convention=market["price_convention"],
        diagnostics=diagnostic,
    )
    save(out / "readiness.json", release)
    manifest["status"] = "complete"
    manifest["completed_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    manifest["artifacts"] = {
        str(f.relative_to(out)): digest(f)
        for f in out.rglob("*")
        if f.is_file()
        and not f.name.endswith(".log")
        and f != out / "manifest.json"
        and "source" not in f.relative_to(out).parts
    }
    save(out / "manifest.json", manifest)
    print(json.dumps(release, indent=2), flush=True)


if __name__ == "__main__":
    main()
