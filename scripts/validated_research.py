"""Offline corrected historical rerun. Existing artifacts are never overwritten.
Run: python scripts/validated_research.py --output data/repairs/RUN --ml
The historical evaluation is exploratory: its dates were inspected previously.
The future prospective holdout remains unopened. No network or broker calls.
"""

from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.research.protocol import Protocol, evidence
from src.backtest.simulator import run_simulation
from src.backtest.metrics import compute_metrics
from src.data.validation import observed_equity_bars


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ml", action="store_true")
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    p = Protocol()
    save_json(out / "protocol.json", p.as_dict())
    sources = {}
    for pattern in ["src/**/*.py", "scripts/*.py", "requirements.txt"]:
        for f in ROOT.glob(pattern):
            rel = f.relative_to(ROOT)
            target = out / "source" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            sources[str(rel)] = hashlib.sha256(f.read_bytes()).hexdigest()
    deps = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    }
    save_json(out / "dependencies.json", deps)
    (out / "requirements-resolved.txt").write_text(
        "\n".join(f"{k}=={v}" for k, v in sorted(deps.items())) + "\n"
    )
    paths = sorted((ROOT / "data/parquet").glob("*_daily.parquet"))
    paths = [
        f
        for f in paths
        if f.stem != "macro_daily" and (not args.tickers or f.stem[:-6] in args.tickers)
    ]
    data = {}
    provenance = {}
    (out / "snapshots").mkdir()
    for f in paths:
        ticker = f.stem[:-6]
        raw = pd.read_parquet(f)
        clean = observed_equity_bars(raw)
        data[ticker] = clean
        shutil.copy2(f, out / "snapshots" / f.name)
        provenance[ticker] = dict(
            sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
            raw_rows=len(raw),
            observed_rows=len(clean),
            start=str(clean.index[0]),
            end=str(clean.index[-1]),
        )
    if "SPY" not in data:
        raise ValueError("SPY required for matching benchmark")
    common = data["SPY"].index
    for df in data.values():
        common = common.intersection(df.index)
    common = common.sort_values()
    data = {t: df.loc[common] for t, df in data.items()}
    manifest = dict(
        status="running",
        protocol=p.as_dict(),
        code=sources,
        data=provenance,
        command=sys.argv,
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        classification="corrected_historical_exploration_not_untouched_holdout",
        unavailable={
            "macro": "no historical release/vintage evidence",
            "sentiment": "no immutable article availability/content vintages",
            "crypto": "provider history does not cover common training period",
        },
        ml_budget=dict(trials=args.trials, epochs=args.epochs),
        created_utc=pd.Timestamp.now(tz="UTC").isoformat(),
    )
    save_json(out / "manifest.json", manifest)
    from src.research.strategies import ALL_STRATEGIES

    records = []
    signals = {}
    curves = {}
    classification = []
    failures = []
    family_size = len(data) * (len(ALL_STRATEGIES) + (7 if args.ml else 0))

    def evaluate(ticker, name, signal_df, trade_result=None):
        sig = signal_df.loc[p.validation_start :]
        signals[(ticker, name)] = sig
        base = (
            run_simulation(sig, commission=p.commission, slippage=p.slippage)
            if trade_result is None
            else trade_result
        )
        path = out / "results" / ticker / name
        path.mkdir(parents=True, exist_ok=True)
        sig.to_parquet(path / "signals.parquet")
        base.ledger.to_parquet(path / "ledger.parquet")
        base.trade_log.to_parquet(path / "trades.parquet")
        base.equity_curve.to_frame().to_parquet(path / "equity.parquet")
        for period, start, end in [
            ("validation", p.validation_start, p.validation_end),
            ("evaluation", p.evaluation_start, None),
        ]:
            r = base.daily_returns.loc[start:end]
            # Both return sequences include the move from the preceding observed
            # close; only the first inception return is zero for both accounts.
            bh = data[ticker].Close.pct_change(fill_method=None).loc[r.index]
            if period == "validation":
                bh.iloc[0] = 0.0
            log = base.trade_log
            log = log.loc[pd.to_datetime(log.date).isin(r.index)] if len(log) else log
            metrics = compute_metrics(r, log)
            ev = evidence(r, bh, family_size, p)
            row = dict(
                ticker=ticker,
                strategy=name,
                period=period,
                start=str(r.index[0]),
                end=str(r.index[-1]),
                **metrics,
                **ev,
            )
            row["qualifies"] = bool(
                ev["positive_evidence"]
                and metrics.get("n_closed_trades", 0) >= p.min_closed_trades
                and metrics["cagr"] > 0
                and metrics["sharpe_ratio"] > 0
            )
            if len(base.ledger):
                led = base.ledger.loc[r.index]
                row["mean_gross_exposure"] = float(
                    (abs(led.shares) * led.price / led.equity).mean()
                )
                row["turnover"] = float(
                    (
                        abs(led.shares.diff().fillna(led.shares.iloc[0]))
                        * led.price
                        / led.equity
                    ).sum()
                )
            records.append(row)
        curves[(ticker, name)] = base.daily_returns
        # Cost stress keeps decisions fixed; it does not search for a better cost.
        stress = []
        for mult in [0, 1, 2, 4]:
            if trade_result is not None:
                continue
            r = run_simulation(
                sig, commission=p.commission * mult, slippage=p.slippage * mult
            )
            stress.append(
                dict(
                    cost_multiplier=mult,
                    **compute_metrics(r.daily_returns.loc[p.evaluation_start :]),
                )
            )
        save_json(path / "cost_sensitivity.json", stress)

    for ticker, df in data.items():
        print("Zoo", ticker, flush=True)
        for strategy in ALL_STRATEGIES:
            name = {
                "Pairs_StatArb": "Relative_Value_Long_Leg",
                "Carry_Proxy": "SMA_200_Trend_Fallback",
                "Momentum_12_1": "Time_Series_Momentum_12_1",
            }.get(strategy.name, strategy.name)
            sig = strategy.build_signal_df(df, pair_df=data["SPY"], macro=None)
            evaluate(ticker, name, sig)
        for name, sig in [("Cash", 1), ("Buy_Hold", 2)]:
            evaluate(
                ticker,
                name,
                pd.DataFrame(
                    {"close": df.Close, "filtered_signal": sig}, index=df.index
                ),
            )
    if args.ml:
        from src.features.pipeline import FeatureEngineer
        from src.models.trainer import ModelTrainer
        from src.models.gating import build_gate_context, GatingModel
        from src.risk.engine import RiskEngine
        from src.risk.policy_engine import PolicyEngine
        from src.models.labels import make_labels

        for ticker, df in data.items():
            print("ML", ticker, flush=True)
            try:
                feat = FeatureEngineer(cache_dir=out / "features").compute_features(
                    df, ticker, spy_df=data["SPY"]
                )
                trainer = ModelTrainer(
                    models_dir=out / "models",
                    mlflow_uri="sqlite:///" + str(out / "tracking.db"),
                )
                trainer.run(
                    feat.loc[: p.training_end],
                    ticker=ticker,
                    horizon=p.horizon,
                    threshold=p.threshold,
                    n_splits=2,
                    n_trials_xgb=args.trials,
                    n_trials_lgbm=args.trials,
                    lstm_epochs=args.epochs,
                    top_n_features=25,
                )
                xgb, lgbm, lstm, cols = trainer.load_models(ticker)
                X = feat[cols].dropna()
                probs = [
                    m.predict_proba(X.to_numpy(dtype=np.float32))
                    for m in [xgb, lgbm, lstm]
                ]
                good = X.index >= pd.Timestamp(p.validation_start, tz="UTC")
                probs = [v[good] for v in probs]
                index = X.index[good]
                y, _ = make_labels(df, horizon=p.horizon, threshold=p.threshold)
                y = y.loc[index]
                context = build_gate_context(
                    feat, probs, y.to_numpy(), index=index, horizon=p.horizon
                )
                gate = GatingModel(min_train=100)
                train_idx = np.flatnonzero(
                    index <= pd.Timestamp(p.validation_end, tz="UTC")
                )[: -p.horizon - 1]
                correct = np.column_stack(
                    [np.argmax(v, axis=1) == y.to_numpy() for v in probs]
                ).astype(int)
                gate.fit(context.iloc[train_idx], correct[train_idx])
                gate.save(str(out / "models" / ticker / "gate.pkl"))
                weights = gate.predict_weights(context)
                static = np.mean(probs, axis=0)
                moe = sum(weights[:, m, None] * v for m, v in enumerate(probs))
                for name, pr in list(zip(["XGB", "LGBM", "LSTM"], probs)) + [
                    ("Static_Ensemble", static),
                    ("MoE", moe),
                ]:
                    raw = np.argmax(pr, axis=1)
                    conf = pr.max(axis=1)
                    sig = pd.DataFrame(
                        {
                            "close": df.Close.loc[index],
                            "filtered_signal": np.where(conf >= 0.38, raw, 1),
                            "p_up": pr[:, 2],
                        },
                        index=index,
                    )
                    # Gate was fitted on validation outcomes; never report that
                    # fitted slice as OOS MoE validation performance.
                    if name == "MoE":
                        sig.loc[: p.validation_end, "filtered_signal"] = 1
                    evaluate(ticker, name, sig)
                    valid = (index >= pd.Timestamp(p.evaluation_start, tz="UTC")) & (
                        y.to_numpy() >= 0
                    )
                    classification.append(
                        dict(
                            ticker=ticker,
                            model=name,
                            n=int(valid.sum()),
                            accuracy=float((raw[valid] == y.to_numpy()[valid]).mean()),
                            always_up_accuracy=float((y.to_numpy()[valid] == 2).mean()),
                            majority_train_accuracy=float(
                                (
                                    y.to_numpy()[valid]
                                    == make_labels(
                                        feat.loc[: p.training_end],
                                        horizon=p.horizon,
                                        threshold=p.threshold,
                                    )[0]
                                    .value_counts()
                                    .idxmax()
                                ).mean()
                            ),
                        )
                    )
                base_sig = signals[(ticker, "Static_Ensemble")]
                for name, policy in [
                    ("Static_Risk", None),
                    ("Dynamic_Risk", PolicyEngine()),
                ]:
                    result = RiskEngine().run(
                        ticker, base_sig, feat, policy_engine=policy
                    )
                    from src.backtest.simulator import SimResult

                    risk = SimResult(
                        result.equity_curve,
                        result.daily_returns,
                        result.trade_log,
                        result.params,
                        result.ledger,
                    )
                    evaluate(ticker, name, base_sig, trade_result=risk)
            except Exception as exc:
                import traceback

                failures.append(
                    dict(
                        ticker=ticker, error=str(exc), traceback=traceback.format_exc()
                    )
                )
                save_json(out / "failures.json", failures)
                print("FAILED", ticker, str(exc), flush=True)
    table = pd.DataFrame(records)
    table.to_csv(out / "scorecard.csv", index=False)
    pd.DataFrame(classification).to_csv(out / "classification.csv", index=False)
    selection = (
        table[
            (table.period == "validation")
            & ~table.strategy.isin(["Cash", "Buy_Hold", "MoE"])
        ]
        .sort_values(["sharpe_ratio", "cagr"], ascending=False)
        .groupby("ticker")
        .head(1)
    )
    selection[["ticker", "strategy", "sharpe_ratio"]].to_csv(
        out / "validation_selection.csv", index=False
    )
    save_json(out / "failures.json", failures)
    save_json(
        out / "output-hashes.json",
        {
            str(f.relative_to(out)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in out.rglob("*")
            if f.is_file() and f.name not in ["manifest.json", "output-hashes.json"]
        },
    )
    manifest.update(
        status="failed" if failures else "complete",
        finished_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        credible_edge_established=False,
        reason="Historical period previously inspected; independent prospective holdout required",
        qualified_exploratory_rows=table[
            (table.period == "evaluation") & table.qualifies
        ][["ticker", "strategy"]].to_dict("records"),
    )
    save_json(out / "manifest.json", manifest)
    if failures:
        raise SystemExit(1)
    print(out, flush=True)


if __name__ == "__main__":
    main()
