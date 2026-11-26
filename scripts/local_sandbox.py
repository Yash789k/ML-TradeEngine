"""Clean-artifact bootstrap and deterministic local replay; no credentials/network."""

from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.validation import calendar, observed_equity_bars
from src.features.pipeline import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.models.bundle import validate_bundle
from src.execution.account import Account, Limits, ExecutionSession
from src.backtest.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capital", type=float, default=10000.0)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    idx = (
        calendar()
        .sessions_in_range("2020-01-01", "2023-12-29")[:800]
        .tz_localize("UTC")
    )
    close = 100 * np.exp(rng.normal(0.0003, 0.012, len(idx)).cumsum())
    raw = pd.DataFrame(
        dict(
            Open=close,
            High=close * 1.01,
            Low=close * 0.99,
            Close=close,
            Adj_Close=close,
            Volume=np.full(len(idx), 1000000),
        ),
        index=idx,
    )
    raw = observed_equity_bars(raw)
    raw.to_parquet(out / "synthetic-input.parquet")
    feat = FeatureEngineer(cache_dir=out / "features").compute_features(
        raw, "SYNTH", spy_df=raw
    )
    trainer = ModelTrainer(
        models_dir=out / "models", mlflow_uri="sqlite:///" + str(out / "tracking.db")
    )
    trainer.run(
        feat.iloc[:600],
        ticker="SYNTH",
        n_splits=2,
        n_trials_xgb=1,
        n_trials_lgbm=1,
        lstm_epochs=2,
    )
    validate_bundle(out / "models/SYNTH")
    import src.live.signal_generator as generator

    generator._MODELS_ROOT = out / "models"
    pr = generator._infer_latest_bar(
        "SYNTH", feat, json.loads((out / "models/SYNTH/feature_cols.json").read_text())
    )
    acct = Account(
        ["SYNTH"], args.capital, Limits(max_position=0.5), out / "account.sqlite"
    )
    session = ExecutionSession(acct)
    rows = []
    for i, date in enumerate(idx[-60:]):
        # Exercise deterministic order lifecycle independent of model quality.
        w = 0.3 if i % 12 < 7 else 0.0
        row = session.on_close(
            date, {"SYNTH": float(raw.loc[date, "Close"])}, {"SYNTH": w}
        )
        rows.append(
            dict(date=date, **{k: v for k, v in row.items() if k != "positions"})
        )
    history = pd.DataFrame(rows).set_index("date")
    history.to_parquet(out / "account-history.parquet")
    restored = Account(
        ["SYNTH"], args.capital, Limits(max_position=0.5), out / "account.sqlite"
    )
    assert restored.snapshot({"SYNTH": float(close[-1])}) == acct.snapshot(
        {"SYNTH": float(close[-1])}
    )
    report = dict(
        kind="synthetic_operational_smoke_not_performance_evidence",
        capital=args.capital,
        model_inference=pr,
        events=len(acct.events),
        fills=sum(e["kind"] == "fill" for e in acct.events),
        reconciliation_max=float(history.reconciliation_error.abs().max()),
        metrics=compute_metrics(history.equity.pct_change().fillna(0)),
        external_orders=0,
    )
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
