"""Export saved CECI2 research and model explanations for the historical simulator."""

import argparse
import json
import shutil
from pathlib import Path

import exchange_calendars as xcals
import joblib
import numpy as np
import pandas as pd
import torch

from src.models.gating import GatingModel
from .download import SYMBOLS
from .experts import SequenceExpert
from .panel import NUMERIC, TEXT, panel
from .train import market_states, observations

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "data" / "ceci2-demo"
RECEIPTS = Path(__file__).with_name("receipts")
PROB = ["down", "flat", "up"]
LANGUAGE = {
    "Qwen statement": ["qdown", "qflat", "qup"],
    "OpenAI asset digest": ["mdown", "mflat", "mup"],
    "OpenAI Decisions": ["odown", "oflat", "oup"],
    "Granite MoE statement": ["gdown", "gflat", "gup"],
}


def load(path):
    return json.loads(path.read_text())


def save(folder, name, data):
    path = folder / name
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False))
    temp.replace(path)


def records(frame):
    return json.loads(frame.to_json(orient="records", double_precision=12))


def days(index):
    return pd.DatetimeIndex(index).strftime("%Y-%m-%d").tolist()


def sources(cache):
    fed = pd.read_csv(RECEIPTS / "finbert-scores.csv")
    docs = pd.read_csv(RECEIPTS / "company-scores.csv")
    year = pd.to_datetime(docs.published, format="mixed", utc=True).dt.year
    counts = docs.groupby([year.rename("year"), "kind"]).size().unstack(fill_value=0)
    counts["fed"] = pd.to_datetime(fed.date.astype(str)).dt.year.value_counts()
    factors = pd.read_csv(cache / "jkp.zip", parse_dates=["date"])
    factor_panel = factors.pivot(index="date", columns="name", values="ret").loc[
        "2017":"2025"
    ]
    perf = (1 + factor_panel).groupby(factor_panel.index.year).prod() - 1
    return {
        "bars": 2262 * len(SYMBOLS),
        "symbols": len(SYMBOLS),
        "statements": len(fed),
        "headlines": int((docs.kind == "news").sum()),
        "filings": int((docs.kind == "filing").sum()),
        "factorThemes": list(factor_panel.columns),
        "first": days(factor_panel.index[:1])[0],
        "last": days(factor_panel.index[-1:])[0],
        "annualText": records(counts.reset_index()),
        "annualFactors": records(perf.rename_axis("year").reset_index()),
    }


def study(folder, horizon, coverage):
    ledger = pd.read_csv(folder / "portfolio.csv.gz")
    equity = ledger.pivot(
        index="decision", columns="model", values="equity"
    ).sort_index()
    marks = ledger.groupby("decision").mark.first().reindex(equity.index)
    reviews = load(folder / "adaptation.json")
    public_reviews = []
    for row in reviews:
        public = {k: v for k, v in row.items() if k not in ("agent", "proposal_prompt")}
        if row["agent"]:
            public["agent"] = {
                k: row["agent"][k]
                for k in ("service", "model", "option", "l2", "reason")
            }
        public_reviews.append(public)
    return {
        "horizon": horizon,
        "dates": equity.index.tolist(),
        "marks": marks.tolist(),
        "models": list(equity.columns),
        "equity": {name: equity[name].round(4).tolist() for name in equity},
        "summary": records(pd.read_csv(folder / "comparison.csv")),
        "annual": records(pd.read_csv(folder / "annual.csv")),
        "reviews": public_reviews,
        "coverage": coverage,
        "interval": load(folder / "run.json")["gate_minus_vote_95pct"],
    }


def expert_classes(rows, windows, folder, experts):
    """Recover earlier forecast correctness for the trained gate's delayed memory."""
    example = np.load(folder / "example.npz")
    x = ((windows - example["center"]) / example["scale"]).clip(-10, 10)
    width = len(NUMERIC)
    outputs = []
    for name in experts:
        if name in LANGUAGE:
            probabilities = rows[LANGUAGE[name]].to_numpy()
        elif (folder / f"{name}.pt").exists():
            model = SequenceExpert(name, width)
            model.load_state_dict(
                torch.load(folder / f"{name}.pt", weights_only=True, map_location="cpu")
            )
            model.eval()
            values = torch.tensor(x[:, :, :width].copy())
            with torch.inference_mode():
                probabilities = torch.cat(
                    [model(batch).softmax(-1) for batch in values.split(512)]
                ).numpy()
        else:
            values = x[:, -1, :width]
            if name == "JKP factors":
                values = x[:, -1, width + len(TEXT) :]
            elif name == "FinBERT logistic":
                values = x[:, -1, width : width + len(TEXT)]
            probabilities = joblib.load(folder / f"{name}.joblib").predict_proba(values)
        # Positive temperature scaling preserves class rankings.
        outputs.append(probabilities.argmax(-1))
    return np.stack(outputs, axis=1)


def explain_year(year, study_dir, rows, windows, market, forecasts, dates, symbols):
    folder = study_dir / f"models-{year}"
    experts = load(folder / "manifest.json")["experts"]
    eligible = np.zeros(len(rows), dtype=bool)
    for prior in (year - 1, year):
        sessions = rows.index[rows.index.year == prior].unique().sort_values()
        eligible |= (
            (rows.index.year == prior)
            & (rows.index >= sessions[5])
            & (rows.endpoint.dt.year == prior)
        )
    subset = rows[eligible].copy()
    classes = expert_classes(subset, windows[eligible], folder, experts)
    correct = classes == subset.label.to_numpy()[:, None]
    mature = (
        pd.DataFrame(correct, index=subset.endpoint)
        .groupby(level=0)
        .mean()
        .rolling(63, min_periods=1)
        .mean()
    )
    target = subset[subset.index.year == year]
    context = target[["vol", "range", "drawdown", "market"]].copy()
    memory = mature.reindex(target.index - pd.Timedelta(days=1), method="ffill").fillna(
        1 / 3
    )
    context[[f"accuracy_{i}" for i in range(len(experts))]] = memory.to_numpy()
    context[["fed_positive", "headline_tone", "annual_report_tone"]] = target[
        ["positive", "news_tone", "filing_tone"]
    ].to_numpy()
    regime = market_states(market, year - 1)
    context[["bear", "middle", "bull"]] = regime.reindex(target.index).to_numpy()
    weights = GatingModel.load(str(folder / "gate.joblib")).predict_weights(context)
    idx = pd.MultiIndex.from_arrays(
        [days(target.index), target.symbol], names=["decision", "symbol"]
    )
    keys = pd.MultiIndex.from_product([dates, symbols], names=["decision", "symbol"])
    cube = (
        pd.DataFrame(weights, index=idx)
        .reindex(keys)
        .to_numpy()
        .reshape(len(dates), len(symbols), len(experts))
    )
    probabilities = np.stack(
        [forecasts[name].reindex(keys)[PROB].to_numpy() for name in experts], axis=1
    )
    probabilities = probabilities.reshape(len(dates), len(symbols), len(experts), 3)
    mix = (cube[:, :, :, None] * probabilities).sum(axis=2)
    archived = forecasts["HMM gate"].reindex(keys)[PROB].to_numpy().reshape(mix.shape)
    error = np.max(np.abs(mix - archived))
    if not np.isfinite(error) or error > 1e-4:
        raise ValueError(
            f"Gate reconstruction differs from saved forecasts in {year}: {error:.8g}"
        )
    print(
        f"{year}: {len(dates)} decisions, maximum gate probability difference {error:.8g}",
        flush=True,
    )
    daily = cube.mean(axis=1).round(7).tolist()
    posterior = regime.reindex(pd.to_datetime(dates)).to_numpy().round(7).tolist()
    detail = {
        "dates": dates,
        "symbols": symbols,
        "experts": experts,
        "weights": cube.round(6).tolist(),
        "probabilities": probabilities.round(6).tolist(),
        "contextNames": [
            "vol",
            "range",
            "drawdown",
            "market",
            "fed_positive",
            "headline_tone",
            "annual_report_tone",
        ],
    }
    scalar_context = context[detail["contextNames"]].copy()
    scalar_context.index = idx
    detail["context"] = (
        scalar_context.reindex(keys)
        .to_numpy()
        .reshape(len(dates), len(symbols), -1)
        .round(6)
        .tolist()
    )
    detail["recentAccuracy"] = (
        memory.iloc[np.flatnonzero(~target.index.duplicated())]
        .to_numpy()
        .round(6)
        .tolist()
    )
    return detail, daily, posterior


def export(output, cache):
    output.mkdir(parents=True, exist_ok=True)
    # The export is offline and must never trigger fresh data or model acquisition.
    required = [
        "jkp.zip",
        "company-scores.csv",
        "finbert-scores.csv",
        "statements.json",
        *[f"{s}.json" for s in SYMBOLS],
    ]
    if any(not (cache / f).is_file() for f in required):
        raise ValueError(
            "Prepare the registered study cache before exporting the website"
        )
    coverage = sources(cache)
    for name, horizon in (("five-session", 5), ("one-session", 1)):
        save(output, f"study-{horizon}.json", study(STUDIES / name, horizon, coverage))
    folder = STUDIES / "five-session"
    tape = pd.read_csv(folder / "predictions.csv.gz", float_precision="round_trip")
    forecasts = {
        name: frame.set_index(["decision", "symbol"]).sort_index()
        for name, frame in tape.groupby("model")
    }
    gated = forecasts["HMM gate"]
    dates = gated.index.get_level_values("decision").unique().tolist()
    symbols = gated.index.get_level_values("symbol").unique().tolist()
    keys = pd.MultiIndex.from_product([dates, symbols], names=["decision", "symbol"])
    if not gated.index.equals(keys):
        raise ValueError("Missing or duplicate symbol sessions in simulation input")
    market, _ = panel(cache)
    windows, rows = observations(market)
    torch.set_num_threads(1)
    sim = {
        "dates": dates,
        "symbols": symbols,
        "weights": [],
        "posterior": [],
        "marks": gated.groupby(level=0).mark.first().tolist(),
        "regime": gated.groupby(level=0).regime.first().astype(int).tolist(),
    }
    calendar = xcals.get_calendar("XNYS", start="2022-12-01", end="2026-01-31")
    sim["entryDates"] = days(
        [calendar.next_session(pd.Timestamp(date)) for date in dates]
    )
    for col in (
        "entry",
        "exit",
        "vol",
        "down",
        "flat",
        "up",
        "close",
        "atr",
        "volume",
        "cap",
        "threshold",
    ):
        # Exact stored double precision preserves ties and trade decisions in replay.
        sim[col] = gated[col].to_numpy().reshape(len(dates), len(symbols)).tolist()
    orders = pd.read_csv(folder / "dynamic_orders.csv.gz")
    sim["orders"] = records(orders)
    for year in (2023, 2024, 2025):
        selected = [d for d in dates if d.startswith(str(year))]
        detail, weights, posterior = explain_year(
            year, folder, rows, windows, market["SPY"], forecasts, selected, symbols
        )
        sim["weights"].extend(weights)
        sim["posterior"].extend(posterior)
        sim["experts"] = detail["experts"]
        save(output, f"detail-{year}.json", detail)
    save(output, "simulation.json", sim)
    for name in ("datasets.png", "sources.png"):
        shutil.copyfile(folder / name, output / name)
    shutil.copyfile(STUDIES / "robustness/site.json", output / "stress.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/raw/ceci2-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "site/assets")
    args = parser.parse_args()
    export(args.output, args.cache)
