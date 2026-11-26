"""Export comparable historical periods from archived, costed daily decisions."""

import csv
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


DATA = Path("data/ceci2-demo")
OUT = Path("site/assets/periods.json")


def read_ledger(path):
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            daily_return = float(row["net_return"])
            opening_equity = float(row["equity"]) / (1 + daily_return)
            record = {
                "mark": row["mark"],
                "net_return": daily_return,
                "fee_rate": float(row["fees"]) / opening_equity,
                "exposure": float(row["exposure"]),
                "trades": int(row["trades"]),
            }
            if opening_equity <= 0 or not all(
                math.isfinite(record[key]) for key in ("net_return", "fee_rate", "exposure")
            ):
                raise ValueError(f"Invalid daily ledger economics in {path}: {row['decision']}")
            yield row.get("scenario", "original"), row["model"], row["decision"], record


def merge_record(models, model, date, record):
    ledger = models[model]
    previous = ledger.get(date)
    if previous is None:
        ledger[date] = record
        return
    if previous["mark"] != record["mark"] or previous["trades"] != record["trades"] or any(
        not math.isclose(previous[key], record[key], rel_tol=1e-9, abs_tol=1e-11)
        for key in ("net_return", "exposure", "fee_rate")
    ):
        raise ValueError(f"Conflicting decisions for {model} on {date}")


def summarize(model, ledger, dates):
    equity, peak, drawdown, fees = 1.0, 1.0, 0.0, 0.0
    returns, exposures, trades = [], [], 0
    for date in dates:
        row = ledger[date]
        fees += equity * 10_000 * row["fee_rate"]
        equity *= 1 + row["net_return"]
        peak = max(peak, equity)
        drawdown = max(drawdown, 1 - equity / peak)
        returns.append(row["net_return"])
        exposures.append(row["exposure"])
        trades += row["trades"]
    deviation = statistics.stdev(returns) if len(returns) > 1 else 0
    sharpe = math.sqrt(252) * statistics.mean(returns) / deviation if deviation else None
    return dict(model=model, net_return=round(equity - 1, 9), max_drawdown=round(drawdown, 9),
                mean_exposure=round(statistics.mean(exposures), 9), fees=round(fees, 5),
                sessions=len(dates), sharpe=round(sharpe, 7) if sharpe is not None else None,
                trades=trades)


def make_group(group_id, label, horizon, scenario, models):
    # The benchmark calendar defines eligibility, independently of model coverage.
    calendar = models["SPY buy and hold"]
    for model, ledger in models.items():
        for date, record in ledger.items():
            if date not in calendar or record["mark"] != calendar[date]["mark"]:
                raise ValueError(f"Unmatched session for {model} on {date}")
    dates = sorted(calendar)
    years = sorted({date[:4] for date in dates})
    windows = []
    first_years = sorted({min(ledger)[:4] for ledger in models.values()})
    for year in first_years:
        window = [day for day in dates if day[:4] >= year]
        windows.append((f"all-{year}-{years[-1]}", f"{year}–{years[-1]}", "all", window))
    for year in years:
        window = [day for day in dates if day[:4] == year]
        windows.append((year, year, "year", window))
        for quarter in range(1, 5):
            sessions = [day for day in window if (int(day[5:7]) - 1) // 3 + 1 == quarter]
            if sessions:
                windows.append((f"{year}-Q{quarter}", f"{year} Q{quarter}", "quarter", sessions))
    periods = []
    for period_id, title, kind, window in windows:
        required = set(window)
        rows = [summarize(model, ledger, window) for model, ledger in sorted(models.items())
                if ledger.keys() >= required]
        periods.append(dict(id=period_id, label=title, kind=kind, start=window[0],
                            end=window[-1], sessions=len(window), rows=rows))
    return dict(id=group_id, label=label, horizon=horizon, scenario=scenario,
                models=sorted(models), periods=periods)


def export():
    books = defaultdict(lambda: defaultdict(dict))
    sources = [
        ("one", "one-session/portfolio.csv.gz"), ("five", "five-session/portfolio.csv.gz"),
        ("five", "moe-v2/daily.csv.gz"), ("five", "periods-v1/daily.csv.gz"),
        ("five", "robustness/daily.csv.gz"), ("five", "adapter-pools-v1/daily.csv.gz"),
        ("five", "rl-router-v1/daily.csv.gz"), ("five", "joint-v1/daily.csv.gz"),
    ]
    for group, source in sources:
        for scenario, model, date, row in read_ledger(DATA / source):
            target = group if scenario == "original" else scenario
            merge_record(books[target], model, date, row)
    # The stress study archives its price-only benchmarks once, under original.
    for stress in ("language_outage", "reversed_tree"):
        for benchmark in ("Cash", "SPY buy and hold", "Equal weight"):
            books[stress][benchmark] = books["five"][benchmark]
    groups = [make_group(key, label, horizon, scenario, books[key]) for key, label, horizon, scenario in [
        ("five", "Five-session forecasts", 5, "Observed"),
        ("one", "One-session forecasts", 1, "Observed"),
        ("language_outage", "Language outage", 5, "Language outage"),
        ("reversed_tree", "Reversed tree forecasts", 5, "Reversed tree forecasts"),
    ]]
    result = dict(version=1, groups=groups, sources=[source for _, source in sources],
                  basis="Periods use archived decision dates and require every benchmark session. "
                        "Each period begins at $10,000. Net returns compound after costs. "
                        "Drawdown is a positive loss fraction from the highest closing equity, "
                        "including starting capital. Fees scale with period equity. "
                        "Sharpe uses sample daily volatility, 252 sessions and zero risk-free rate. "
                        "The full windows chain archived evaluations across year boundaries.")
    paths = {}
    for group in groups:
        models = books[group["id"]]
        dates = sorted(models["SPY buy and hold"])
        paths[group["id"]] = {
            "dates": dates, "marks": [models["SPY buy and hold"][d]["mark"] for d in dates],
            "returns": {name: [round(ledger[d]["net_return"], 11) if d in ledger else None for d in dates]
                        for name, ledger in models.items()}
        }
    OUT.with_name("period-paths.json").write_text(json.dumps(paths, separators=(",", ":"), allow_nan=False))
    OUT.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    export()
