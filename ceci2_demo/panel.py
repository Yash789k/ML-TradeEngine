import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .download import SYMBOLS
from .factors import signals
from .company_text import FEATURES, align, events, score

REV = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
NUMERIC = ["r1", "r5", "r21", "vol", "range", "volume_z", "drawdown", "market"]
FED_TEXT = ["positive", "neutral", "negative", "age"]
TEXT = FED_TEXT + FEATURES


def sentiments(folder):
    records = pd.DataFrame(json.loads((folder / "statements.json").read_text()))
    cache = folder / "finbert-scores.csv"
    if cache.exists() and (folder / "company-scores.csv").exists():
        scores = pd.read_csv(cache)
        if (
            scores.url.tolist() == records.url.tolist()
            and scores.text.tolist() == records.text.tolist()
        ):
            return scores, pd.read_csv(folder / "company-scores.csv")
    source = folder / "finbert"
    name = (
        str(source) if (source / "model.safetensors").exists() else "ProsusAI/finbert"
    )
    tokenizer = AutoTokenizer.from_pretrained(name, revision=REV)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, revision=REV
    ).eval()
    probs = score(model, tokenizer, records.text.tolist())
    for k, label in model.config.id2label.items():
        records[label.lower()] = probs[:, k]
    from .language import classify

    for family, keys in [
        ("qwen", ["qdown", "qflat", "qup"]),
        ("granite", ["gdown", "gflat", "gup"]),
    ]:
        score_cache = folder / f"{family}.csv"
        saved = pd.read_csv(score_cache) if score_cache.exists() else None
        if saved is not None and saved.text.tolist() == records.text.tolist():
            records[keys] = saved[keys]
        else:
            records[keys] = classify(folder, records.text.tolist(), family)
            records[["url", "text"] + keys].to_csv(score_cache, index=False)
    records.to_csv(cache, index=False)
    return records, events(folder, model, tokenizer)


def panel(folder, horizon=5, refresh_adjustments=False):
    folder = Path(folder)
    records, documents = sentiments(folder)
    path = folder / "decisions.json"
    if not path.exists():
        path = Path(__file__).with_name("receipts") / "decisions.json"
    decisions = {r["url"]: r["probabilities"] for r in json.loads(path.read_text())}
    for name in ("DOWN", "FLAT", "UP"):
        records["o" + name.lower()] = [decisions[url][name] for url in records.url]
    records.index = pd.DatetimeIndex(pd.to_datetime(records.date.astype(str))).as_unit(
        "ns"
    )
    packet_path = folder / "monthly-decisions.json"
    if not packet_path.exists():
        packet_path = Path(__file__).with_name("receipts") / "monthly-decisions.json"
    packets = json.loads(packet_path.read_text())
    reference = pd.read_csv(
        Path(__file__).with_name("receipts") / "adjustments.csv.gz",
        index_col="date",
        parse_dates=["date"],
        float_precision="round_trip",
    )
    frames = {}
    for symbol in SYMBOLS:
        payload = json.loads((folder / f"{symbol}.json").read_text())["chart"][
            "result"
        ][0]
        dates = (
            pd.to_datetime(payload["timestamp"], unit="s", utc=True)
            .tz_convert("America/New_York")
            .tz_localize(None)
            .normalize()
        )
        frame = pd.DataFrame(payload["indicators"]["quote"][0], index=dates)
        adj = np.asarray(payload["indicators"]["adjclose"][0]["adjclose"]) / frame.close
        if not refresh_adjustments:
            if not dates.equals(reference.index):
                raise ValueError(f"Snapshot date coverage changed for {symbol}")
            adj = reference[symbol]
        frame[["open", "high", "low", "close"]] = frame[
            ["open", "high", "low", "close"]
        ].mul(adj, axis=0)
        if (
            not dates.is_monotonic_increasing
            or not dates.is_unique
            or not np.isfinite(frame).all().all()
            or (frame <= 0).any().any()
            or (frame.high < frame.low).any()
        ):
            raise ValueError(f"Invalid OHLCV for {symbol}")
        frames[symbol] = frame
    market = frames["SPY"].close.pct_change()
    import exchange_calendars as xcals

    expected = xcals.get_calendar("XNYS", start="2017-01-01", end="2025-12-31").sessions
    expected = expected.tz_localize(None)
    if not market.index.equals(expected):
        raise ValueError(
            f"Invalid SPY calendar: {market.index.symmetric_difference(expected)}"
        )
    text = pd.merge_asof(
        pd.DataFrame({"session": market.index.as_unit("ns")}),
        records.rename_axis("publication").reset_index(),
        left_on="session",
        right_on="publication",
        allow_exact_matches=False,
    )
    text["age"] = (text.session - text.publication).dt.days
    factors, _ = signals(folder, market.index)
    for symbol, f in frames.items():
        if not f.index.equals(market.index):
            raise ValueError(
                f"Missing sessions for {symbol}: {f.index.symmetric_difference(market.index)}"
            )
        for lag in (1, 5, 21):
            f[f"r{lag}"] = f.close.pct_change(lag)
        f["vol"] = f.r1.rolling(21).std()
        f["atr"] = (
            pd.concat(
                [
                    f.high - f.low,
                    (f.high - f.close.shift()).abs(),
                    (f.low - f.close.shift()).abs(),
                ],
                axis=1,
            )
            .max(axis=1)
            .rolling(14)
            .mean()
        )
        f["range"] = (f.high - f.low) / f.close
        f["volume_z"] = np.log(f.volume / f.volume.rolling(21).mean())
        f["drawdown"] = f.close / f.close.rolling(63).max() - 1
        f["market"] = market
        f[
            FED_TEXT
            + [
                "qdown",
                "qflat",
                "qup",
                "gdown",
                "gflat",
                "gup",
                "odown",
                "oflat",
                "oup",
            ]
        ] = text[
            FED_TEXT
            + [
                "qdown",
                "qflat",
                "qup",
                "gdown",
                "gflat",
                "gup",
                "odown",
                "oflat",
                "oup",
            ]
        ].to_numpy()
        align(f, symbol, documents)
        f[factors.columns] = factors
        probabilities = pd.DataFrame(
            [r["forecasts"][symbol] for r in packets],
            index=pd.to_datetime([r["date"] for r in packets]),
        )
        f[["mdown", "mflat", "mup"]] = probabilities[["DOWN", "FLAT", "UP"]].reindex(
            f.index, method="ffill"
        )
        threshold = np.maximum(0.001, 0.5 * f.vol * np.sqrt(horizon))
        ret = f.open.shift(-(1 + horizon)) / f.open.shift(-1) - 1
        f["label"] = np.select([ret < -threshold, ret > threshold], [0, 2], default=1)
        f["endpoint"] = f.index.to_series().shift(-(1 + horizon))
        f["entry"] = f.open.shift(-1)
        f["exit"] = f.open.shift(-2)
        f["mark"] = f.index.to_series().shift(-2)
        f["symbol"] = symbol
    return frames, records
