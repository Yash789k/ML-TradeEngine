"""Finance-language features with conservative session alignment and explicit coverage."""

import subprocess

import numpy as np
import pandas as pd
import torch

from .download import SYMBOLS

NEWS_URL = (
    "https://huggingface.co/datasets/benstaf/FNSPID-nasdaq-100-post2019-1newsperrow/"
    "resolve/fd457e063a9476cd2f8d5469d5e519f8ba6334ce/data/train-00000-of-00001.parquet"
)
FEATURES = ["news_tone", "news_available", "filing_tone", "filing_available"]


def score(model, tokenizer, texts):
    results = []
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    with torch.inference_mode():
        for start in range(0, len(texts), 8):
            inputs = tokenizer(
                texts[start : start + 8],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            results.append(model(**inputs).logits.softmax(-1).cpu().numpy())
    return np.concatenate(results)


def headlines(folder):
    news_path = folder / "news.parquet"
    if not news_path.exists():
        tmp = news_path.with_suffix(".part")
        subprocess.run(
            ["curl", "-fsSL", "--retry", "3", NEWS_URL, "-o", str(tmp)], check=True
        )
        tmp.replace(news_path)
    items = pd.read_parquet(
        news_path, columns=["Date", "Stock_symbol", "Article_title", "Url"]
    )
    items = items.rename(
        columns=dict(
            Date="published", Stock_symbol="symbol", Article_title="text", Url="url"
        )
    )
    items = items.dropna().drop_duplicates(["symbol", "url"])
    items = items[items.symbol.isin(SYMBOLS)].copy()
    items["published"] = pd.to_datetime(items.published, utc=True, format="mixed")
    # Bound inference cost at one observed headline per ticker per ISO week.
    # First by timestamp, then URL for deterministic ties; no lookahead selection.
    items = items.sort_values(["published", "url"])
    week = items.published.dt.strftime("%G-%V")
    news = items.groupby(["symbol", week]).head(1).assign(kind="news")
    return news


def score_events(folder, kind, model, tokenizer):
    cache = folder / f"{kind}-scores.csv"
    if cache.exists():
        return pd.read_csv(cache)
    if kind == "news":
        documents = headlines(folder).copy()
    else:
        from .filings import collect

        documents = (
            pd.DataFrame(collect(folder))
            .rename(columns={"accepted": "published"})
            .assign(kind="filing")
        )
    probs = score(model, tokenizer, documents.text.tolist())
    labels = {v.lower(): k for k, v in model.config.id2label.items()}
    documents["tone"] = probs[:, labels["positive"]] - probs[:, labels["negative"]]
    output = documents[["symbol", "published", "url", "kind", "tone"]].copy()
    output.to_csv(cache, index=False)
    return output


def events(folder, model, tokenizer):
    combined = pd.concat(
        [score_events(folder, kind, model, tokenizer) for kind in ("news", "filing")]
    )
    combined.to_csv(folder / "company-scores.csv", index=False)
    return combined


def align(frame, symbol, documents):
    for kind, expiry in (("news", 7), ("filing", 370)):
        group = documents[
            (documents.kind == kind) & (documents.symbol == symbol)
        ].copy()
        # Wait until a later US trading date than the UTC publication date.
        group["date"] = (
            pd.to_datetime(group.published, utc=True, format="mixed")
            .dt.tz_localize(None)
            .dt.normalize()
            .astype("datetime64[ns]")
        )
        joined = pd.merge_asof(
            pd.DataFrame({"session": frame.index.as_unit("ns")}),
            group.sort_values("date"),
            left_on="session",
            right_on="date",
            allow_exact_matches=False,
            tolerance=pd.Timedelta(days=expiry),
        )
        frame[f"{kind}_tone"] = joined.tone.fillna(0).values
        frame[f"{kind}_available"] = joined.tone.notna().astype(float).values
