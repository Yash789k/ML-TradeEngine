"""Stock-specific OpenAI probabilities from masked monthly evidence packets."""

import argparse
import getpass
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests

from .company_text import align
from .decisions import LABELS, MODEL
from .download import SYMBOLS, save


def collect(folder, key):
    text = pd.read_csv(folder / "company-scores.csv")
    fed = pd.DataFrame(json.loads((folder / "statements.json").read_text()))
    fed.index = pd.to_datetime(fed.date)
    market = {}
    for symbol in SYMBOLS:
        obj = json.loads((folder / f"{symbol}.json").read_text())["chart"]["result"][0]
        close = pd.Series(
            obj["indicators"]["adjclose"][0]["adjclose"],
            index=pd.to_datetime(obj["timestamp"], unit="s").normalize(),
        )
        frame = pd.DataFrame(
            {
                "five_session_return": close.pct_change(5),
                "monthly_return": close.pct_change(21),
                "daily_volatility": close.pct_change().rolling(21).std(),
            }
        )
        align(frame, symbol, text)
        market[symbol] = frame
    dates = market["SPY"].dropna().index
    first = pd.Series(dates, index=dates).groupby(dates.to_period("M")).first().tolist()
    first = [date for date in first if (fed.index < date).any()]
    if len(first) > 110 or len(SYMBOLS) > 22:
        raise ValueError("Packet count exceeds the registered budget")
    receipts = folder / "monthly-decisions"
    receipts.mkdir(exist_ok=True)

    def request(date):
        path = receipts / f"{date:%Y-%m-%d}.json"
        if path.exists():
            return json.loads(path.read_text())
        # The service sees anonymous asset IDs, today's trailing numbers and
        # the most recent earlier statement. Future returns and dates are absent.
        evidence = {
            f"asset_{i}": market[s].loc[date].round(5).to_dict()
            for i, s in enumerate(SYMBOLS)
        }
        body = {
            "model": MODEL,
            "input": json.dumps(
                {
                    "prior_statement": fed.loc[fed.index < date].iloc[-1].text,
                    "assets": evidence,
                }
            ),
            "questions": [
                {
                    "name": name,
                    "type": "choice",
                    "instructions": f"Classify {name}'s next five-session return. FLAT means within one half of its daily volatility times sqrt(5).",
                    "choices": [{"value": label} for label in LABELS],
                }
                for name in evidence
            ],
        }
        session = requests.Session()
        session.mount(
            "https://",
            HTTPAdapter(
                max_retries=Retry(
                    total=3,
                    backoff_factor=1,
                    status_forcelist=(429, 500, 502, 503, 504),
                    allowed_methods={"POST"},
                )
            ),
        )
        response = session.post(
            "https://api.openai.com/v1/decisions",
            headers={"Authorization": "Bearer " + key},
            json=body,
            timeout=90,
        )
        if not response.ok:
            raise RuntimeError(
                f"Decisions HTTP {response.status_code}: {response.json().get('error', {}).get('message', '')[:300]}"
            )
        result = response.json()
        answers = {
            a["name"]: {p["value"]: p["probability"] for p in a["probabilities"]}
            for a in result["answers"]
        }
        if set(evidence) != set(answers) or any(
            set(p) != set(LABELS)
            or min(p.values()) < 0
            or not np.isclose(sum(p.values()), 1)
            for p in answers.values()
        ):
            raise ValueError("Invalid monthly probability distribution")
        record = dict(
            date=date.strftime("%Y-%m-%d"),
            model=result["model"],
            usage=result["usage"],
            forecasts={s: answers[f"asset_{i}"] for i, s in enumerate(SYMBOLS)},
        )
        save(path, json.dumps(record))
        return record

    request(first[0])
    with ThreadPoolExecutor(max_workers=3) as pool:
        outputs = list(pool.map(request, first))
    save(folder / "monthly-decisions.json", json.dumps(outputs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    collect(
        parser.parse_args().folder,
        os.environ.get("OPENAI_API_KEY") or getpass.getpass("OpenAI API key: "),
    )
