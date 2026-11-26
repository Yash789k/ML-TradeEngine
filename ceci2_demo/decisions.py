"""Score a fixed corpus using the OpenAI Decisions choice endpoint."""

import argparse
import getpass
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .download import save

LABELS = ("DOWN", "FLAT", "UP")
MODEL = "gpt-6-luna"
QUESTION = {
    "type": "choice",
    "name": "direction",
    "instructions": "Estimate the broad US equity market direction over the next five sessions conditional on this statement alone. Small moves are FLAT.",
    "choices": [{"value": label} for label in LABELS],
}


def score_all(folder, key):
    statements = json.loads((folder / "statements.json").read_text())
    if len(statements) > 100 or any(
        len(s["text"].encode()) > 16000 for s in statements
    ):
        raise ValueError("Corpus exceeds the fixed request budget")
    receipts = folder / "decisions"
    receipts.mkdir(exist_ok=True)

    def request(statement):
        path = receipts / (statement["date"] + ".json")
        if path.exists():
            return json.loads(path.read_text())
        session = requests.Session()
        session.mount(
            "https://",
            HTTPAdapter(
                max_retries=Retry(
                    total=4,
                    backoff_factor=0.6,
                    status_forcelist=(429, 500, 502, 503, 504),
                    allowed_methods={"POST"},
                )
            ),
        )
        response = session.post(
            "https://api.openai.com/v1/decisions",
            headers={"Authorization": "Bearer " + key},
            json={"model": MODEL, "input": statement["text"], "questions": [QUESTION]},
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Decisions HTTP {response.status_code}: {response.json().get('error', {}).get('code')}"
            )
        body = response.json()
        answer = body["answers"][0]
        probs = {p["value"]: p["probability"] for p in answer["probabilities"]}
        if (
            set(probs) != set(LABELS)
            or min(probs.values()) < 0
            or not np.isclose(sum(probs.values()), 1)
        ):
            raise ValueError("Invalid Decisions distribution")
        result = dict(
            url=statement["url"],
            probabilities=probs,
            model=body["model"],
            usage=body["usage"],
        )
        save(path, json.dumps(result))
        return result

    # Check access and response schema once before dispatching a batch.
    request(statements[0])
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(request, statements))
    save(folder / "decisions.json", json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    score_all(
        args.folder,
        os.environ.get("OPENAI_API_KEY") or getpass.getpass("OpenAI API key: "),
    )
