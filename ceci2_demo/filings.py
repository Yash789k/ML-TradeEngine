"""Collect annual report excerpts, keyed by EDGAR acceptance time."""

import json
import os
import re
import subprocess
import time
from pathlib import Path

from bs4 import BeautifulSoup

from .download import save

CIKS = {
    "AAPL": 320193,
    "AMZN": 1018724,
    "BAC": 70858,
    "GOOGL": 1652044,
    "GS": 886982,
    "JPM": 19617,
    "META": 1326801,
    "MSFT": 789019,
    "NVDA": 1045810,
    "TSLA": 1318605,
}


def sec(url):
    time.sleep(0.25)
    return subprocess.check_output(
        [
            "curl",
            "-fsSL",
            "--retry",
            "2",
            "--compressed",
            "--max-time",
            "300",
            "-A",
            "CECI2 academic research " + os.environ["SEC_CONTACT"],
            url,
        ],
        text=True,
    )


def collect(folder):
    folder = Path(folder) / "filings"
    folder.mkdir(parents=True, exist_ok=True)
    results = []

    def issuer(item):
        symbol, cik = item
        path = folder / f"{symbol}.json"
        if not path.exists():
            root = "https://data.sec.gov/submissions/"
            submissions = json.loads(sec(root + f"CIK{cik:010d}.json?download=1"))[
                "filings"
            ]
            chunks = [submissions["recent"]]
            for old in submissions["files"]:
                if old["filingTo"] >= "2017" and old["filingFrom"] < "2026":
                    chunks.append(json.loads(sec(root + old["name"] + "?download=1")))
            records = []
            for chunk in chunks:
                for i, form in enumerate(chunk["form"]):
                    date = chunk["filingDate"][i]
                    if form != "10-K" or not "2017" <= date < "2026":
                        continue
                    accession = chunk["accessionNumber"][i].replace("-", "")
                    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{chunk['primaryDocument'][i]}"
                    receipt = folder / f"{accession}.json"
                    if receipt.exists():
                        records.append(json.loads(receipt.read_text()))
                        continue
                    dom = BeautifulSoup(sec(url), "lxml")
                    for tag in dom.select("script,style,ix\\:header,head"):
                        tag.decompose()
                    body = " ".join(dom.stripped_strings)
                    # Match substantive narrative after the final section heading.
                    headings = list(
                        re.finditer(
                            r"item\s+7\s*[.\-:]?\s*management[’']?s?\s+discussion",
                            body,
                            re.I,
                        )
                    )
                    if headings:
                        text = body[headings[-1].end() :][:1600]
                    else:
                        text = next(
                            (
                                s
                                for s in dom.stripped_strings
                                if len(s) > 250
                                and re.search(
                                    r"\b(revenue|business|operations)\b", s, re.I
                                )
                            ),
                            "",
                        )
                    if len(text) < 150:
                        raise ValueError(f"No usable annual filing excerpt at {url}")
                    record = dict(
                        symbol=symbol,
                        accepted=chunk["acceptanceDateTime"][i],
                        url=url,
                        text=text[:1600],
                        form=form,
                    )
                    save(receipt, json.dumps(record))
                    records.append(record)
            if not records:
                raise ValueError(f"No annual reports for {symbol}")
            save(path, json.dumps(records))
        return json.loads(path.read_text())

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as pool:
        for group in pool.map(issuer, CIKS.items()):
            results.extend(group)
    return results


if __name__ == "__main__":
    import sys

    collect(sys.argv[1])
