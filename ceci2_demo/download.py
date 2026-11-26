"""Download a reproducible historical sample into a user-owned cache."""

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

SYMBOLS = (
    "AAPL",
    "AMZN",
    "BAC",
    "DIA",
    "GLD",
    "GOOGL",
    "GS",
    "IWM",
    "JPM",
    "META",
    "MSFT",
    "NVDA",
    "QQQ",
    "SPY",
    "TLT",
    "TSLA",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLV",
)
FED = "https://www.federalreserve.gov"


def get(url):
    return subprocess.check_output(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--retry",
            "3",
            "--max-time",
            "60",
            "-A",
            "Mozilla/5.0",
            url,
        ],
        text=True,
    )


def save(path, content):
    tmp = path.with_suffix(".part")
    tmp.write_text(content)
    tmp.replace(path)


def fetch(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for symbol in SYMBOLS:
        path = folder / f"{symbol}.json"
        if not path.exists():
            url = (
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                "?period1=1483228800&period2=1767225600&interval=1d"
            )
            payload = json.loads(get(url))
            if payload["chart"]["result"][0]["meta"]["symbol"] != symbol:
                raise ValueError(f"Unexpected instrument for {symbol}")
            save(
                path,
                json.dumps(
                    dict(
                        url=url,
                        retrieved=datetime.now(timezone.utc).isoformat(),
                        chart=payload["chart"],
                    )
                ),
            )
    path = folder / "statements.json"
    if path.exists():
        return
    calendar = get(FED + "/monetarypolicy/fomccalendars.htm")
    for year in range(2017, 2026):
        if f"monetary{year}" not in calendar:
            calendar += get(FED + f"/monetarypolicy/fomchistorical{year}.htm")
    links = sorted(
        set(re.findall(r"/newsevents/pressreleases/monetary(20\d{6})a.htm", calendar))
    )
    dates = [d for d in links if "20170101" <= d < "20260101"]

    def statement(day):
        url = FED + f"/newsevents/pressreleases/monetary{day}a.htm"
        page = get(url)
        from bs4 import BeautifulSoup

        dom = BeautifulSoup(page, "html.parser")
        votes = next(
            (
                p
                for p in dom.find_all("p")
                if p.get_text(strip=True).startswith("Voting for")
            ),
            None,
        )
        if votes is None:
            return None
        release = dom.select_one(".releaseTime")
        if release is None:
            raise ValueError(f"Missing release time at {url}")
        paragraphs = list(reversed(votes.find_previous_siblings("p")))
        text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
        if not text:
            raise ValueError(f"Empty statement at {url}")
        return dict(
            date=day, release=release.get_text(" ", strip=True), text=text, url=url
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        records = [item for item in pool.map(statement, dates) if item]
    for year in range(2017, 2026):
        if sum(r["date"].startswith(str(year)) for r in records) < 6:
            raise ValueError(f"Incomplete FOMC archive in {year}")
    save(path, json.dumps(records, indent=2))


if __name__ == "__main__":
    import sys

    fetch(sys.argv[1])
