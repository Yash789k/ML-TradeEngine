"""Data-quality checks on the fresh download used by the simulated replay.

The replay and the later mark-to-market rest on one public download. Two things
are recorded here so a reader can judge it:

* the largest one-day moves in the window (a bad print would show up here);
* a second-source comparison of every daily return against Alpaca's
  consolidated daily bars, when credentials for that read-only data API exist.

The second source is cached under ``data/paper/review-v1/second_source`` so a
rebuild does not need the network. If it cannot be fetched the summary says so;
nothing is assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from review_v1.paths import WORK, read_json, write_json
from review_v1.registration import POST_RELEASE_START, SYMBOLS

CACHE = WORK / "second_source"
URL = "https://data.alpaca.markets/v2/stocks/bars"
TOLERANCE = 0.005  # half a percentage point of daily return


def returns(frames: dict, first: str = POST_RELEASE_START) -> pd.DataFrame:
    """Long table of close-to-close returns from ``first`` onwards."""
    rows = []
    for symbol in SYMBOLS:
        close = frames[symbol].Close
        change = close.pct_change().loc[first:]
        rows.append(pd.DataFrame(dict(symbol=symbol, session=change.index.strftime("%Y-%m-%d"), close=close.loc[change.index].to_numpy(), **{"return": change.to_numpy()})))
    return pd.concat(rows, ignore_index=True)


def largest_moves(table: pd.DataFrame, count: int = 10) -> list:
    order = table["return"].abs().sort_values(ascending=False).index[:count]
    return table.loc[order].to_dict("records")


def fetch_second_source(first: str, last: str, name: str) -> pd.DataFrame:
    """Alpaca consolidated daily closes (adjusted). Read-only market-data request."""
    path = CACHE / f"{name}.json"
    if path.exists():
        return pd.DataFrame(read_json(path)["rows"])
    import requests

    from review_v1.sentiment_link import credentials

    key, secret = credentials()
    if not key or not secret:
        raise RuntimeError("no data-API credentials in .env")
    start = (pd.Timestamp(first) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    rows, token = [], None
    while True:
        params = dict(symbols=",".join(SYMBOLS), timeframe="1Day", start=start, end=last, adjustment="all", feed="sip", limit=10000)
        if token:
            params["page_token"] = token
        response = requests.get(URL, headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}, params=params, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"second source returned HTTP {response.status_code}")
        body = response.json()
        for symbol, bars in body.get("bars", {}).items():
            rows += [dict(symbol=symbol, session=bar["t"][:10], close=float(bar["c"])) for bar in bars]
        token = body.get("next_page_token")
        if not token:
            break
    write_json(path, dict(source="Alpaca market data v2, daily bars, consolidated feed, adjustment=all", fetched_at=pd.Timestamp.now(tz="UTC").isoformat(), rows=rows))
    return pd.DataFrame(rows)


def compare(table: pd.DataFrame, other: pd.DataFrame) -> dict:
    other = other.sort_values(["symbol", "session"]).copy()
    other["other_return"] = other.groupby("symbol").close.pct_change()
    merged = table.merge(other[["symbol", "session", "other_return"]], on=["symbol", "session"], how="left")
    matched = merged.dropna(subset=["other_return"])
    difference = (matched["return"] - matched.other_return).abs()
    worst = matched.loc[difference.idxmax()]
    return dict(
        available=True,
        source="Alpaca market data v2, consolidated daily bars, adjusted",
        rows_compared=int(len(matched)), rows_expected=int(len(table)), symbols=int(matched.symbol.nunique()),
        max_abs_return_difference=float(difference.max()),
        median_abs_return_difference=float(difference.median()),
        rows_over_tolerance=int((difference > TOLERANCE).sum()), tolerance=TOLERANCE,
        worst=dict(symbol=worst.symbol, session=worst.session, primary=float(worst["return"]), second=float(worst.other_return)),
        passed=bool(len(matched) == len(table) and (difference <= TOLERANCE).all()),
        returns={(r.symbol, r.session): float(r.other_return) for r in matched.itertuples()},
    )


def report(frames: dict, name: str, first: str = POST_RELEASE_START) -> dict:
    table = returns(frames, first)
    moves = largest_moves(table)
    try:
        result = compare(table, fetch_second_source(first, table.session.max(), name))
        lookup = result.pop("returns")
        for move in moves:
            move["second_source_return"] = lookup.get((move["symbol"], move["session"]))
    except Exception as exc:  # no credentials or no network: say so
        result = dict(available=False, reason=f"{type(exc).__name__}: {exc}")
    return dict(
        first=first, last=table.session.max(), rows=int(len(table)),
        largest_moves=moves, second_source=result,
        note=(
            "Returns are close to close on adjusted prices. The primary source is the public download used everywhere "
            "in this review; the second source is used only for this comparison."
        ),
        share_of_days_over_5pct=float(np.mean(table["return"].abs() > 0.05)),
    )
