"""Fresh public daily bars for the post-release window and the live demo.

Uses the pilot's own downloader and validator (``runtime_v2.market``): Yahoo
Finance public chart v8, about ten years of daily bars, current-vintage
dividend/split-adjusted OHLC. Snapshots are immutable and live only in the
isolated, gitignored review workspace. A snapshot is reused when one already
exists for the session, so every stage of a build sees the same bytes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from review_v1.paths import WORK, read_json
from review_v1.registration import SYMBOLS

MARKET = WORK / "market"


def latest_completed_session(now=None) -> pd.Timestamp:
    """Most recent XNYS session whose close is at least 30 minutes old."""
    from runtime_v2.calendar import calendar
    from runtime_v2.policy import utc

    now = utc(now)
    cal = calendar()
    sessions = cal.sessions_in_range((now - pd.Timedelta(days=12)).tz_localize(None).normalize(), now.tz_localize(None).normalize())
    done = [s for s in sessions if cal.session_close(s) + pd.Timedelta(minutes=30) <= now]
    if not done:
        raise RuntimeError("No completed session in the last twelve days")
    return done[-1]


def existing(session) -> Path | None:
    session = str(pd.Timestamp(session).date())
    if not MARKET.exists():
        return None
    for path in sorted(MARKET.glob(f"{session}-*"), reverse=True):
        manifest = read_json(path / "manifest.json", {})
        if manifest.get("status") == "complete" and set(manifest.get("data", {})) == set(SYMBOLS):
            return path
    return None


def snapshot(session=None, refresh: bool = False) -> Path:
    """Path to a complete snapshot ending at ``session``; downloads once if needed."""
    from runtime_v2 import market
    from runtime_v2.policy import utc

    session = latest_completed_session() if session is None else pd.Timestamp(session)
    found = None if refresh else existing(session)
    if found is not None:
        return found
    MARKET.mkdir(parents=True, exist_ok=True)
    stamp = utc().strftime("%Y%m%dT%H%M%S%f")
    path = MARKET / f"{session.date()}-{stamp}"
    market.download(path, SYMBOLS, session)
    return path


def load(path: Path):
    """Validated frames and manifest (file hashes re-checked by the pilot loader)."""
    from runtime_v2 import market

    manifest = read_json(Path(path) / "manifest.json")
    return market.load(path, SYMBOLS, manifest["expected_session"], now=manifest["finished_at"])
