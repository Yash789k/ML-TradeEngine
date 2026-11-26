"""Read-only access to the recorded pilot evidence.

Five decision receipts were written by the scheduled collector in October 2026
(US sessions 1, 2, 14, 15 and 22 October). Their SHA-256 hashes were registered
in the outcome-recovery protocol on 27 October; every load here re-checks the
file against that registration, so a changed receipt fails loudly instead of
being replayed. SQLite ledgers are opened read-only. Nothing is written back.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pandas as pd

from review_v1.paths import ATTEMPTS, R7_RELEASE, R7_REPORT, R7_WORKSPACE, ROOT, SEGMENTS, read_json, sha256

OLD_PREFIX = "/Users/a/Downloads/ml-trade-engine/"


def local(path) -> Path:
    """Receipts store absolute paths from the collecting machine; resolve them here."""
    text = str(path)
    return ROOT / text[len(OLD_PREFIX):] if text.startswith(OLD_PREFIX) else Path(text)


def available() -> bool:
    """Recorded receipts are gitignored local evidence; a fresh clone does not have them."""
    return (R7_RELEASE / "protocol.json").exists() and all(
        (segment["workspace"] / "shadow.sqlite").exists() for segment in SEGMENTS.values()
    )


def registered_decisions() -> list:
    """The five decision receipts as registered on 27 October 2026 (with hashes)."""
    protocol = read_json(R7_RELEASE / "protocol.json")
    rows = []
    for item in protocol["decisions"]:
        rows.append(
            dict(
                case_id=f"{item['segment']}-{item['session']}",
                segment=item["segment"],
                session=item["session"],
                receipt=local(item["receipt"]),
                receipt_sha256=item["receipt_sha256"],
                content_sha256=item["content_sha256"],
                entry=item["entry"],
                endpoint=item["endpoint"],
            )
        )
    return rows


def load_decision(case: dict) -> dict:
    """Plan, result and snapshot manifest for one recorded decision, hash-checked."""
    if sha256(case["receipt"]) != case["receipt_sha256"]:
        raise ValueError(f"Recorded receipt changed since registration: {case['receipt']}")
    day = case["receipt"].parent
    result = read_json(case["receipt"])
    if result["content_sha256"] != case["content_sha256"]:
        raise ValueError("Recorded receipt content hash differs from registration")
    plan = read_json(day / "plan.json")
    if result["plan_sha256"] != plan["content_sha256"]:
        raise ValueError("Recorded result does not bind to its plan")
    snapshot = local(result["snapshot"])
    manifest = read_json(snapshot / "manifest.json")
    if sha256(snapshot / "manifest.json") != result["snapshot_sha256"]:
        raise ValueError("Recorded market snapshot manifest changed")
    return dict(plan=plan, result=result, snapshot=snapshot, manifest=manifest, day=day)


def load_snapshot(snapshot: Path, symbols, session):
    """Bars exactly as the collector validated them (file hashes re-checked)."""
    from runtime_v2 import market

    manifest = read_json(Path(snapshot) / "manifest.json")
    return market.load(snapshot, symbols, session, now=manifest["finished_at"])


def ledger_events(segment: str, account: str = "shadow") -> list:
    path = SEGMENTS[segment]["workspace"] / f"{account}.sqlite"
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        return [json.loads(row[0]) for row in db.execute("SELECT payload FROM events ORDER BY id")]


def ledger_state(segment: str, account: str = "shadow") -> dict:
    path = SEGMENTS[segment]["workspace"] / f"{account}.sqlite"
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        return json.loads(db.execute("SELECT payload FROM state WHERE id=1").fetchone()[0])


def segment_status(segment: str) -> dict:
    workspace = SEGMENTS[segment]["workspace"]
    return dict(
        status=read_json(workspace / "status.json"),
        halt=read_json(workspace / "halt.json"),
        config=read_json(SEGMENTS[segment]["release"] / "config.json"),
        days=sorted(p.name for p in (workspace / "days").iterdir() if p.is_dir()),
    )


def events_in(workspace: Path) -> list:
    """Scheduler event receipts (daily, retries, watchdog, pre-open, outcomes)."""
    rows = []
    for path in sorted((Path(workspace) / "events").glob("*.json")):
        value = read_json(path)
        value["file"] = path.name
        rows.append(value)
    return rows


def collector_events(segment: str) -> list:
    return events_in(SEGMENTS[segment]["workspace"])


def attempts() -> list:
    """Every registered collection attempt, including those that recorded nothing.

    Read from each workspace's own status, halt and scheduler events. The cause
    of a missed window is recorded only where the scheduler logged one (a
    rejected download); a window with no run at all has no recorded cause.
    """
    rows = []
    for name, workspace in ATTEMPTS.items():
        if not (workspace / "status.json").exists():
            continue
        status = read_json(workspace / "status.json")
        halt = read_json(workspace / "halt.json")
        config = read_json(local(read_json(workspace / "binding.json")["release"]) / "config.json")
        events = events_in(workspace)
        daily = {e["session"]: e["status"] for e in events if e["kind"] == "daily"}
        rows.append(
            dict(
                attempt=name, trial_id=status["trial_id"], start=config["start"], end=config["end"],
                required_sessions=int(config["sessions"]), valid_sessions=int(status["valid_observed_sessions"]),
                missing_sessions=status.get("missing_sessions", []),
                not_run=sorted(s for s, v in daily.items() if v == "missed_deadline"),
                rejected=sorted(s for s, v in daily.items() if v == "failed"),
                halted=bool(halt), halt_at=(halt or {}).get("at"), halt_reason=(halt or {}).get("reason"),
                daily=daily,
                outcomes={e["session"]: e["status"] for e in events if e["kind"] == "outcomes"},
                host_requirement=next((x for x in status.get("limitations", []) if "host" in x), None),
            )
        )
    return rows


def recorded_outcomes() -> pd.DataFrame:
    """Outcome rows written by the outcome-recovery collector (four decision dates)."""
    rows = read_json(R7_REPORT / "prediction_rows.json", [])
    return pd.DataFrame(rows)


def outcome_report() -> dict:
    return read_json(R7_REPORT / "report.json", {})


def outcome_snapshots() -> list:
    """Later price snapshots recorded by scheduled outcome reads, oldest first."""
    found = []
    for segment in SEGMENTS.values():
        root = segment["workspace"] / "outcome-snapshots"
        if root.exists():
            found += [p for p in root.iterdir() if (p / "manifest.json").exists()]
    root = R7_WORKSPACE / "snapshots"
    if root.exists():
        found += [p for p in root.iterdir() if (p / "manifest.json").exists()]
    rows = []
    for path in found:
        manifest = read_json(path / "manifest.json")
        if manifest.get("status") == "complete":
            rows.append(dict(path=path, session=manifest["expected_session"], retrieved_at=manifest["retrieved_at"]))
    return sorted(rows, key=lambda r: (r["session"], r["retrieved_at"]))
