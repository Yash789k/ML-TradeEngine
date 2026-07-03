"""
Mirror the local SQLite signal log (data/live/signal_log.db) to Supabase
Postgres via the PostgREST API.

Why Supabase (and not the GitHub Actions cache):
  - The Actions cache is EVICTED after 7 days without use and capped at 10 GB;
    it is a build accelerator, not a database.
  - Workflow artifacts expire (90 days here).
  - Supabase gives a durable, queryable Postgres table + dashboard for free.

The sync is a full idempotent upsert — tables are small (a few rows per
trading day), so we push everything and let ON CONFLICT dedupe.

Requires (set as GitHub Actions secrets; never commit them):
  SUPABASE_URL          e.g. https://xyzcompany.supabase.co
  SUPABASE_SERVICE_KEY  service-role key (server-side only!)

Exits 0 silently when the env vars are missing, so the workflow step is a
no-op until Supabase is configured. Run scripts/supabase_schema.sql once in
the Supabase SQL editor before the first sync.

Usage:
    python3 scripts/sync_supabase.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import requests

ROOT    = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "live" / "signal_log.db"

# local table → (remote table, conflict target columns)
TABLES = {
    "signals": ("live_signals", "ticker,run_ts"),
    "orders":  ("live_orders",  "ticker,run_ts"),
    "equity":  ("live_equity",  "run_ts"),
}

BATCH = 500


def fetch_rows(conn: sqlite3.Connection, table: str) -> list[dict]:
    conn.row_factory = sqlite3.Row
    cur = conn.execute(f"SELECT * FROM {table}")  # noqa: S608 — fixed table names
    rows = [dict(r) for r in cur.fetchall()]
    for r in rows:
        r.pop("id", None)   # local autoincrement id is not meaningful remotely
    return rows


def upsert(base_url: str, key: str, table: str, conflict: str, rows: list[dict]) -> int:
    """Upsert rows into `table`; returns number of rows sent."""
    if not rows:
        return 0
    url = f"{base_url}/rest/v1/{table}"
    headers = {
        "apikey":        key,
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
        "Prefer":        "resolution=merge-duplicates,return=minimal",
    }
    sent = 0
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        resp = requests.post(
            url, json=chunk, headers=headers,
            params={"on_conflict": conflict}, timeout=30,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"{table}: HTTP {resp.status_code} — {resp.text[:300]}")
        sent += len(chunk)
    return sent


def main() -> None:
    base_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key      = os.environ.get("SUPABASE_SERVICE_KEY", "")

    if not base_url or not key:
        print("[sync_supabase] SUPABASE_URL / SUPABASE_SERVICE_KEY not set — skipping.")
        return

    if not DB_PATH.exists():
        print(f"[sync_supabase] {DB_PATH} not found — nothing to sync.")
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        for local, (remote, conflict) in TABLES.items():
            try:
                rows = fetch_rows(conn, local)
            except sqlite3.OperationalError:
                print(f"[sync_supabase] local table '{local}' missing — skipped.")
                continue
            n = upsert(base_url, key, remote, conflict, rows)
            print(f"[sync_supabase] {local} → {remote}: upserted {n} rows")
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Persistence must never break the trading run — report and fail soft.
        print(f"[sync_supabase] WARNING: sync failed: {exc}", file=sys.stderr)
        sys.exit(0)
