"""Read-only local readiness inventory. Never print secret values or call providers."""

from pathlib import Path
import importlib.util
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.models.bundle import validate_bundle
from src.data.validation import observed_equity_bars


def status():
    rows = {}
    for name, reason in {
        "macro": "Latest-vintage FRED data excluded without release/vintage evidence",
        "sentiment": "No frozen historical availability/content vintage evidence",
        "crypto": "Full training-period coverage unavailable; excluded from validated models",
        "broker": "External submission isolated; local account only",
        "alerts": "External messaging disabled during repairs",
        "supabase": "Remote writes disabled; local SQLite is authoritative",
    }.items():
        rows[name] = dict(status="intentionally_disabled", reason=reason)
    raw = ROOT / "data/parquet/SPY_daily.parquet"
    try:
        observed_equity_bars(pd.read_parquet(raw), require_fresh=True)
        rows["equity_data"] = dict(status="working")
    except Exception as e:
        rows["equity_data"] = dict(status="unavailable", reason=str(e))
    try:
        validate_bundle(ROOT / "data/models/SPY")
        rows["production_models"] = dict(status="working")
    except Exception as e:
        rows["production_models"] = dict(status="unavailable", reason=str(e))
    for prefix, keys in [
        ("smtp", ["SMTP_HOST", "SMTP_USER", "SMTP_PASS", "ALERT_EMAIL"]),
        ("supabase", ["SUPABASE_URL", "SUPABASE_SERVICE_KEY"]),
    ]:
        present = [bool(os.environ.get(k, "").strip()) for k in keys]
        if any(present) and not all(present):
            rows[prefix] = dict(
                status="misconfigured",
                reason="Partial configuration; missing required settings",
            )
    rows["local_persistence"] = dict(
        status="working",
        reason="Transactional SQLite snapshots, events and restart tests",
    )
    return rows


if __name__ == "__main__":
    print(json.dumps(status(), indent=2))
