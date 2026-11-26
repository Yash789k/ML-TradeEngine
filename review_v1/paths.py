"""Locations, provenance vocabulary and small JSON helpers shared by the package."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Versioned, committed outputs (JSON/CSV only: *.parquet is gitignored).
OUT = ROOT / "data/review-v1"
# Isolated mutable state for replays and demo runs. Never a pilot workspace.
WORK = ROOT / "data/paper/review-v1"

ADAPTIVE_RELEASE = ROOT / "data/releases/2026-09-28-adaptive"
RESEARCH_V2 = ROOT / "data/research-v2"
CONTROLLER_V1 = ROOT / "data/research-controller-v1"

# The three pilot segments that produced valid decision receipts.
SEGMENTS = {
    "r2": dict(
        release=ROOT / "data/releases/research-v2-runtime-20261001-week1-r2",
        workspace=ROOT / "data/paper/research-week1-r2",
    ),
    "r4": dict(
        release=ROOT / "data/releases/research-v2-runtime-20261014-week1-r4",
        workspace=ROOT / "data/paper/research-week1-r4",
    ),
    "r6": dict(
        release=ROOT / "data/releases/research-v2-runtime-20261022-week1-r6",
        workspace=ROOT / "data/paper/research-week1-r6",
    ),
}
# Every registered collection attempt, in order, including the three that
# recorded no decision. Their halts are evidence about operations.
ATTEMPTS = {
    "week1": ROOT / "data/paper/research-week1",
    "r2": ROOT / "data/paper/research-week1-r2",
    "r3": ROOT / "data/paper/research-week1-r3",
    "r4": ROOT / "data/paper/research-week1-r4",
    "r5": ROOT / "data/paper/research-week1-r5",
    "r6": ROOT / "data/paper/research-week1-r6",
}
R7_RELEASE = ROOT / "data/releases/outcomes-r7-20261027"
R7_WORKSPACE = ROOT / "data/paper/research-outcomes-r7"
R7_REPORT = ROOT / "data/research-outcomes-r7/reports/incomplete-closeout-20261105"

CLASSES = ["DOWN", "FLAT", "UP"]

# Every fact shown in the walkthrough carries exactly one of these labels.
RECORDED = "recorded"            # written by the scheduled collector at the time
RECONSTRUCTED = "reconstructed"  # recomputed now from inputs recorded at the time
SIMULATED = "simulated"          # produced by a replay or demo run; never a pilot observation
PROVENANCE = (RECORDED, RECONSTRUCTED, SIMULATED)


def rel(path) -> str:
    """Repository-relative path for committed artifacts (no machine-specific prefix)."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    """JSON-safe copy: NaN/inf become null, numpy scalars become Python scalars."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (AttributeError, ValueError):
            pass
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return rel(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def write_json(path, value) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default
