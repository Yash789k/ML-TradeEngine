"""Attach passing engineering evidence without approving an unproven edge."""

import argparse
import json
import importlib.metadata
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.snapshot import digest
from src.live.adaptive import verify_release, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--test-report", type=Path, required=True)
    p.add_argument("--focused-report", type=Path, required=True)
    p.add_argument("--daily-receipt", type=Path, required=True)
    p.add_argument("--reproduction", type=Path, required=True)
    args = p.parse_args()
    out = args.release.resolve()
    verify_release(out)
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest.get("runtime_source"):
        raise ValueError("Runtime release is already sealed; create a new version")
    tests = ET.parse(args.test_report).getroot()
    cases = tests.findall(".//testcase")
    if not cases or tests.findall(".//failure") or tests.findall(".//error"):
        raise ValueError("Passing test evidence required")
    focused = ET.parse(args.focused_report).getroot()
    if not focused.findall(".//testcase") or focused.findall(".//failure") or focused.findall(".//error"):
        raise ValueError("Passing final focused test evidence required")
    receipt = json.loads(args.daily_receipt.read_text())
    if len(receipt["signals"]) != len(manifest["symbols"]):
        raise ValueError("Daily inference did not cover the universe")
    repro = json.loads(args.reproduction.read_text())
    if repro["status"] != "passed" or not repro["exact_numeric_match"]:
        raise ValueError("Exact reproduction required")
    ready = json.loads((out / "readiness.json").read_text())
    ready["software_contract"] = "approved_for_local_simulation"
    ready["combined_readiness"] = "not_approved"
    ready["approved_strategy"] = "cash"
    ready["profitable_trader_claim"] = "not_approved"
    save_json(out / "readiness.json", ready)
    save_json(
        out / "verification.json",
        dict(
            tests_passed=len(cases) - len(tests.findall(".//skipped")),
            tests_skipped=len(tests.findall(".//skipped")),
            provider_integration_tests_unrun=17,
            test_report_sha256=digest(args.test_report),
            final_focused_report_sha256=digest(args.focused_report),
            commissioning_receipt_sha256=digest(args.daily_receipt),
            fresh_public_instruments=len(receipt["signals"]),
            external_broker_orders=0,
            reproduction=repro,
            scope="Local simulated execution; no economic-edge or broker certification",
        ),
    )
    runtime = out / "runtime_source"
    code = {}
    for pattern in [
        "src/**/*.py",
        "scripts/*.py",
        "trader.py",
        "app.py",
        "requirements.txt",
    ]:
        for f in ROOT.glob(pattern):
            rel = f.relative_to(ROOT)
            target = runtime / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            code[str(rel)] = digest(f)
    # Only execution/imported modules and entrypoints constrain daily runtime;
    # paper builders are archived but can change presentation without retraining.
    manifest["runtime_source"] = {
        k: v
        for k, v in code.items()
        if k.startswith("src/") or k in ["trader.py", "app.py", "requirements.txt"]
    }
    manifest["runtime_dependencies"] = {
        p: importlib.metadata.version(p)
        for p in [
            "numpy",
            "pandas",
            "scikit-learn",
            "joblib",
            "scipy",
            "pyarrow",
            "exchange-calendars",
        ]
    }
    manifest["artifacts"] = {
        str(f.relative_to(out)): digest(f)
        for f in out.rglob("*")
        if f.is_file()
        and not f.name.endswith(".log")
        and f != out / "manifest.json"
        and "source" not in f.relative_to(out).parts
        and "runtime_source" not in f.relative_to(out).parts
    }
    save_json(out / "manifest.json", manifest)
    verify_release(out)
    print(json.dumps(ready, indent=2))


if __name__ == "__main__":
    main()
