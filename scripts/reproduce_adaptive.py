"""Reproduce the dated experiment with archived source and verified snapshots.

Freshness is intentionally waived for an explicitly archived reproduction;
checksum, session, chronology and data-completeness checks remain in force.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.live.adaptive import verify_release


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    release, output = args.release.resolve(), args.output.resolve()
    verify_release(release)
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"
    shutil.copytree(release / "source", workspace)
    replay = output / "release"
    replay.mkdir()
    for name in ["protocol.md", "registration.json"]:
        shutil.copy2(release / name, replay / name)
    shutil.copytree(release / "market", replay / "market")
    script = workspace / "scripts/train_adaptive_release.py"
    code = """import sys, runpy
sys.path.insert(0, sys.argv[1])
import src.data.snapshot as snapshot
original = snapshot.load_snapshot
snapshot.load_snapshot = lambda destination, require_fresh=False: original(destination, require_fresh=False)
script, release = sys.argv[2:4]
sys.argv = [script, '--release', release]
runpy.run_path(script, run_name='__main__')
"""
    subprocess.run(
        [sys.executable, "-c", code, str(workspace), str(script), str(replay)],
        cwd=workspace,
        check=True,
    )
    import pandas as pd

    checked = []
    for f in sorted((release / "experiment").glob("*.parquet")):
        pd.testing.assert_frame_equal(
            pd.read_parquet(f),
            pd.read_parquet(replay / "experiment" / f.name),
            check_exact=True,
        )
        checked.append(f.name)
    for f in sorted((release / "experiment").glob("*.csv")):
        pd.testing.assert_frame_equal(
            pd.read_csv(f),
            pd.read_csv(replay / "experiment" / f.name),
            check_exact=True,
        )
        checked.append(f.name)
    for f in sorted((release / "experiment").glob("*.json")):
        if json.loads(f.read_text()) != json.loads(
            (replay / "experiment" / f.name).read_text()
        ):
            raise AssertionError(f"Reproduction differs: {f.name}")
        checked.append(f.name)
    report = dict(
        status="passed",
        archived_source=True,
        archived_data=True,
        exact_numeric_match=True,
        checked_files=checked,
        note="Source and snapshot hashes verified; historical freshness waived only for this replay.",
    )
    (output / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
