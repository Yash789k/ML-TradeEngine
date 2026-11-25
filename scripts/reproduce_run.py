"""Verify a frozen research run, or reproduce it from its archived code and inputs."""

from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys


def verify(run):
    manifest = json.loads((run / "manifest.json").read_text())
    for name, digest in manifest["code"].items():
        if hashlib.sha256((run / "source" / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Archived code hash mismatch: {name}")
    for ticker, record in manifest["data"].items():
        if (
            hashlib.sha256(
                (run / "snapshots" / f"{ticker}_daily.parquet").read_bytes()
            ).hexdigest()
            != record["sha256"]
        ):
            raise ValueError(f"Data hash mismatch: {ticker}")
    for name, digest in json.loads((run / "output-hashes.json").read_text()).items():
        if hashlib.sha256((run / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Output hash mismatch: {name}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = verify(run)
    if args.verify_only:
        print("Verified frozen code, input snapshots and output hashes.")
        return
    if args.output is None:
        parser.error("--output is required for reproduction")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"
    shutil.copytree(run / "source", workspace)
    shutil.copytree(run / "snapshots", workspace / "data/parquet")
    # The historical runner records git HEAD. A standalone copied snapshot gets
    # its own local provenance commit; the original HEAD remains in parent.json.
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Research Reproduction",
            "-c",
            "user.email=reproduction@localhost",
            "commit",
            "-qm",
            "Frozen research source and data snapshot",
        ],
        cwd=workspace,
        check=True,
    )
    (output / "parent.json").write_text(
        json.dumps(
            dict(
                run=str(run),
                git_head=manifest["git_head"],
                manifest_sha256=hashlib.sha256(
                    (run / "manifest.json").read_bytes()
                ).hexdigest(),
            ),
            indent=2,
        )
    )
    budget = manifest["ml_budget"]
    command = [
        sys.executable,
        str(workspace / "scripts/validated_research.py"),
        "--output",
        str(output / "research"),
        "--tickers",
        *manifest["data"].keys(),
        "--trials",
        str(budget["trials"]),
        "--epochs",
        str(budget["epochs"]),
    ]
    if "--ml" in manifest["command"]:
        command.append("--ml")
    subprocess.run(command, cwd=workspace, check=True)


if __name__ == "__main__":
    main()
