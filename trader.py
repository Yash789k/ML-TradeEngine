"""Public-data → rolling model → local paper ledger. No external orders."""

import argparse
import json
from pathlib import Path
import sys
import fcntl

from src.live.adaptive import run_daily, verify_release, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["daily", "verify", "status"])
    p.add_argument(
        "--release", type=Path, default=Path("data/releases/2026-09-28-adaptive")
    )
    p.add_argument("--workspace", type=Path, default=Path("data/paper/adaptive-v1"))
    p.add_argument(
        "--snapshot",
        type=Path,
        help="Use a verified local snapshot; freshness is still required",
    )
    args = p.parse_args()
    try:
        if args.command in ["verify", "status"]:
            print(json.dumps(verify_release(args.release), indent=2))
        else:
            args.workspace.mkdir(parents=True, exist_ok=True)
            with (args.workspace / ".daily.lock").open("a") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = run_daily(args.release, args.workspace, args.snapshot)
            (args.workspace / "last_error.json").unlink(missing_ok=True)
            print(
                json.dumps(
                    {k: result[k] for k in ["session", "mode", "deployment"]}, indent=2
                )
            )
            print(f"Full decision record: {args.workspace.resolve() / 'latest.json'}")
    except Exception as exc:
        error = dict(
            status="failed", command=args.command, error=f"{type(exc).__name__}: {exc}"
        )
        save_json(args.workspace / "last_error.json", error)
        print(json.dumps(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
