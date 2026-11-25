"""Download a fresh isolated snapshot; never overwrite the research caches."""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.snapshot import download_snapshot

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--tickers", nargs="+")
    args = p.parse_args()
    symbols = args.tickers or sorted(
        f.stem[:-6]
        for f in (ROOT / "data/parquet").glob("*_daily.parquet")
        if f.stem != "macro_daily"
    )
    download_snapshot(args.output, symbols)
