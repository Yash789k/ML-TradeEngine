"""Immutable public-data download, with explicit freshness and provenance."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data.loader import _split_adjust
from src.data.sources import fetch_equity_daily
from src.data.validation import observed_equity_bars


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download_snapshot(destination, symbols, period="10y"):
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    (out / "raw").mkdir()
    (out / "adjusted").mkdir()
    manifest = dict(
        status="fetching",
        provider="Yahoo Finance public daily chart",
        retrieved_at=pd.Timestamp.now(tz="UTC").isoformat(),
        period=period,
        price_convention="retrospectively split/dividend-adjusted OHLC proxy",
        data={},
        errors={},
    )

    def fetch(symbol):
        raw = fetch_equity_daily(symbol, period=period)
        raw_path = out / "raw" / f"{symbol}.parquet"
        raw.to_parquet(raw_path)
        # Preserve provider response rows even if freshness validation fails.
        adjusted = observed_equity_bars(_split_adjust(raw.copy()), require_fresh=True)
        path = out / "adjusted" / f"{symbol}.parquet"
        adjusted.to_parquet(path)
        return dict(
            rows=len(adjusted),
            start=str(adjusted.index[0]),
            end=str(adjusted.index[-1]),
            raw_sha256=digest(raw_path),
            sha256=digest(path),
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch, s): s for s in sorted(set(symbols))}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                manifest["data"][symbol] = future.result()
                print(
                    f"Downloaded {symbol}: {manifest['data'][symbol]['end']}",
                    flush=True,
                )
            except Exception as exc:
                manifest["errors"][symbol] = f"{type(exc).__name__}: {exc}"
                print(f"Download failed {symbol}: {exc}", flush=True)
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    manifest["status"] = "failed" if manifest["errors"] else "complete"
    manifest["finished_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if manifest["errors"]:
        raise RuntimeError(f"Snapshot incomplete: {manifest['errors']}")
    return manifest


def load_snapshot(destination, require_fresh=False):
    out = Path(destination)
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["errors"]:
        raise ValueError("Incomplete data snapshot")
    frames = {}
    for symbol, meta in manifest["data"].items():
        path = out / "adjusted" / f"{symbol}.parquet"
        raw = out / "raw" / f"{symbol}.parquet"
        if digest(path) != meta["sha256"] or digest(raw) != meta["raw_sha256"]:
            raise ValueError(f"Data hash mismatch: {symbol}")
        frames[symbol] = observed_equity_bars(
            pd.read_parquet(path), require_fresh=require_fresh
        )
    ends = {frame.index[-1] for frame in frames.values()}
    if len(ends) != 1:
        raise ValueError("Universe does not share a last completed session")
    return frames, manifest
