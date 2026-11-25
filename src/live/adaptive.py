"""Daily local shadow operation for a sealed research release.

Units are normalized total-return proxy units, not broker share quantities.
No broker adapter, email client, or credential loader is imported here.
"""

import hashlib
import json
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data.snapshot import digest, download_snapshot, load_snapshot
from src.data.validation import calendar, session_closes
from src.execution.account import Account, ExecutionSession
from src.models.adaptive import AdaptiveModel, features
from src.research.adaptive_portfolio import constrain_weights, limits


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))
    tmp.replace(path)


def verify_release(release):
    release = Path(release)
    manifest = json.loads((release / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Release training has not completed")
    for package, expected in manifest.get("runtime_dependencies", {}).items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(
                f"Runtime dependency differs from verified release: {package}"
            )
    for relative, expected in manifest["artifacts"].items():
        if digest(release / relative) != expected:
            raise ValueError(f"Release artifact changed: {relative}")
    for relative, expected in manifest["source"].items():
        if digest(release / "source" / relative) != expected:
            raise ValueError(f"Archived source changed: {relative}")
    for relative, expected in manifest.get("runtime_source", {}).items():
        if digest(release / "runtime_source" / relative) != expected:
            raise ValueError(f"Archived runtime changed: {relative}")
        installed = Path(__file__).resolve().parents[2] / relative
        if digest(installed) != expected:
            raise ValueError(
                f"Installed runtime differs from verified release: {relative}"
            )
    ready = json.loads((release / "readiness.json").read_text())
    if ready["approved_strategy"] != "cash":
        raise ValueError(
            "This release only authorizes cash deployment plus a research shadow"
        )
    if sorted(ready["symbols"]) != sorted(manifest["symbols"]):
        raise ValueError("Release universe mismatch")
    return ready


def daily_targets(data, model_paths, candidate, model_destination=None):
    symbols = sorted(data)
    date = data["SPY"].index[-1]
    raw_targets, rows = {}, []
    with threadpool_limits(limits=1):
        for symbol in symbols:
            raw = data[symbol]
            x = features(raw, data["SPY"])
            model = AdaptiveModel.load(model_paths[symbol])
            age = int((raw.index > pd.Timestamp(model.metadata["decision"])).sum())
            if age >= model.config.retrain_every:
                if model_destination is None:
                    raise ValueError(
                        "Scheduled model refresh requires a versioned destination"
                    )
                model = AdaptiveModel(model.config).fit_at(x, raw.Close, date)
                target = Path(model_destination) / symbol
                if target.exists():
                    # Resume an interrupted batch without overwriting a fit.
                    model = AdaptiveModel.load(target)
                    if pd.Timestamp(model.metadata["decision"]) != date:
                        raise ValueError("Model retraining date mismatch")
                else:
                    model.save(target)
                model_paths[symbol] = target
            p = model.predict(x.iloc[[-1]])["hybrid"][0]
            expected = float(p @ model.class_returns)
            ml = float(expected > 0.003)
            trend = float(x.average_200.iloc[-1] > 0 and x.momentum_12_1.iloc[-1] > 0)
            score = {
                "trend": trend,
                "ml": ml,
                "adaptive_hybrid": (1 - 0.5 * model.gate) * trend
                + 0.5 * model.gate * ml,
            }[candidate]
            raw_targets[symbol] = score * 0.8 / len(symbols)
            rows.append(
                dict(
                    symbol=symbol,
                    p_down=float(p[0]),
                    p_flat=float(p[1]),
                    p_up=float(p[2]),
                    expected_five_session_return=expected,
                    ml_gate=bool(model.gate),
                    model_decision=model.metadata["decision"],
                    latest_label_end=model.metadata["latest_label_end"],
                    model_manifest_sha256=digest(
                        Path(model_paths[symbol]) / "manifest.json"
                    ),
                )
            )
    prices = pd.DataFrame({s: data[s].Close for s in symbols})
    w = constrain_weights(pd.Series(raw_targets), prices.pct_change(fill_method=None))
    return w.to_dict(), rows, {s: str(p) for s, p in model_paths.items()}


def normalized_marks(data, previous=None):
    """Chain the adjusted return onto stored marks across vendor rescaling.

    This provides coherent dividend/split-adjusted simulation units. It does
    not pretend those units are executable broker shares or cash dividends.
    """
    if previous is None:
        return {s: float(f.Close.iloc[-1]) for s, f in data.items()}
    last = pd.Timestamp(previous["session"])
    return {
        s: float(previous["marks"][s] * f.Close.iloc[-1] / f.loc[last, "Close"])
        for s, f in data.items()
    }


def check_contiguous(previous, current):
    sessions = calendar().sessions_in_range(
        pd.Timestamp(previous).date(), pd.Timestamp(current).date()
    )
    if len(sessions) != 2:
        raise ValueError(
            "Missed prospective session: refusing to backfill decisions or fills"
        )


def run_daily(release, workspace, snapshot=None, now=None):
    release, workspace = Path(release).resolve(), Path(workspace).resolve()
    ready = verify_release(release)
    workspace.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    cal = calendar()
    completed = [
        s
        for s in cal.sessions_in_range((now - pd.Timedelta(days=10)).date(), now.date())
        if cal.session_close(s) <= now
    ]
    date = completed[-1]
    day = workspace / "days" / str(date.date())
    day.mkdir(parents=True, exist_ok=True)
    plan_path = day / "plan.json"
    if snapshot is None and plan_path.exists():
        if digest(plan_path) != (day / "plan.sha256").read_text().strip():
            raise ValueError("Daily decision plan was modified")
        snapshot = Path(json.loads(plan_path.read_text())["snapshot"])
    if snapshot is None:
        snapshot = day / "market"
        if not snapshot.exists():
            download_snapshot(snapshot, ready["symbols"])
    data, market = load_snapshot(snapshot)
    if set(data) != set(ready["symbols"]) or any(
        f.index[-1].date() != date.date() for f in data.values()
    ):
        raise ValueError("Daily snapshot is stale or universe is incomplete")
    # One snapshot cannot be reused after a new session has completed.
    date = data["SPY"].index[-1]
    release_hash = digest(release / "manifest.json")
    plans = sorted(workspace.glob("days/*/result.json"))
    prior = None
    for path in plans:
        result = json.loads(path.read_text())
        if result["release_sha256"] != release_hash:
            raise ValueError(
                "Workspace is bound to a different release; use a new workspace"
            )
        if (
            result["mode"] == "prospective_shadow"
            and pd.Timestamp(result["session"]) < date
        ):
            prior = result
    if prior:
        check_contiguous(prior["session"], date)
    if plan_path.exists():
        if digest(plan_path) != (day / "plan.sha256").read_text().strip():
            raise ValueError("Daily decision plan was modified")
        plan = json.loads(plan_path.read_text())
        if plan["release_sha256"] != release_hash or plan[
            "data_manifest_sha256"
        ] != digest(Path(snapshot) / "manifest.json"):
            raise ValueError(
                "Same session was retried with a different release or snapshot"
            )
    else:
        model_paths = {s: release / "models" / s for s in ready["symbols"]}
        if prior:
            model_paths = {s: Path(p) for s, p in prior["models"].items()}
        w, signals, models = daily_targets(
            data, model_paths, ready["candidate"], day / "models"
        )
        marks = normalized_marks(data, prior)
        plan = dict(
            session=str(date),
            prepared_at=now.isoformat(),
            information_cutoff=str(session_closes(pd.DatetimeIndex([date]))[0]),
            release_sha256=release_hash,
            data_manifest_sha256=digest(Path(snapshot) / "manifest.json"),
            snapshot=str(Path(snapshot).resolve()),
            marks=marks,
            shadow_weights=w,
            signals=signals,
            models=models,
            rebalance=prior is None or date.weekday() == 4,
            mode=(
                "prospective_shadow"
                if date.date() >= pd.Timestamp(ready["prospective_start"]).date()
                else "commissioning_preview"
            ),
        )
        save_json(plan_path, plan)
        (day / "plan.sha256").write_text(digest(plan_path))
    result = dict(plan)
    if plan["mode"] == "prospective_shadow":
        for name, w in [
            ("deployment", {s: 0.0 for s in ready["symbols"]}),
            ("shadow", plan["shadow_weights"]),
        ]:
            account = Account(
                ready["symbols"],
                ready["capital"],
                limits(),
                workspace / f"{name}.sqlite",
            )
            result[name] = ExecutionSession(account).on_close(
                plan["information_cutoff"],
                plan["marks"],
                w if plan["rebalance"] else None,
            )
    else:
        result["deployment"] = dict(
            equity=ready["capital"],
            cash=ready["capital"],
            positions={s: 0 for s in ready["symbols"]},
        )
    result["completed_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    save_json(day / "result.json", result)
    save_json(workspace / "latest.json", result)
    return result
