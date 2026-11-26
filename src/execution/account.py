"""Cash-constrained long-only account and next-close target execution.

Supported: a fixed, explicitly listed US-equity/ETF universe, fractional shares,
no leverage, no shorts. Stops/targets are absent from this contract. Policy may
reduce target weights; its effects then pass through the same checks as all orders.
An order reserves cash against a configurable price buffer. Gap fills outside its
reservation or projected limits are rejected, never financed with implicit credit.
"""

from dataclasses import dataclass, asdict, field
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Limits:
    max_gross: float = 0.8
    max_position: float = 0.2
    max_turnover: float = 0.8
    drawdown_limit: float = 0.2
    commission: float = 0.001
    slippage: float = 0.0005
    reserve_buffer: float = 0.03
    fractional: bool = True
    # Correlated holdings can share an explicit exposure bucket. Limits apply
    # to the whole bucket, not independently to its constituent instruments.
    groups: dict = field(default_factory=dict)
    group_limits: dict = field(default_factory=dict)

    def __post_init__(self):
        vals = [
            self.max_gross,
            self.max_position,
            self.max_turnover,
            self.drawdown_limit,
            self.commission,
            self.slippage,
            self.reserve_buffer,
            *self.group_limits.values(),
        ]
        if not all(math.isfinite(v) and v >= 0 for v in vals):
            raise ValueError("Limits must be finite and nonnegative")
        if (
            not 0 < self.max_position <= self.max_gross <= 1
            or not 0 < self.drawdown_limit < 1
        ):
            raise ValueError("Only unlevered long-only limits are supported")


class Account:
    def __init__(self, symbols, initial_capital=10000.0, limits=None, state_path=None):
        self.symbols = sorted(set(symbols))
        self.initial_capital = float(initial_capital)
        self.limits = limits or Limits()
        self.state_path = Path(state_path) if state_path else None
        if (
            not self.symbols
            or initial_capital <= 0
            or not math.isfinite(initial_capital)
        ):
            raise ValueError("Explicit universe and positive capital required")
        self.cash = float(initial_capital)
        self.positions = {s: 0.0 for s in self.symbols}
        self.basis = {s: 0.0 for s in self.symbols}
        self.realized = 0.0
        self.fees = 0.0
        self.orders = {}
        self.decisions = {}
        self.fill_ids = {}
        self.events = []
        self.peak = float(initial_capital)
        self.last_event = None
        self.revision = 0
        self._pending_events = []
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as db:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, revision INTEGER, payload TEXT)"
                )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, payload TEXT)"
                )
                row = db.execute(
                    "SELECT revision,payload FROM state WHERE id=1"
                ).fetchone()
                if row:
                    self._restore(json.loads(row[1]))
                    self.revision = row[0]
                    self.events = [
                        json.loads(r[0])
                        for r in db.execute("SELECT payload FROM events ORDER BY id")
                    ]
                else:
                    self._save(db)

    @property
    def cost(self):
        return self.limits.commission + self.limits.slippage

    def _connect(self):
        return sqlite3.connect(self.state_path, timeout=20)

    def _payload(self):
        return dict(
            version=1,
            symbols=self.symbols,
            initial_capital=self.initial_capital,
            limits=asdict(self.limits),
            **{
                k: getattr(self, k)
                for k in [
                    "cash",
                    "positions",
                    "basis",
                    "realized",
                    "fees",
                    "orders",
                    "decisions",
                    "fill_ids",
                    "peak",
                    "last_event",
                ]
            },
        )

    def _restore(self, state):
        if (
            state.get("version") != 1
            or state["symbols"] != self.symbols
            or state["limits"] != asdict(self.limits)
            or state["initial_capital"] != self.initial_capital
        ):
            raise ValueError(
                "Account configuration mismatch; do not silently reset persistent state"
            )
        for k in [
            "cash",
            "positions",
            "basis",
            "realized",
            "fees",
            "orders",
            "decisions",
            "fill_ids",
            "peak",
            "last_event",
        ]:
            setattr(self, k, state[k])
        if self.cash < -1e-7 or any(q < 0 for q in self.positions.values()):
            raise ValueError("Corrupt persisted account")

    def _save(self, db=None):
        if not self.state_path or getattr(self, "_defer_save", False):
            return
        if db is None:
            with self._connect() as conn:
                self._save(conn)
            return
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT revision FROM state WHERE id=1").fetchone()
        if row and row[0] != self.revision:
            raise RuntimeError(
                "Concurrent account update detected; reload before retrying"
            )
        self.revision += 1
        db.execute(
            "INSERT OR REPLACE INTO state VALUES (1,?,?)",
            (self.revision, json.dumps(self._payload(), allow_nan=False)),
        )
        db.executemany(
            "INSERT INTO events(payload) VALUES (?)",
            [(json.dumps(e, allow_nan=False),) for e in self._pending_events],
        )
        self._pending_events.clear()

    def event(self, kind, **payload):
        e = dict(kind=kind, **payload)
        self.events.append(e)
        self._pending_events.append(e)

    def prices(self, prices):
        if set(prices) != set(self.symbols) or any(
            not math.isfinite(float(v)) or v <= 0 for v in prices.values()
        ):
            raise ValueError(
                "Complete finite positive marks required for the configured universe"
            )

    def equity(self, prices):
        self.prices(prices)
        return self.cash + sum(self.positions[s] * prices[s] for s in self.symbols)

    def pending(self):
        return [
            o for o in self.orders.values() if o["status"] in ("pending", "partial")
        ]

    def reserved_cash(self):
        return sum(
            o["remaining"] * o["limit_price"] * (1 + self.cost)
            for o in self.pending()
            if o["side"] == "buy"
        )

    def projected(self, prices):
        # Count all unfilled buys, without spending proceeds of pending sales.
        q = dict(self.positions)
        for o in self.pending():
            if o["side"] == "buy":
                q[o["symbol"]] += o["remaining"]
        return {s: q[s] * prices[s] for s in self.symbols}

    def _risk_ok(self, values, equity):
        if equity <= 0:
            return False
        if sum(values.values()) > self.limits.max_gross * equity + 1e-6:
            return False
        if any(v > self.limits.max_position * equity + 1e-6 for v in values.values()):
            return False
        for group, limit in self.limits.group_limits.items():
            if (
                sum(v for s, v in values.items() if self.limits.groups.get(s) == group)
                > limit * equity + 1e-6
            ):
                return False
        return True

    def snapshot(self, prices):
        eq = self.equity(prices)
        unrealized = sum(
            self.positions[s] * prices[s] - self.basis[s] for s in self.symbols
        )
        error = eq - (self.initial_capital + self.realized + unrealized)
        if (
            abs(error) > 1e-6
            or self.cash < -1e-6
            or self.cash - self.reserved_cash() < -1e-6
        ):
            raise RuntimeError(f"Ledger reconciliation failed: {error}")
        return dict(
            equity=eq,
            cash=self.cash,
            reserved_cash=self.reserved_cash(),
            positions=dict(self.positions),
            realized_pnl=self.realized,
            unrealized_pnl=unrealized,
            fees=self.fees,
            gross_exposure=sum(self.positions[s] * prices[s] for s in self.symbols)
            / eq,
            drawdown=eq / self.peak - 1,
            reconciliation_error=error,
        )

    def cancel(self, order_id, reason="superseded"):
        o = self.orders[order_id]
        if o["status"] in ("pending", "partial"):
            o["status"] = "cancelled"
            self.event("cancel", order_id=order_id, reason=reason)
            self._save()

    def rebalance(self, decision_id, asof, weights, prices):
        self.prices(prices)
        asof = pd.Timestamp(asof).isoformat()
        if set(weights) != set(self.symbols) or any(
            not math.isfinite(v) or v < 0 for v in weights.values()
        ):
            raise ValueError(
                "Full long-only target vector required; FLAT means target zero"
            )
        signature = hashlib.sha256(
            json.dumps([asof, weights, prices], sort_keys=True).encode()
        ).hexdigest()
        if decision_id in self.decisions:
            if self.decisions[decision_id] != signature:
                raise ValueError("Decision id reused with different inputs")
            return []
        if self.last_event and pd.Timestamp(asof) < pd.Timestamp(self.last_event):
            raise ValueError("Out-of-order decision")
        eq = self.equity(prices)
        self.peak = max(self.peak, eq)
        if not self._risk_ok(weights, 1.0):
            raise ValueError("Target weights breach exposure limits")
        if eq / self.peak - 1 <= -self.limits.drawdown_limit:
            weights = {s: 0.0 for s in self.symbols}
            self.event("drawdown_halt", asof=asof)
        marked = {s: self.positions[s] * prices[s] for s in self.symbols}
        if not self._risk_ok(marked, eq):
            self.event(
                "exposure_breach",
                asof=asof,
                reason="Market drift; increases blocked, reductions permitted",
                values=marked,
            )
        for o in self.pending():
            self.cancel(o["id"])
        deltas = {
            s: weights[s] * eq / prices[s] - self.positions[s] for s in self.symbols
        }
        turnover = sum(abs(q) * prices[s] for s, q in deltas.items())
        scale = min(1.0, self.limits.max_turnover * eq / turnover) if turnover else 1.0
        ids = []
        for s, delta in sorted(deltas.items(), key=lambda item: item[1]):
            delta *= scale
            if abs(delta) * prices[s] < 0.01:
                continue
            side = "buy" if delta > 0 else "sell"
            qty = abs(delta)
            cap = prices[s] * (1 + self.limits.reserve_buffer)
            if side == "buy":
                qty = min(
                    qty,
                    max(0, self.cash - self.reserved_cash()) / (cap * (1 + self.cost)),
                )
                projected = self.projected(prices)
                # Allow the same equity haircut as entry friction.
                max_by_asset = max(0, self.limits.max_position * eq - projected[s]) / (
                    prices[s] * (1 + self.limits.max_position * self.cost)
                )
                max_by_gross = max(
                    0, self.limits.max_gross * eq - sum(projected.values())
                ) / (prices[s] * (1 + self.limits.max_gross * self.cost))
                qty = min(qty, max_by_asset, max_by_gross)
            else:
                qty = min(qty, self.positions[s])
            unit = 1e-6 if self.limits.fractional else 1.0
            qty = math.floor(qty / unit) * unit
            if qty * prices[s] < 0.01:
                continue
            projected = self.projected(prices)
            if side == "buy":
                projected[s] += qty * prices[s]
                if not self._risk_ok(projected, eq - qty * prices[s] * self.cost):
                    self.event(
                        "risk_rejection",
                        asof=asof,
                        symbol=s,
                        reason="projected exposure including pending orders",
                    )
                    continue
            oid = f"{decision_id}:{s}"
            self.orders[oid] = dict(
                id=oid,
                symbol=s,
                side=side,
                qty=qty,
                remaining=qty,
                limit_price=cap,
                decision_at=asof,
                status="pending",
            )
            self.event("order", **self.orders[oid])
            ids.append(oid)
        self.decisions[decision_id] = signature
        self.last_event = asof
        self.event(
            "decision",
            decision_id=decision_id,
            information_cutoff=asof,
            weights=weights,
        )
        self.snapshot(prices)
        self._save()
        return ids

    def fill(self, order_id, qty, price, asof, prices, fill_id):
        if fill_id in self.fill_ids:
            if self.fill_ids[fill_id] != [order_id, qty, price, str(asof)]:
                raise ValueError("Fill id reused with changed content")
            return
        self.prices(prices)
        o = self.orders[order_id]
        s = o["symbol"]
        if (
            o["status"] not in ("pending", "partial")
            or not math.isfinite(qty)
            or qty <= 0
            or qty > o["remaining"] + 1e-8
            or not math.isfinite(price)
            or price <= 0
        ):
            raise ValueError("Invalid fill or inactive order")
        if pd.Timestamp(asof) <= pd.Timestamp(o["decision_at"]):
            raise ValueError("Fill must follow information cutoff")
        if self.last_event and pd.Timestamp(asof) < pd.Timestamp(self.last_event):
            raise ValueError("Out-of-order fill")
        fee = qty * price * self.cost
        if o["side"] == "buy":
            remaining_reservation = self.reserved_cash() - qty * o["limit_price"] * (
                1 + self.cost
            )
            vals = self.projected(prices)
            if (
                price > o["limit_price"]
                or qty * price + fee + remaining_reservation > self.cash + 1e-7
                or not self._risk_ok(vals, self.equity(prices) - fee)
            ):
                o["status"] = "rejected"
                self.event(
                    "rejection",
                    order_id=order_id,
                    asof=str(asof),
                    reason="gap/cash/projected exposure",
                )
                self._save()
                return
            self.cash -= qty * price + fee
            self.positions[s] += qty
            self.basis[s] += qty * price + fee
            pnl = None
        else:
            if qty > self.positions[s] + 1e-8:
                raise ValueError("Shorts are ineligible")
            cost_basis = self.basis[s] * qty / self.positions[s]
            pnl = qty * price - fee - cost_basis
            self.realized += pnl
            self.basis[s] -= cost_basis
            self.cash += qty * price - fee
            self.positions[s] -= qty
        self.fees += fee
        o["remaining"] = max(0, o["remaining"] - qty)
        o["status"] = "filled" if o["remaining"] < 1e-8 else "partial"
        self.fill_ids[fill_id] = [order_id, qty, price, str(asof)]
        self.event(
            "fill",
            order_id=order_id,
            fill_id=fill_id,
            asof=str(asof),
            symbol=s,
            side=o["side"],
            qty=qty,
            price=price,
            fee=fee,
            pnl=pnl,
        )
        self.snapshot(prices)
        self._save()


class ExecutionSession:
    """Identical event handler used for historical replay and local paper runs."""

    def __init__(self, account):
        self.account = account

    def on_close(self, asof, prices, weights):
        a = self.account
        key = pd.Timestamp(asof).isoformat()
        # None is an explicit hold session in a scheduled-rebalance policy.
        # Fills, risk marking and persistence still run each observed session.
        hold_signature = hashlib.sha256(
            json.dumps([key, None, prices], sort_keys=True).encode()
        ).hexdigest()
        if key in a.decisions:
            if weights is None:
                if a.decisions[key] != hold_signature:
                    raise ValueError("Decision id reused with different inputs")
            else:
                a.rebalance(key, key, weights, prices)
            return a.snapshot(prices)
        # All fills, cancellations, targets and the mark commit as one event.
        # A failed check restores memory and leaves durable state unchanged.
        before = json.loads(json.dumps(a._payload()))
        event_count = len(a.events)
        pending = list(a._pending_events)
        revision = a.revision
        a._defer_save = True
        try:
            a.prices(prices)
            if a.last_event and pd.Timestamp(asof) < pd.Timestamp(a.last_event):
                raise ValueError("Out-of-order close")
            for o in sorted(a.pending(), key=lambda o: o["side"] == "buy"):
                if pd.Timestamp(o["decision_at"]) < pd.Timestamp(asof):
                    a.fill(
                        o["id"],
                        o["remaining"],
                        prices[o["symbol"]],
                        key,
                        prices,
                        fill_id=o["id"] + ":close",
                    )
            if weights is None:
                eq = a.equity(prices)
                a.peak = max(a.peak, eq)
                if eq / a.peak - 1 <= -a.limits.drawdown_limit:
                    a.rebalance(key, key, {s: 0.0 for s in a.symbols}, prices)
                elif not a._risk_ok(
                    {s: a.positions[s] * prices[s] for s in a.symbols}, eq
                ):
                    a.event(
                        "exposure_breach",
                        asof=key,
                        reason="Market drift on hold session",
                    )
                a.decisions[key] = hold_signature
                a.last_event = key
            else:
                a.rebalance(key, key, weights, prices)
            snapshot = a.snapshot(prices)
            a.event("mark", asof=key, **snapshot)
            a._defer_save = False
            a._save()
            return snapshot
        except Exception:
            a._defer_save = False
            a._restore(before)
            a.revision = revision
            del a.events[event_count:]
            a._pending_events = pending
            raise


def combine_targets(strategy_weights, allocations, symbols):
    """Combine conflicting strategy targets by their preassigned capital shares."""
    if (
        set(strategy_weights) != set(allocations)
        or any(v < 0 for v in allocations.values())
        or sum(allocations.values()) > 1 + 1e-9
    ):
        raise ValueError("Invalid strategy capital allocation")
    return {
        s: sum(
            allocations[name] * weights.get(s, 0.0)
            for name, weights in strategy_weights.items()
        )
        for s in symbols
    }
