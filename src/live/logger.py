"""
Phase 06E — Signal Logger (SQLite)

Persists every signal generated and every order submitted to a local SQLite
database at data/live/signal_log.db.

Schema
------
  signals  — one row per ticker per run:
             ticker, date, direction, label, confidence, p_up, p_flat, p_down,
             kelly_frac, atr, stop_loss, close, run_ts

  orders   — one row per Alpaca order submitted:
             order_id, ticker, side, qty, order_type, status,
             stop_price, take_profit, error, run_ts

  equity   — daily snapshot of paper account equity:
             equity, buying_power, run_ts

  circuit_breaker — peak equity tracker per ticker for drawdown guard:
             ticker, peak_equity, current_equity, last_updated
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DB_DIR       = _PROJECT_ROOT / "data" / "live"
_DB_PATH      = _DB_DIR / "signal_log.db"

_DDL = """
CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    direction   INTEGER NOT NULL,
    label       TEXT    NOT NULL,
    confidence  REAL,
    p_up        REAL,
    p_flat      REAL,
    p_down      REAL,
    kelly_frac  REAL,
    atr         REAL,
    stop_loss   REAL,
    close       REAL,
    run_ts      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id    TEXT,
    ticker      TEXT    NOT NULL,
    side        TEXT    NOT NULL,
    qty         REAL,
    order_type  TEXT,
    status      TEXT,
    stop_price  REAL,
    take_profit REAL,
    error       TEXT,
    run_ts      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS equity (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    equity        REAL    NOT NULL,
    buying_power  REAL,
    run_ts        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS circuit_breaker (
    ticker          TEXT    PRIMARY KEY,
    peak_equity     REAL    NOT NULL,
    current_equity  REAL,
    last_updated    TEXT    NOT NULL
);
"""


class SignalLogger:
    """
    SQLite-backed logger for signals, orders, and account state.

    Parameters
    ----------
    db_path : override default database location
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path else _DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Signal logging
    # ------------------------------------------------------------------

    def log_signal(self, signal) -> None:
        """Persist a LiveSignal to the signals table."""
        sql = """
            INSERT INTO signals
              (ticker, date, direction, label, confidence,
               p_up, p_flat, p_down, kelly_frac, atr, stop_loss, close, run_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                signal.ticker,
                signal.date.isoformat(),
                signal.direction,
                signal.label,
                signal.confidence,
                signal.p_up,
                signal.p_flat,
                signal.p_down,
                signal.kelly_frac,
                signal.atr,
                signal.stop_loss,
                signal.close,
                self._now(),
            ))
        log.debug("[Logger] signal logged: %s → %s", signal.ticker, signal.label)

    # ------------------------------------------------------------------
    # Order logging
    # ------------------------------------------------------------------

    def log_order(self, order) -> None:
        """Persist an OrderResult to the orders table."""
        sql = """
            INSERT INTO orders
              (order_id, ticker, side, qty, order_type, status,
               stop_price, take_profit, error, run_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                order.order_id,
                order.ticker,
                order.side,
                order.qty,
                order.order_type,
                order.status,
                order.stop_price,
                order.take_profit,
                order.error,
                self._now(),
            ))
        log.debug("[Logger] order logged: %s %s %s", order.side, order.ticker, order.status)

    # ------------------------------------------------------------------
    # Equity snapshot
    # ------------------------------------------------------------------

    def log_equity(self, equity: float, buying_power: float) -> None:
        sql = "INSERT INTO equity (equity, buying_power, run_ts) VALUES (?, ?, ?)"
        with self._conn() as conn:
            conn.execute(sql, (equity, buying_power, self._now()))

    # ------------------------------------------------------------------
    # Circuit breaker state
    # ------------------------------------------------------------------

    def update_circuit_breaker(self, ticker: str, current_equity: float) -> None:
        """
        Upsert per-ticker peak equity.  Called after every run so the engine
        can detect a >15% peak-to-trough drawdown without querying Alpaca.
        """
        now = self._now()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT peak_equity FROM circuit_breaker WHERE ticker = ?", (ticker,)
            ).fetchone()

            if row is None:
                conn.execute(
                    "INSERT INTO circuit_breaker (ticker, peak_equity, current_equity, last_updated)"
                    " VALUES (?, ?, ?, ?)",
                    (ticker, current_equity, current_equity, now),
                )
            else:
                new_peak = max(float(row[0]), current_equity)
                conn.execute(
                    "UPDATE circuit_breaker SET peak_equity=?, current_equity=?, last_updated=?"
                    " WHERE ticker=?",
                    (new_peak, current_equity, now, ticker),
                )

    def is_circuit_broken(self, ticker: str, threshold: float = 0.15) -> bool:
        """
        Return True if current_equity has fallen more than `threshold` from
        peak_equity (i.e. drawdown > threshold).
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT peak_equity, current_equity FROM circuit_breaker WHERE ticker = ?",
                (ticker,),
            ).fetchone()

        if row is None:
            return False
        peak, current = float(row[0]), float(row[1])
        if peak <= 0:
            return False
        drawdown = (peak - current) / peak
        return drawdown > threshold

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def recent_signals(self, n: int = 50) -> list[dict]:
        """Return the `n` most recent signal rows as dicts."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM signals ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    def recent_orders(self, n: int = 50) -> list[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM orders ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    def equity_history(self, n: int = 90) -> list[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM equity ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]
