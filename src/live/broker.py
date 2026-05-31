"""
Phase 06E — Alpaca Paper Trading Broker

Thin wrapper around the alpaca-trade-api REST client targeting the paper
trading endpoint.  Handles bracket order submission, position queries, and
account equity tracking.

Required environment variables
-------------------------------
  ALPACA_API_KEY    — paper account API key
  ALPACA_SECRET_KEY — paper account secret key
  ALPACA_BASE_URL   — defaults to https://paper-api.alpaca.markets

Bracket order logic
-------------------
  entry  : market order submitted after market close / premarket (fills at open)
  stop   : ATR-based hard stop-loss  (entry - 2×ATR)
  target : 3×ATR profit target above entry  (risk:reward ≈ 1.5:1)

All positions are long-only (matches Phase 05 risk engine default mode).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

_PAPER_URL = "https://paper-api.alpaca.markets"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OrderResult:
    order_id:   str
    ticker:     str
    side:       str           # "buy" | "sell"
    qty:        float
    order_type: str           # "bracket" | "market"
    status:     str           # Alpaca order status
    stop_price: Optional[float] = None
    take_profit: Optional[float] = None
    error:      Optional[str]  = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class Position:
    ticker:     str
    qty:        float
    side:       str           # "long" | "short"
    avg_entry:  float
    market_val: float
    unrealized_pl: float


# ---------------------------------------------------------------------------
# AlpacaBroker
# ---------------------------------------------------------------------------

class AlpacaBroker:
    """
    Paper-trading broker using Alpaca REST API.

    Parameters
    ----------
    api_key    : Alpaca API key (defaults to ALPACA_API_KEY env var)
    secret_key : Alpaca secret  (defaults to ALPACA_SECRET_KEY env var)
    base_url   : API base URL   (defaults to ALPACA_BASE_URL env var or paper URL)
    """

    def __init__(
        self,
        api_key:    Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url:   Optional[str] = None,
    ) -> None:
        self.api_key    = api_key    or os.environ.get("ALPACA_API_KEY",    "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self.base_url   = base_url   or os.environ.get("ALPACA_BASE_URL",   _PAPER_URL)

        if not self.api_key or not self.secret_key:
            raise EnvironmentError(
                "Alpaca credentials not set. "
                "Export ALPACA_API_KEY and ALPACA_SECRET_KEY."
            )

        self._api = self._build_client()

    # ------------------------------------------------------------------
    # Alpaca client construction
    # ------------------------------------------------------------------

    def _build_client(self):
        try:
            import alpaca_trade_api as tradeapi
            return tradeapi.REST(
                key_id     = self.api_key,
                secret_key = self.secret_key,
                base_url   = self.base_url,
                api_version = "v2",
            )
        except ImportError as exc:
            raise ImportError(
                "alpaca-trade-api not installed. "
                "Run: pip install alpaca-trade-api"
            ) from exc

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def account_equity(self) -> float:
        """Return current paper account equity in USD."""
        acct = self._api.get_account()
        return float(acct.equity)

    def buying_power(self) -> float:
        """Return available buying power."""
        acct = self._api.get_account()
        return float(acct.buying_power)

    def is_market_open(self) -> bool:
        clock = self._api.get_clock()
        return bool(clock.is_open)

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_position(self, ticker: str) -> Optional[Position]:
        """Return current position for `ticker`, or None if flat."""
        try:
            p = self._api.get_position(ticker)
            return Position(
                ticker        = ticker,
                qty           = float(p.qty),
                side          = p.side,
                avg_entry     = float(p.avg_entry_price),
                market_val    = float(p.market_value),
                unrealized_pl = float(p.unrealized_pl),
            )
        except Exception:
            return None   # no open position

    def get_all_positions(self) -> list[Position]:
        """Return all open positions."""
        positions = []
        for p in self._api.list_positions():
            positions.append(Position(
                ticker        = p.symbol,
                qty           = float(p.qty),
                side          = p.side,
                avg_entry     = float(p.avg_entry_price),
                market_val    = float(p.market_value),
                unrealized_pl = float(p.unrealized_pl),
            ))
        return positions

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_bracket_buy(
        self,
        ticker:      str,
        qty:         float,
        stop_price:  float,
        take_profit: float,
    ) -> OrderResult:
        """
        Submit a market bracket buy order.

        Parameters
        ----------
        ticker      : asset symbol
        qty         : number of shares (fractional supported on Alpaca)
        stop_price  : stop-loss price  (entry - 2×ATR)
        take_profit : limit profit target (entry + 3×ATR)
        """
        qty = round(qty, 4)
        if qty <= 0:
            return OrderResult(
                order_id="", ticker=ticker, side="buy", qty=qty,
                order_type="bracket", status="rejected",
                error=f"qty={qty} must be > 0",
            )

        try:
            order = self._api.submit_order(
                symbol         = ticker,
                qty            = qty,
                side           = "buy",
                type           = "market",
                time_in_force  = "day",
                order_class    = "bracket",
                stop_loss      = {"stop_price": round(stop_price,  2)},
                take_profit    = {"limit_price": round(take_profit, 2)},
            )
            log.info(
                "[Broker] BUY bracket submitted: %s  qty=%.4f  stop=%.2f  tp=%.2f  id=%s",
                ticker, qty, stop_price, take_profit, order.id,
            )
            return OrderResult(
                order_id    = order.id,
                ticker      = ticker,
                side        = "buy",
                qty         = qty,
                order_type  = "bracket",
                status      = order.status,
                stop_price  = stop_price,
                take_profit = take_profit,
            )
        except Exception as exc:
            log.error("[Broker] Failed to submit buy for %s: %s", ticker, exc)
            return OrderResult(
                order_id="", ticker=ticker, side="buy", qty=qty,
                order_type="bracket", status="error", error=str(exc),
            )

    def close_position(self, ticker: str) -> OrderResult:
        """Close (liquidate) an existing position at market."""
        try:
            resp = self._api.close_position(ticker)
            log.info("[Broker] CLOSE position: %s  id=%s", ticker, resp.id)
            return OrderResult(
                order_id   = resp.id,
                ticker     = ticker,
                side       = "sell",
                qty        = float(resp.qty),
                order_type = "market",
                status     = resp.status,
            )
        except Exception as exc:
            log.error("[Broker] Failed to close position for %s: %s", ticker, exc)
            return OrderResult(
                order_id="", ticker=ticker, side="sell", qty=0,
                order_type="market", status="error", error=str(exc),
            )

    def cancel_open_orders(self, ticker: str) -> int:
        """Cancel all open orders for `ticker`. Returns count of cancelled orders."""
        orders = self._api.list_orders(status="open", symbols=[ticker])
        cancelled = 0
        for order in orders:
            try:
                self._api.cancel_order(order.id)
                cancelled += 1
            except Exception as exc:
                log.warning("[Broker] Could not cancel order %s: %s", order.id, exc)
        return cancelled

    def today_orders(self, ticker: Optional[str] = None) -> list[dict]:
        """Return all orders placed today (for idempotency check)."""
        from datetime import date, datetime, timezone
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
        kwargs: dict = {"status": "all", "after": today_str, "limit": 200}
        if ticker:
            kwargs["symbols"] = [ticker]
        orders = self._api.list_orders(**kwargs)
        return [
            {
                "id":     o.id,
                "ticker": o.symbol,
                "side":   o.side,
                "status": o.status,
                "qty":    float(o.qty),
            }
            for o in orders
        ]

    def already_ordered_today(self, ticker: str) -> bool:
        """True if any order for `ticker` was submitted today."""
        return len(self.today_orders(ticker)) > 0
