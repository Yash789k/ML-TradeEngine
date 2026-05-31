"""
Phase 06E — Live Engine

Orchestrates the full daily live-trading pipeline:

  1. Fetch fresh data + run ensemble inference → LiveSignal per ticker
  2. Apply risk gates (confidence, Kelly, circuit breaker, portfolio heat)
  3. Submit bracket orders to Alpaca paper trading
  4. Log signals + orders + account equity to SQLite
  5. Send Slack / email alerts

Designed to be idempotent — running twice on the same day for the same
ticker will skip order submission if a position or order already exists.

Usage (from live.py CLI or GitHub Actions)
-------------------------------------------
  engine = LiveEngine()
  results = engine.run(tickers=["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"])

Configuration via environment variables (see broker.py and alerts.py).

Risk gates applied in order
---------------------------
  1. direction == FLAT → skip (no action)
  2. kelly_frac <= 0   → skip (model has no statistical edge)
  3. confidence < threshold → already filtered in SignalGenerator
  4. circuit_breaker   → skip new entries; close existing if circuit broken
  5. already_ordered   → skip (idempotency guard)
  6. portfolio heat    → skip if total deployed capital > max_heat
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from src.live.signal_generator import SignalGenerator  # noqa: F401  (imported for patchability)
from src.live.broker import AlpacaBroker               # noqa: F401
from src.live.logger import SignalLogger               # noqa: F401
from src.live.alerts import Alerts                     # noqa: F401

log = logging.getLogger(__name__)

_DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]

_MAX_PORTFOLIO_HEAT   = 0.80   # max fraction of account equity deployed at once
_CIRCUIT_BREAKER_DD   = 0.15   # trip at 15% drawdown (matches Phase 05)
_ATR_TP_MULTIPLIER    = 3.0    # take-profit at entry + 3×ATR (R:R ≈ 1.5:1)
_MAX_POSITION_FRAC    = 0.05   # max 5% of equity per single position


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    signals:       dict = field(default_factory=dict)    # ticker → LiveSignal
    orders:        dict = field(default_factory=dict)    # ticker → OrderResult | None
    skipped:       list = field(default_factory=list)    # [(ticker, reason)]
    errors:        list = field(default_factory=list)    # [(ticker, error_str)]
    account_equity: Optional[float] = None


# ---------------------------------------------------------------------------
# LiveEngine
# ---------------------------------------------------------------------------

class LiveEngine:
    """
    End-to-end live signal and execution engine.

    Parameters
    ----------
    tickers               : default ticker list (overridable at run() time)
    confidence_threshold  : minimum ensemble confidence to act  (default 0.38)
    max_portfolio_heat    : max fraction of equity deployed      (default 80%)
    max_position_frac     : max fraction per position            (default  5%)
    circuit_breaker_dd    : circuit breaker drawdown threshold   (default 15%)
    atr_tp_multiplier     : take-profit = entry + N × ATR        (default 3×)
    dry_run               : log signals but do NOT submit orders (default False)
    """

    def __init__(
        self,
        tickers:              Optional[list[str]] = None,
        confidence_threshold: float = 0.38,
        max_portfolio_heat:   float = _MAX_PORTFOLIO_HEAT,
        max_position_frac:    float = _MAX_POSITION_FRAC,
        circuit_breaker_dd:   float = _CIRCUIT_BREAKER_DD,
        atr_tp_multiplier:    float = _ATR_TP_MULTIPLIER,
        dry_run:              bool  = False,
    ) -> None:
        self.tickers              = tickers or _DEFAULT_TICKERS
        self.confidence_threshold = confidence_threshold
        self.max_portfolio_heat   = max_portfolio_heat
        self.max_position_frac    = max_position_frac
        self.circuit_breaker_dd   = circuit_breaker_dd
        self.atr_tp_multiplier    = atr_tp_multiplier
        self.dry_run              = dry_run

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        tickers: Optional[list[str]] = None,
        force_refresh: bool = True,
    ) -> RunResult:
        """
        Run the full daily pipeline.

        Parameters
        ----------
        tickers       : override the engine's default ticker list
        force_refresh : bypass DataLoader/FeatureEngineer caches

        Returns
        -------
        RunResult with signals, orders, skipped list, and errors
        """
        ticker_list = tickers or self.tickers
        result      = RunResult()

        log.info("══════════════════════════════════════")
        log.info("  Live Engine  dry_run=%s", self.dry_run)
        log.info("  Tickers: %s", ticker_list)
        log.info("══════════════════════════════════════")

        # ── Instantiate components ──────────────────────────────────────
        gen    = SignalGenerator(confidence_threshold=self.confidence_threshold)
        logger = SignalLogger()
        alerts = Alerts()

        broker: Optional[AlpacaBroker] = None
        if not self.dry_run:
            try:
                broker = AlpacaBroker()
            except EnvironmentError as exc:
                log.warning(
                    "Alpaca credentials not found (%s). Switching to dry_run=True.", exc
                )
                self.dry_run = True

        # ── Account snapshot ────────────────────────────────────────────
        equity = buying_power = 0.0
        if broker:
            try:
                equity        = broker.account_equity()
                buying_power  = broker.buying_power()
                result.account_equity = equity
                logger.log_equity(equity, buying_power)
                log.info("  Account equity: $%.2f  buying_power: $%.2f", equity, buying_power)
            except Exception as exc:
                log.warning("Could not fetch account data: %s", exc)

        # ── Open positions (for heat check) ─────────────────────────────
        open_positions: dict[str, object] = {}
        if broker:
            try:
                for pos in broker.get_all_positions():
                    open_positions[pos.ticker] = pos
                total_market_val = sum(p.market_val for p in open_positions.values())
                if equity > 0:
                    current_heat = total_market_val / equity
                    log.info(
                        "  Portfolio heat: %.1f%%  (%d open positions)",
                        current_heat * 100, len(open_positions),
                    )
            except Exception as exc:
                log.warning("Could not fetch positions: %s", exc)

        # ── Generate signals ─────────────────────────────────────────────
        log.info("\n── Generating signals …")
        signals = gen.generate_batch(ticker_list, force_refresh=force_refresh)
        result.signals = signals

        # ── Process each signal ──────────────────────────────────────────
        for ticker, signal in signals.items():
            logger.log_signal(signal)

            log.info(
                "  %s → %s  conf=%.3f  kelly=%.3f",
                ticker, signal.label, signal.confidence, signal.kelly_frac,
            )

            # ── Risk gate 1: FLAT signal ─────────────────────────────────
            if signal.direction == 1:
                result.skipped.append((ticker, "flat_signal"))
                log.info("    → SKIP (%s is FLAT)", ticker)
                continue

            # ── Risk gate 2: No Kelly edge ───────────────────────────────
            if signal.direction == 2 and signal.kelly_frac <= 0.0:
                result.skipped.append((ticker, "no_kelly_edge"))
                log.info("    → SKIP (%s has no Kelly edge)", ticker)
                continue

            # ── Risk gate 3: Circuit breaker check ──────────────────────
            if broker and signal.direction == 2:
                try:
                    acct_eq = broker.account_equity()
                    logger.update_circuit_breaker(ticker, acct_eq)
                    if logger.is_circuit_broken(ticker, self.circuit_breaker_dd):
                        result.skipped.append((ticker, "circuit_breaker"))
                        alerts.circuit_breaker_tripped(ticker, self.circuit_breaker_dd)
                        log.warning("    → SKIP (%s circuit breaker active)", ticker)
                        continue
                except Exception as exc:
                    log.warning("    Circuit breaker check failed: %s", exc)

            # ── Risk gate 4: Idempotency — already ordered today ─────────
            if broker and not self.dry_run:
                try:
                    if broker.already_ordered_today(ticker):
                        result.skipped.append((ticker, "already_ordered_today"))
                        log.info("    → SKIP (%s already ordered today)", ticker)
                        continue
                except Exception as exc:
                    log.warning("    Could not check today's orders: %s", exc)

            # ── Risk gate 5: Portfolio heat ──────────────────────────────
            if broker and equity > 0 and signal.direction == 2:
                try:
                    positions = broker.get_all_positions()
                    total_val = sum(p.market_val for p in positions)
                    heat = total_val / equity
                    if heat >= self.max_portfolio_heat:
                        result.skipped.append((ticker, "portfolio_heat_limit"))
                        log.info(
                            "    → SKIP (%s: heat %.1f%% >= limit %.1f%%)",
                            ticker, heat * 100, self.max_portfolio_heat * 100,
                        )
                        continue
                except Exception as exc:
                    log.warning("    Heat check failed: %s", exc)

            # ── Execute ──────────────────────────────────────────────────
            order = self._execute(ticker, signal, broker, equity)
            result.orders[ticker] = order

            if order is not None:
                logger.log_order(order)
                if order.ok:
                    alerts.signal_fired(signal, order)
                else:
                    alerts.order_error(ticker, order.error or "unknown error")

        # ── Daily summary alert ──────────────────────────────────────────
        if result.signals:
            alerts.daily_summary(list(result.signals.values()), equity)

        log.info("\n── Run complete.")
        log.info("  Signals: %d  |  Orders: %d  |  Skipped: %d  |  Errors: %d",
                 len(result.signals), len(result.orders),
                 len(result.skipped), len(result.errors))
        return result

    # ------------------------------------------------------------------
    # Execution logic
    # ------------------------------------------------------------------

    def _execute(
        self,
        ticker:  str,
        signal,
        broker:  Optional[object],
        equity:  float,
    ):
        """
        Decide and submit the appropriate order based on the signal direction.
        Returns OrderResult or None in dry_run mode.
        """
        from src.live.broker import OrderResult

        if self.dry_run or broker is None:
            log.info("    [dry_run] would %s %s", signal.label, ticker)
            return None

        try:
            if signal.direction == 2:   # UP → enter long
                qty         = self._compute_qty(signal, equity, broker)
                take_profit = round(signal.close + self.atr_tp_multiplier * signal.atr, 2)
                order = broker.submit_bracket_buy(
                    ticker      = ticker,
                    qty         = qty,
                    stop_price  = signal.stop_loss,
                    take_profit = take_profit,
                )

            elif signal.direction == 0:   # DOWN → exit long if held
                pos = broker.get_position(ticker)
                if pos is None:
                    log.info("    %s: DOWN signal but no open position — skip", ticker)
                    return None
                order = broker.close_position(ticker)

            else:
                return None

            return order

        except Exception as exc:
            log.error("[Engine] order execution failed for %s: %s", ticker, exc)
            return OrderResult(
                order_id="", ticker=ticker, side="?", qty=0,
                order_type="?", status="error", error=str(exc),
            )

    def _compute_qty(self, signal, equity: float, broker) -> float:
        """
        Position size = min(kelly_frac, max_position_frac) × equity / close.
        Clips to available buying power.
        """
        if equity <= 0 or signal.close <= 0:
            return 0.0

        frac        = min(signal.kelly_frac, self.max_position_frac)
        dollar_size = frac * equity
        try:
            bp          = broker.buying_power()
            dollar_size = min(dollar_size, bp * 0.95)   # leave 5% buffer
        except Exception:
            pass

        qty = dollar_size / signal.close
        return round(max(qty, 0.0), 4)
