"""
Phase 06E — Tests for the Live Signal Engine

Coverage
--------
  SignalGenerator  — LiveSignal construction, is_actionable, label
  SignalLogger     — SQLite CRUD: signals, orders, equity, circuit breaker
  AlpacaBroker     — credential guard (no network calls)
  Alerts           — smoke tests (no real HTTP / SMTP)
  LiveEngine       — dry_run flow with mocked generator and broker
  _kelly_fraction  — edge cases
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

def _make_signal(
    ticker="AAPL",
    direction=2,
    confidence=0.55,
    kelly_frac=0.08,
    p_up=0.55,
    p_flat=0.25,
    p_down=0.20,
    atr=1.5,
    stop_loss=178.0,
    close=181.0,
):
    from src.live.signal_generator import LiveSignal
    return LiveSignal(
        ticker      = ticker,
        date        = datetime(2025, 1, 10, tzinfo=timezone.utc),
        direction   = direction,
        confidence  = confidence,
        p_up        = p_up,
        p_flat      = p_flat,
        p_down      = p_down,
        kelly_frac  = kelly_frac,
        atr         = atr,
        stop_loss   = stop_loss,
        close       = close,
    )


def _make_order(order_id="ord-001", ticker="AAPL", side="buy", qty=1.0,
                order_type="bracket", status="new",
                stop_price=178.0, take_profit=185.5, error=None):
    from src.live.broker import OrderResult
    return OrderResult(
        order_id    = order_id,
        ticker      = ticker,
        side        = side,
        qty         = qty,
        order_type  = order_type,
        status      = status,
        stop_price  = stop_price,
        take_profit = take_profit,
        error       = error,
    )


# ===========================================================================
# LiveSignal
# ===========================================================================

class TestLiveSignal:
    def test_label_up(self):
        s = _make_signal(direction=2)
        assert s.label == "UP"

    def test_label_flat(self):
        s = _make_signal(direction=1, kelly_frac=0.0)
        assert s.label == "FLAT"

    def test_label_down(self):
        s = _make_signal(direction=0, kelly_frac=0.05)
        assert s.label == "DOWN"

    def test_is_actionable_up_with_kelly(self):
        s = _make_signal(direction=2, kelly_frac=0.05)
        assert s.is_actionable()

    def test_not_actionable_flat(self):
        s = _make_signal(direction=1, kelly_frac=0.0)
        assert not s.is_actionable()

    def test_not_actionable_up_zero_kelly(self):
        s = _make_signal(direction=2, kelly_frac=0.0)
        assert not s.is_actionable()

    def test_not_actionable_down_zero_kelly(self):
        # DOWN with kelly=0 — engine will close existing positions anyway
        # but is_actionable reflects no directional edge
        s = _make_signal(direction=0, kelly_frac=0.0)
        assert not s.is_actionable()


# ===========================================================================
# Kelly fraction
# ===========================================================================

class TestKellyFraction:
    def _kelly(self, p_win, wl_ratio, k=0.5):
        from src.live.signal_generator import _kelly_fraction
        return _kelly_fraction(p_win, wl_ratio, k)

    def test_positive_edge(self):
        # p=0.6, b=2 → f* = (0.6×2 - 0.4)/2 = 0.4 → half-Kelly = 0.2
        f = self._kelly(0.6, 2.0)
        assert abs(f - 0.2) < 1e-9

    def test_zero_edge_returns_zero(self):
        # p=0.5, b=1 → f* = (0.5-0.5)/1 = 0 → half-Kelly = 0
        assert self._kelly(0.5, 1.0) == pytest.approx(0.0)

    def test_negative_edge_returns_zero(self):
        # p=0.3, b=1 → f* = -0.4 → clip to 0
        assert self._kelly(0.3, 1.0) == 0.0

    def test_invalid_p_zero(self):
        assert self._kelly(0.0, 1.5) == 0.0

    def test_invalid_p_one(self):
        assert self._kelly(1.0, 1.5) == 0.0

    def test_invalid_wl_zero(self):
        assert self._kelly(0.6, 0.0) == 0.0

    def test_custom_multiplier(self):
        f_half = self._kelly(0.6, 2.0, k=0.5)
        f_full = self._kelly(0.6, 2.0, k=1.0)
        assert abs(f_full - 2 * f_half) < 1e-9


# ===========================================================================
# SignalLogger (SQLite)
# ===========================================================================

class TestSignalLogger:
    def _logger(self, tmp_path: Path):
        from src.live.logger import SignalLogger
        return SignalLogger(db_path=tmp_path / "test.db")

    def test_db_created(self, tmp_path):
        logger = self._logger(tmp_path)
        assert (tmp_path / "test.db").exists()

    def test_log_signal(self, tmp_path):
        logger = self._logger(tmp_path)
        logger.log_signal(_make_signal())
        rows = logger.recent_signals(10)
        assert len(rows) == 1
        assert rows[0]["ticker"] == "AAPL"
        assert rows[0]["label"] == "UP"

    def test_log_multiple_signals(self, tmp_path):
        logger = self._logger(tmp_path)
        for t in ["AAPL", "MSFT", "GOOGL"]:
            logger.log_signal(_make_signal(ticker=t))
        rows = logger.recent_signals(10)
        assert len(rows) == 3

    def test_log_order(self, tmp_path):
        logger = self._logger(tmp_path)
        logger.log_order(_make_order())
        rows = logger.recent_orders(10)
        assert len(rows) == 1
        assert rows[0]["order_id"] == "ord-001"
        assert rows[0]["side"] == "buy"

    def test_log_equity(self, tmp_path):
        logger = self._logger(tmp_path)
        logger.log_equity(10_000.0, 5_000.0)
        rows = logger.equity_history(5)
        assert len(rows) == 1
        assert rows[0]["equity"] == pytest.approx(10_000.0)
        assert rows[0]["buying_power"] == pytest.approx(5_000.0)

    def test_circuit_breaker_no_drawdown(self, tmp_path):
        logger = self._logger(tmp_path)
        logger.update_circuit_breaker("AAPL", 10_000.0)
        logger.update_circuit_breaker("AAPL", 10_000.0)
        assert not logger.is_circuit_broken("AAPL", threshold=0.15)

    def test_circuit_breaker_trips(self, tmp_path):
        logger = self._logger(tmp_path)
        logger.update_circuit_breaker("AAPL", 10_000.0)
        logger.update_circuit_breaker("AAPL", 8_000.0)   # 20% drawdown
        assert logger.is_circuit_broken("AAPL", threshold=0.15)

    def test_circuit_breaker_no_trip_small_dd(self, tmp_path):
        logger = self._logger(tmp_path)
        logger.update_circuit_breaker("AAPL", 10_000.0)
        logger.update_circuit_breaker("AAPL", 9_200.0)   # 8% drawdown
        assert not logger.is_circuit_broken("AAPL", threshold=0.15)

    def test_circuit_breaker_peak_persists(self, tmp_path):
        """Peak equity must not decrease even after equity drops."""
        logger = self._logger(tmp_path)
        logger.update_circuit_breaker("AAPL", 10_000.0)
        logger.update_circuit_breaker("AAPL", 12_000.0)  # new peak
        logger.update_circuit_breaker("AAPL", 9_000.0)   # 25% from 12k
        assert logger.is_circuit_broken("AAPL", threshold=0.15)

    def test_missing_ticker_not_broken(self, tmp_path):
        logger = self._logger(tmp_path)
        assert not logger.is_circuit_broken("NONEXISTENT")

    def test_recent_signals_order(self, tmp_path):
        """Most recent signal should appear first."""
        logger = self._logger(tmp_path)
        logger.log_signal(_make_signal(ticker="AAPL"))
        logger.log_signal(_make_signal(ticker="MSFT"))
        rows = logger.recent_signals(5)
        assert rows[0]["ticker"] == "MSFT"   # most recent first

    def test_log_error_order(self, tmp_path):
        logger = self._logger(tmp_path)
        order = _make_order(status="error", error="insufficient funds", order_id="")
        logger.log_order(order)
        rows = logger.recent_orders(5)
        assert rows[0]["error"] == "insufficient funds"


# ===========================================================================
# OrderResult
# ===========================================================================

class TestOrderResult:
    def test_ok_when_no_error(self):
        order = _make_order(error=None)
        assert order.ok

    def test_not_ok_when_error(self):
        order = _make_order(error="rejected")
        assert not order.ok


# ===========================================================================
# AlpacaBroker — credential guard (no network)
# ===========================================================================

class TestAlpacaBrokerCredentials:
    def test_raises_without_credentials(self):
        from src.live.broker import AlpacaBroker
        with patch.dict("os.environ", {}, clear=True):
            # Remove any existing Alpaca env vars
            import os
            for k in ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]:
                os.environ.pop(k, None)
            with pytest.raises(EnvironmentError, match="credentials not set"):
                AlpacaBroker(api_key="", secret_key="")

    def test_import_error_bubbles(self):
        """If alpaca-trade-api is not installed, a clear ImportError is raised."""
        from src.live.broker import AlpacaBroker
        with patch.dict("os.environ", {
            "ALPACA_API_KEY":    "test-key",
            "ALPACA_SECRET_KEY": "test-secret",
        }):
            with patch("builtins.__import__", side_effect=ImportError("no module")):
                with pytest.raises((ImportError, Exception)):
                    AlpacaBroker()


# ===========================================================================
# Alerts — smoke tests (no real network calls)
# ===========================================================================

class TestAlerts:
    def test_no_op_without_config(self):
        from src.live.alerts import Alerts
        alerts = Alerts(slack_url=None, alert_email=None)
        # Should not raise
        alerts.signal_fired(_make_signal(), _make_order())
        alerts.circuit_breaker_tripped("AAPL", 0.18)
        alerts.order_error("AAPL", "something went wrong")
        alerts.daily_summary([_make_signal()], 10_000.0)

    def test_slack_called_when_configured(self):
        from src.live.alerts import Alerts
        alerts = Alerts(slack_url="https://hooks.slack.com/fake")
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.status = 200
            mock_open.return_value = mock_resp
            alerts.signal_fired(_make_signal(), _make_order())
        mock_open.assert_called_once()

    def test_email_called_when_configured(self):
        from src.live.alerts import Alerts
        alerts = Alerts(
            alert_email = "trader@example.com",
            smtp_host   = "smtp.example.com",
            smtp_port   = 587,
            smtp_user   = "sender@example.com",
            smtp_pass   = "password",
        )
        with patch("smtplib.SMTP") as mock_smtp:
            ctx = MagicMock()
            mock_smtp.return_value.__enter__ = MagicMock(return_value=ctx)
            mock_smtp.return_value.__exit__  = MagicMock(return_value=False)
            alerts.signal_fired(_make_signal(), _make_order())
        mock_smtp.assert_called_once()


# ===========================================================================
# LiveEngine — dry_run with mocked dependencies
# ===========================================================================

class TestLiveEngine:
    def _engine(self, tmp_path=None):
        from src.live.engine import LiveEngine
        return LiveEngine(
            tickers  = ["AAPL", "MSFT"],
            dry_run  = True,
        )

    def test_dry_run_produces_result(self, tmp_path):
        from src.live.engine import LiveEngine
        from src.live.logger import SignalLogger

        # Patch SignalGenerator and SignalLogger to avoid disk I/O
        mock_gen = MagicMock()
        mock_gen.generate_batch.return_value = {
            "AAPL": _make_signal("AAPL", direction=2, kelly_frac=0.05),
            "MSFT": _make_signal("MSFT", direction=1, kelly_frac=0.0),
        }

        mock_logger = MagicMock()

        with patch("src.live.engine.SignalGenerator", return_value=mock_gen), \
             patch("src.live.engine.SignalLogger", return_value=mock_logger), \
             patch("src.live.engine.AlpacaBroker", side_effect=EnvironmentError("no creds")), \
             patch("src.live.engine.Alerts"):
            engine = LiveEngine(tickers=["AAPL", "MSFT"], dry_run=True)
            result = engine.run()

        assert "AAPL" in result.signals
        assert "MSFT" in result.signals
        # MSFT is FLAT → should be skipped
        any_skip_msft = any(t == "MSFT" for t, _ in result.skipped)
        assert any_skip_msft

    def test_dry_run_no_orders(self, tmp_path):
        from src.live.engine import LiveEngine

        mock_gen = MagicMock()
        mock_gen.generate_batch.return_value = {
            "AAPL": _make_signal("AAPL", direction=2, kelly_frac=0.05),
        }

        with patch("src.live.engine.SignalGenerator", return_value=mock_gen), \
             patch("src.live.engine.SignalLogger"), \
             patch("src.live.engine.AlpacaBroker", side_effect=EnvironmentError("no creds")), \
             patch("src.live.engine.Alerts"):
            engine = LiveEngine(tickers=["AAPL"], dry_run=True)
            result = engine.run()

        # dry_run → orders dict has None values (no real order submitted)
        for order in result.orders.values():
            assert order is None

    def test_flat_signal_skipped(self):
        from src.live.engine import LiveEngine

        mock_gen = MagicMock()
        mock_gen.generate_batch.return_value = {
            "SPY": _make_signal("SPY", direction=1, kelly_frac=0.0),
        }

        with patch("src.live.engine.SignalGenerator", return_value=mock_gen), \
             patch("src.live.engine.SignalLogger"), \
             patch("src.live.engine.AlpacaBroker", side_effect=EnvironmentError("no creds")), \
             patch("src.live.engine.Alerts"):
            engine = LiveEngine(tickers=["SPY"], dry_run=True)
            result = engine.run()

        reasons = [r for _, r in result.skipped]
        assert "flat_signal" in reasons

    def test_zero_kelly_skipped(self):
        from src.live.engine import LiveEngine

        mock_gen = MagicMock()
        mock_gen.generate_batch.return_value = {
            "QQQ": _make_signal("QQQ", direction=2, kelly_frac=0.0),
        }

        with patch("src.live.engine.SignalGenerator", return_value=mock_gen), \
             patch("src.live.engine.SignalLogger"), \
             patch("src.live.engine.AlpacaBroker", side_effect=EnvironmentError("no creds")), \
             patch("src.live.engine.Alerts"):
            engine = LiveEngine(tickers=["QQQ"], dry_run=True)
            result = engine.run()

        reasons = [r for _, r in result.skipped]
        assert "no_kelly_edge" in reasons
