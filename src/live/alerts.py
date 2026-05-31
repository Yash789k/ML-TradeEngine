"""
Phase 06E — Alerts (Slack webhook + Email)

Sends notifications when:
  - A directional signal fires (UP / DOWN with an actionable order)
  - An order is placed or rejected
  - A circuit breaker trips (>15% drawdown)

Configuration via environment variables
----------------------------------------
  SLACK_WEBHOOK_URL — Slack incoming webhook URL (optional)
  ALERT_EMAIL       — recipient email address (optional)
  SMTP_HOST         — SMTP server host (default: smtp.gmail.com)
  SMTP_PORT         — SMTP server port (default: 587)
  SMTP_USER         — SMTP login user (sender address)
  SMTP_PASS         — SMTP login password / app-password
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import urllib.request
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Optional

log = logging.getLogger(__name__)


class Alerts:
    """
    Thin notification wrapper.  Each method is a no-op when the relevant
    credentials are not configured — the live engine is never blocked by
    a missing alert destination.

    Parameters
    ----------
    slack_url   : Slack webhook URL (falls back to SLACK_WEBHOOK_URL env var)
    alert_email : recipient email  (falls back to ALERT_EMAIL env var)
    smtp_host   : SMTP host  (falls back to SMTP_HOST env var)
    smtp_port   : SMTP port  (falls back to SMTP_PORT env var)
    smtp_user   : SMTP user  (falls back to SMTP_USER env var)
    smtp_pass   : SMTP password (falls back to SMTP_PASS env var)
    """

    def __init__(
        self,
        slack_url:   Optional[str] = None,
        alert_email: Optional[str] = None,
        smtp_host:   Optional[str] = None,
        smtp_port:   Optional[int] = None,
        smtp_user:   Optional[str] = None,
        smtp_pass:   Optional[str] = None,
    ) -> None:
        self.slack_url   = slack_url   or os.environ.get("SLACK_WEBHOOK_URL")
        self.alert_email = alert_email or os.environ.get("ALERT_EMAIL")
        self.smtp_host   = smtp_host   or os.environ.get("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port   = smtp_port   or int(os.environ.get("SMTP_PORT", "587"))
        self.smtp_user   = smtp_user   or os.environ.get("SMTP_USER")
        self.smtp_pass   = smtp_pass   or os.environ.get("SMTP_PASS")

    # ------------------------------------------------------------------
    # High-level event methods
    # ------------------------------------------------------------------

    def signal_fired(self, signal, order) -> None:
        """Alert when an actionable signal triggers an order."""
        emoji   = "🟢" if signal.direction == 2 else "🔴"
        subject = f"{emoji} ML Signal: {signal.ticker} → {signal.label}"
        body    = (
            f"Ticker:      {signal.ticker}\n"
            f"Direction:   {signal.label}\n"
            f"Confidence:  {signal.confidence:.1%}\n"
            f"Kelly frac:  {signal.kelly_frac:.3f}\n"
            f"Close:       ${signal.close:.2f}\n"
            f"Stop loss:   ${signal.stop_loss:.2f}\n"
            f"ATR:         {signal.atr:.4f}\n"
            f"\nOrder ID:    {order.order_id}\n"
            f"Status:      {order.status}\n"
            f"Qty:         {order.qty:.4f}\n"
            f"Run time:    {datetime.now(timezone.utc).isoformat()}\n"
        )
        self._send(subject, body)

    def circuit_breaker_tripped(self, ticker: str, drawdown: float) -> None:
        """Alert when the portfolio circuit breaker trips."""
        subject = f"⚠️  Circuit Breaker: {ticker} ({drawdown:.1%} drawdown)"
        body    = (
            f"Portfolio drawdown for {ticker} has exceeded the 15% circuit breaker.\n"
            f"Drawdown: {drawdown:.2%}\n"
            f"No new entries will be placed until equity recovers.\n"
            f"Time: {datetime.now(timezone.utc).isoformat()}\n"
        )
        self._send(subject, body)

    def order_error(self, ticker: str, error: str) -> None:
        """Alert when an order submission fails."""
        subject = f"❌  Order Error: {ticker}"
        body    = (
            f"Failed to submit order for {ticker}.\n"
            f"Error: {error}\n"
            f"Time: {datetime.now(timezone.utc).isoformat()}\n"
        )
        self._send(subject, body)

    def daily_summary(self, signals: list, equity: float) -> None:
        """Send end-of-day summary of all signals and account equity."""
        lines = [f"ML Trade Engine — Daily Summary  ({datetime.now(timezone.utc).date()})\n"]
        lines.append(f"Account equity: ${equity:,.2f}\n")
        lines.append("\nSignals generated today:")
        for s in signals:
            lines.append(
                f"  {s.ticker:<8} {s.label:<5}  conf={s.confidence:.1%}  "
                f"kelly={s.kelly_frac:.3f}  close=${s.close:.2f}"
            )
        body = "\n".join(lines)
        self._send("📊 ML Trade Engine — Daily Summary", body)

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _send(self, subject: str, body: str) -> None:
        self._slack(subject, body)
        self._email(subject, body)

    def _slack(self, subject: str, body: str) -> None:
        if not self.slack_url:
            return
        payload = json.dumps({
            "text": f"*{subject}*\n```{body}```"
        }).encode("utf-8")
        req = urllib.request.Request(
            self.slack_url,
            data    = payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status != 200:
                    log.warning("[Alerts] Slack returned HTTP %d", resp.status)
                else:
                    log.debug("[Alerts] Slack notification sent: %s", subject)
        except Exception as exc:
            log.warning("[Alerts] Slack send failed: %s", exc)

    def _email(self, subject: str, body: str) -> None:
        if not (self.alert_email and self.smtp_user and self.smtp_pass):
            return
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"]    = self.smtp_user
        msg["To"]      = self.alert_email
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.sendmail(self.smtp_user, self.alert_email, msg.as_string())
            log.debug("[Alerts] Email sent to %s: %s", self.alert_email, subject)
        except Exception as exc:
            log.warning("[Alerts] Email send failed: %s", exc)
