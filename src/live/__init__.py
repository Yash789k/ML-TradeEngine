"""
Phase 06E — Live Signal Engine

Exposes:
  SignalGenerator  — fetch → features → model inference → signal for today
  AlpacaBroker     — paper trading via Alpaca API (bracket orders)
  SignalLogger     — SQLite-backed signal + order log
  Alerts           — Slack webhook + email alerts
  LiveEngine       — orchestrates all four components end-to-end
"""

from src.live.signal_generator import LiveSignal, SignalGenerator  # noqa: F401
from src.live.broker import AlpacaBroker                            # noqa: F401
from src.live.logger import SignalLogger                            # noqa: F401
from src.live.alerts import Alerts                                  # noqa: F401
from src.live.engine import LiveEngine                              # noqa: F401
