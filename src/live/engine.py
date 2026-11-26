"""Daily local paper execution using the same target account as research replay.

External brokers and alert transports are deliberately not invoked. Authorizing a
prospective broker trial is a separate operation after adapter reconciliation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import pandas as pd
from src.live.signal_generator import SignalGenerator
from src.live.logger import SignalLogger
from src.live.broker import AlpacaBroker  # legacy adapter; never instantiated here
from src.live.alerts import Alerts  # opt-in transport; never instantiated here
from src.execution.account import Account, Limits, ExecutionSession


@dataclass
class RunResult:
    signals: dict = field(default_factory=dict)
    orders: dict = field(default_factory=dict)
    skipped: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    account_equity: Optional[float] = None
    targets: dict = field(default_factory=dict)
    snapshot: dict = field(default_factory=dict)


class LiveEngine:
    def __init__(
        self,
        tickers=None,
        confidence_threshold=0.38,
        max_portfolio_heat=0.8,
        max_position_frac=0.05,
        circuit_breaker_dd=0.15,
        atr_tp_multiplier=3.0,
        dry_run=True,
        use_policy=False,
        strategy_mode="ensemble",
        initial_capital=10000.0,
        state_path=None,
    ):
        if strategy_mode not in ("ensemble", "meta"):
            raise ValueError("strategy_mode must be ensemble or meta")
        self.tickers = tickers or ["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]
        self.confidence_threshold = confidence_threshold
        self.max_position_frac = max_position_frac
        self.max_portfolio_heat = max_portfolio_heat
        self.dry_run = dry_run
        self.use_policy = use_policy
        self.strategy_mode = strategy_mode
        self.initial_capital = initial_capital
        self.limits = Limits(
            max_gross=max_portfolio_heat,
            max_position=max_position_frac,
            drawdown_limit=circuit_breaker_dd,
        )
        self.state_path = (
            Path(state_path)
            if state_path
            else Path(__file__).resolve().parents[2] / "data/local-paper/account.sqlite"
        )

    def target_weights(self, signals):
        weights = {}
        for ticker, signal in signals.items():
            frac = (
                min(1 / len(signals), self.max_position_frac)
                if signal.strategy_mode == "meta"
                else min(max(signal.kelly_frac, 0.0), self.max_position_frac)
            )
            weights[ticker] = (
                frac * signal.size_cap
                if signal.direction == 2 and signal.eligible
                else 0.0
            )
        if sum(weights.values()) > self.max_portfolio_heat:
            scale = self.max_portfolio_heat / sum(weights.values())
            weights = {s: w * scale for s, w in weights.items()}
        return weights

    def run(self, tickers=None, force_refresh=True):
        ticker_list = tickers or self.tickers
        result = RunResult()
        from src.risk.policy_engine import PolicyEngine

        policy = (
            PolicyEngine(base_confidence=self.confidence_threshold)
            if self.use_policy
            else None
        )
        if self.strategy_mode == "meta":
            from src.live.meta_signal_generator import MetaSignalGenerator

            gen = MetaSignalGenerator(policy_engine=policy)
        else:
            gen = SignalGenerator(
                confidence_threshold=self.confidence_threshold, policy_engine=policy
            )
        signals = gen.generate_batch(ticker_list, force_refresh=force_refresh)
        result.signals = signals
        for ticker in set(ticker_list) - set(signals):
            errors = getattr(gen, "errors", {})
            message = (
                errors.get(ticker, "No usable signal")
                if isinstance(errors, dict)
                else "No usable signal"
            )
            result.errors.append((ticker, message))
        # Batch atomicity: do not advance an account with missing instruments.
        if result.errors or not signals:
            return result
        dates = {pd.Timestamp(s.date).isoformat() for s in signals.values()}
        if len(dates) != 1:
            result.errors.append(
                ("batch", "Signals refer to different information cutoffs")
            )
            return result
        result.targets = self.target_weights(signals)
        prices = {t: s.close for t, s in signals.items()}
        if self.dry_run:
            result.orders = {t: None for t in signals}
            return result
        try:
            account = Account(
                ticker_list, self.initial_capital, self.limits, self.state_path
            )
            before = set(account.orders)
            result.snapshot = ExecutionSession(account).on_close(
                next(iter(dates)), prices, result.targets
            )
            result.account_equity = result.snapshot["equity"]
            result.orders = {
                oid: account.orders[oid] for oid in set(account.orders) - before
            }
        except Exception as exc:
            result.errors.append(("execution", str(exc)))
        return result

    def _compute_qty(self, signal, equity, broker, n_tickers=1):
        """Legacy sizing query only. Actual orders use target deltas and reservations."""
        if equity <= 0 or signal.close <= 0:
            return 0.0
        frac = (
            min(1 / max(n_tickers, 1), self.max_position_frac)
            if signal.strategy_mode == "meta"
            else min(signal.kelly_frac, self.max_position_frac)
        )
        return round(
            max(
                0,
                min(
                    equity * frac * signal.size_cap,
                    broker.buying_power() * 0.95 if broker is not None else equity,
                )
                / signal.close,
            ),
            4,
        )
