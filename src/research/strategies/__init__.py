"""
Phase 06A — Strategy Zoo

All 8 strategies available from a single import.
"""

from src.research.strategies.alpha_trends import AlphaTrendsStrategy
from src.research.strategies.carry_proxy import CarryProxyStrategy
from src.research.strategies.ema_crossover import EMACrossoverStrategy
from src.research.strategies.mean_reversion import MeanReversionStrategy
from src.research.strategies.momentum import MomentumStrategy
from src.research.strategies.pairs_arb import PairsArbStrategy
from src.research.strategies.turtle import TurtleStrategy
from src.research.strategies.vol_breakout import VolBreakoutStrategy

ALL_STRATEGIES = [
    MomentumStrategy(),
    MeanReversionStrategy(),
    EMACrossoverStrategy(),
    TurtleStrategy(),
    PairsArbStrategy(),
    CarryProxyStrategy(),
    VolBreakoutStrategy(),
    AlphaTrendsStrategy(),
]

__all__ = [
    "MomentumStrategy",
    "MeanReversionStrategy",
    "EMACrossoverStrategy",
    "TurtleStrategy",
    "PairsArbStrategy",
    "CarryProxyStrategy",
    "VolBreakoutStrategy",
    "AlphaTrendsStrategy",
    "ALL_STRATEGIES",
]
