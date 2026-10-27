from .base import BaseStrategy, Signal
from .ema_cross import EMACrossVolumeStrategy
from .head_shoulders import HeadAndShouldersStrategy
from .double_pattern import DoubleTopBottomStrategy
from .trendline import TrendlineBreakoutStrategy
from .triangle import TrianglePatternStrategy
from .flag_cup_handle import BullBearFlagStrategy, CupAndHandleStrategy

ALL_STRATEGIES = [
    HeadAndShouldersStrategy,
    DoubleTopBottomStrategy,
    TrendlineBreakoutStrategy,
    TrianglePatternStrategy,
    BullBearFlagStrategy,
    CupAndHandleStrategy,
    EMACrossVolumeStrategy,
]

__all__ = [
    "BaseStrategy", "Signal", "ALL_STRATEGIES",
    "HeadAndShouldersStrategy", "DoubleTopBottomStrategy",
    "TrendlineBreakoutStrategy", "TrianglePatternStrategy",
    "BullBearFlagStrategy", "CupAndHandleStrategy", "EMACrossVolumeStrategy",
]
