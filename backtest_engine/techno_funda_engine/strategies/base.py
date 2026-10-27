"""
strategies/base.py
-------------------
Every pattern strategy implements ONE method: `generate_signals(df) -> df`
that adds these columns to a copy of the OHLCV DataFrame:

    signal        : +1 (long entry), -1 (short entry / exit long), 0 (flat)
    trigger_price : price level at which the signal is actionable; the
                    backtest engine fills the entry at this level on the
                    signal bar (falling back to that bar's close)
    stop_loss     : initial protective stop for that entry
    pattern_tag   : human-readable label, e.g. "H&S neckline break"
    confidence    : optional pattern-quality score in [0, 1]

Design choice: strategies are STATELESS and PURE -- given the same df they
always produce the same signals. All position sizing, capital allocation,
and trade lifecycle management belongs to backtest.py, not here. This
keeps pattern math testable in isolation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .. import config
from ..pivots import PivotDetector


@dataclass
class Signal:
    timestamp: pd.Timestamp
    direction: int          # +1 long, -1 short
    trigger_price: float
    stop_loss: float
    pattern_tag: str
    confidence: float = 1.0  # optional pattern-quality score, 0-1


class BaseStrategy(ABC):
    name: str = "base"

    def __init__(self, timeframe: str = "1D"):
        self.timeframe = timeframe
        self.pivots = PivotDetector(timeframe=timeframe)

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """df: OHLCV with DatetimeIndex, ascending, columns
        open/high/low/close/volume. Returns df.copy() with the columns
        described above appended (signal defaults to 0 where nothing fires).
        """
        ...

    # ---- shared helpers used by multiple pattern subclasses ---- #
    @staticmethod
    def _volume_confirmed(df: pd.DataFrame, i: int,
                           window: int = None, multiple: float = None) -> bool:
        window = window or config.STRATEGY.volume_avg_window
        multiple = multiple or config.PATTERNS.volume_surge_multiple
        if i < window:
            return False
        avg_vol = df["volume"].iloc[i - window:i].mean()
        return bool(df["volume"].iloc[i] >= multiple * avg_vol)

    @staticmethod
    def _pct_diff(a: float, b: float) -> float:
        return abs(a - b) / b

    @staticmethod
    def _empty_signal_frame(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal"] = 0
        out["trigger_price"] = float("nan")
        out["stop_loss"] = float("nan")
        out["pattern_tag"] = ""
        out["confidence"] = float("nan")
        return out
