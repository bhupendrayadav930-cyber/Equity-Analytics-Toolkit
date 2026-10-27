"""
pivots.py -- the mathematical foundation all Phase-3 patterns are built on.

A "pivot high" at bar i is a local maximum: high[i] > high[i-k..i-1] AND
high[i] > high[i+1..i+k] for a window k ("order"). scipy.signal.argrelextrema
finds these in vectorized C code instead of a Python loop.

CRITICAL -- look-ahead bias: a pivot at bar i is only *knowable* once you've
seen k bars *after* it (you need those future bars to confirm nothing higher
came along). So every pivot carries a `confirmed_at_idx = i + order`: no
strategy is allowed to act on a pivot before that bar. This is the single
most common bug in retail "pattern scanner" code -- it detects the peak at
bar i and pretends you could have traded it at bar i, when in reality you
only find out it WAS the peak `order` bars later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from . import config


@dataclass
class Pivot:
    idx: int                 # integer position in the DataFrame
    confirmed_at_idx: int    # earliest bar this pivot may legally be used
    timestamp: pd.Timestamp
    price: float
    kind: str                # "high" or "low"


class PivotDetector:
    def __init__(self, order: int = None, timeframe: str = "1D"):
        self.order = order or config.PATTERNS.pivot_order.get(timeframe, 5)
        self.timeframe = timeframe

    def find_pivots(self, df: pd.DataFrame) -> List[Pivot]:
        """df must have 'high' and 'low' columns, DatetimeIndex."""
        if len(df) < 2 * self.order + 1:
            return []

        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()

        high_idx = argrelextrema(highs, np.greater_equal, order=self.order)[0]
        low_idx = argrelextrema(lows, np.less_equal, order=self.order)[0]

        # argrelextrema with >=/<= can flag flat plateaus repeatedly;
        # de-duplicate adjacent identical-price runs.
        high_idx = self._dedupe_plateaus(high_idx, highs)
        low_idx = self._dedupe_plateaus(low_idx, lows)

        pivots: List[Pivot] = []
        for i in high_idx:
            pivots.append(Pivot(
                idx=int(i), confirmed_at_idx=int(i) + self.order,
                timestamp=df.index[i], price=float(highs[i]), kind="high",
            ))
        for i in low_idx:
            pivots.append(Pivot(
                idx=int(i), confirmed_at_idx=int(i) + self.order,
                timestamp=df.index[i], price=float(lows[i]), kind="low",
            ))
        pivots.sort(key=lambda p: p.idx)
        return pivots

    @staticmethod
    def _dedupe_plateaus(idx_array: np.ndarray, values: np.ndarray, min_gap: int = 2) -> np.ndarray:
        if len(idx_array) == 0:
            return idx_array
        keep = [idx_array[0]]
        for i in idx_array[1:]:
            if i - keep[-1] >= min_gap:
                keep.append(i)
        return np.array(keep)

    def pivot_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: pivots as a DataFrame aligned to df's index, with
        NaN everywhere except confirmed pivot bars -- handy for plotting
        and for vectorized joins in strategy code.
        """
        pivots = self.find_pivots(df)
        out = pd.DataFrame(index=df.index, columns=["pivot_high", "pivot_low"], dtype=float)
        for p in pivots:
            col = "pivot_high" if p.kind == "high" else "pivot_low"
            out.iloc[p.idx, out.columns.get_loc(col)] = p.price
        return out
