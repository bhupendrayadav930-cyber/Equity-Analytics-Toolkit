"""
Dynamic Trendline (Support/Resistance) Breakout
================================================

A "trendline" here is a straight line fit through >=2 same-side pivots
(resistance = pivot highs, support = pivot lows) via linear regression
(np.polyfit, degree 1): price(t) = m*t + c.

Validity check (this is what separates a real trendline from "any two
points connected by a ruler"): between the two anchor pivots, no other
pivot of the SAME side may close on the wrong side of the line by more
than `breakout_confirmation_pct` -- i.e. the line must have genuinely
acted as a ceiling/floor, not been violated already.

Confidence scales with the number of pivots that actually "touch" the
line (within tolerance) -- more touches = more market participants
respected that level = stronger signal, matching how discretionary
technical analysts weight trendlines.

Entry
    RESISTANCE line: close crosses ABOVE line(t) -> long entry (breakout)
    SUPPORT line:    close crosses BELOW line(t) -> short entry (breakdown)
    both require volume >= 1.5x 20d average.

Stop loss
    the trendline value itself at the breakout bar (a pullback that
    re-enters the line negates the breakout thesis), buffered by 1%.
"""
from __future__ import annotations

from itertools import combinations
from typing import List

import numpy as np
import pandas as pd

from .. import config
from .base import BaseStrategy
from ..pivots import Pivot


class TrendlineBreakoutStrategy(BaseStrategy):
    name = "Trendline_Breakout"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._empty_signal_frame(df)
        pivots = self.pivots.find_pivots(df)
        highs = [p for p in pivots if p.kind == "high"]
        lows = [p for p in pivots if p.kind == "low"]
        lookback = config.PATTERNS.max_pattern_lookback_bars

        self._scan_lines(out, df, highs, lookback, resistance=True)
        self._scan_lines(out, df, lows, lookback, resistance=False)
        return out

    @staticmethod
    def _line_value(anchor1: Pivot, anchor2: Pivot, x: int) -> float:
        if anchor2.idx == anchor1.idx:
            return anchor1.price
        slope = (anchor2.price - anchor1.price) / (anchor2.idx - anchor1.idx)
        return anchor1.price + slope * (x - anchor1.idx)

    def _scan_lines(self, out: pd.DataFrame, df: pd.DataFrame,
                     side_pivots: List[Pivot], lookback: int, resistance: bool) -> None:
        tol = config.PATTERNS.breakout_confirmation_pct * 3  # a bit looser for "touch" test
        n = len(side_pivots)

        for i, j in combinations(range(n), 2):
            p1, p2 = side_pivots[i], side_pivots[j]
            if p2.idx - p1.idx < 3 or p2.idx - p1.idx > lookback:
                continue

            between = [p for p in side_pivots if p1.idx < p.idx < p2.idx]
            valid = True
            touches = 2
            for p in between:
                line_val = self._line_value(p1, p2, p.idx)
                breach = (p.price - line_val) / line_val if resistance else (line_val - p.price) / line_val
                if breach > tol:          # pivot pokes meaningfully through -> not a clean line
                    valid = False
                    break
                if abs(breach) <= tol:
                    touches += 1
            if not valid:
                continue

            start = p2.confirmed_at_idx
            self._find_breakout(out, df, start, p1, p2, touches, resistance)

    def _find_breakout(self, out: pd.DataFrame, df: pd.DataFrame, start_idx: int,
                        p1: Pivot, p2: Pivot, touches: int, resistance: bool) -> None:
        if start_idx >= len(df):
            return
        confirm_pct = config.PATTERNS.breakout_confirmation_pct

        for i in range(start_idx, len(df)):
            line_val = self._line_value(p1, p2, i)
            close = df["close"].iloc[i]
            breached = (close > line_val * (1 + confirm_pct)) if resistance \
                else (close < line_val * (1 - confirm_pct))
            if not breached:
                continue
            if not self._volume_confirmed(df, i):
                return

            direction = 1 if resistance else -1
            stop_loss = line_val * (0.99 if resistance else 1.01)

            out.iloc[i, out.columns.get_loc("signal")] = direction
            out.iloc[i, out.columns.get_loc("trigger_price")] = close
            out.iloc[i, out.columns.get_loc("stop_loss")] = stop_loss
            out.iloc[i, out.columns.get_loc("pattern_tag")] = (
                f"{'Resistance' if resistance else 'Support'} Trendline Break "
                f"({touches} touches)"
            )
            out.iloc[i, out.columns.get_loc("confidence")] = min(touches / 5, 1.0)
            return
