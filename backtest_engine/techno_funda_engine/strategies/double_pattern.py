"""
Double Top / Double Bottom
==========================

Definition (Double Top, bearish)
---------------------------------
Two pivot HIGHS, P1 and P2, such that:
    |P1.price - P2.price| / P1.price <= tolerance   (default ±1.5%)
    P2 occurs after P1 with a MINIMUM separation (avoids treating two bars
    of the same spike as "two peaks")
    Between P1 and P2 there is a pivot LOW (the "neckline"/trough) that is
    meaningfully below both peaks (>2% lower, else it's just chop)

Entry (short / exit-long)
    close crosses BELOW the neckline price, with volume >= 1.5x 20d avg,
    AFTER P2 is confirmed (idx >= P2.confirmed_at_idx) -- this is the
    look-ahead guard.

Stop loss
    max(P1.price, P2.price) * 1.01 -- i.e. structure failure: if price
    re-takes the twin peaks, the "double top" thesis is invalidated.

Double Bottom is the exact mirror using pivot LOWS and a rally-neckline
in between, entry on an upside breakout.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .. import config
from .base import BaseStrategy
from ..pivots import Pivot


class DoubleTopBottomStrategy(BaseStrategy):
    name = "Double_Top_Bottom"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._empty_signal_frame(df)
        pivots = self.pivots.find_pivots(df)
        if len(pivots) < 3:
            return out

        highs = [p for p in pivots if p.kind == "high"]
        lows = [p for p in pivots if p.kind == "low"]
        tol = config.PATTERNS.double_pattern_tolerance
        min_gap_bars = max(self.pivots.order * 2, 5)

        self._scan_side(out, df, highs, lows, tol, min_gap_bars, bearish=True)
        self._scan_side(out, df, lows, highs, tol, min_gap_bars, bearish=False)
        return out

    def _scan_side(self, out: pd.DataFrame, df: pd.DataFrame,
                    same_side: List[Pivot], opp_side: List[Pivot],
                    tol: float, min_gap_bars: int, bearish: bool) -> None:
        for i in range(len(same_side) - 1):
            p1, p2 = same_side[i], same_side[i + 1]
            if p2.idx - p1.idx < min_gap_bars:
                continue
            if self._pct_diff(p1.price, p2.price) > tol:
                continue

            trough_candidates = [p for p in opp_side if p1.idx < p.idx < p2.idx]
            if not trough_candidates:
                continue
            neckline_pivot = (min(trough_candidates, key=lambda p: p.price) if bearish
                               else max(trough_candidates, key=lambda p: p.price))
            depth = self._pct_diff(neckline_pivot.price, p1.price)
            if depth < 0.02:   # not a meaningful trough -> not a real double top/bottom
                continue

            start_search = max(p2.confirmed_at_idx, neckline_pivot.confirmed_at_idx)
            self._find_breakout_and_mark(out, df, start_search, neckline_pivot.price,
                                          stop_ref=max(p1.price, p2.price) if bearish
                                          else min(p1.price, p2.price),
                                          bearish=bearish,
                                          tag="Double Top" if bearish else "Double Bottom")

    def _find_breakout_and_mark(self, out: pd.DataFrame, df: pd.DataFrame,
                                 start_idx: int, neckline: float, stop_ref: float,
                                 bearish: bool, tag: str) -> None:
        if start_idx >= len(df):
            return
        window = df.iloc[start_idx:]
        confirm_pct = config.PATTERNS.breakout_confirmation_pct

        if bearish:
            breached = window["close"] < neckline * (1 - confirm_pct)
        else:
            breached = window["close"] > neckline * (1 + confirm_pct)

        if not breached.any():
            return
        breakout_pos_in_window = breached.idxmax()   # first True label
        i = df.index.get_loc(breakout_pos_in_window)

        if not self._volume_confirmed(df, i):
            return  # pattern found, but no volume conviction -> skip (still a real trade rule)

        direction = -1 if bearish else 1
        stop_loss = stop_ref * (1.01 if bearish else 0.99)

        out.iloc[i, out.columns.get_loc("signal")] = direction
        out.iloc[i, out.columns.get_loc("trigger_price")] = df["close"].iloc[i]
        out.iloc[i, out.columns.get_loc("stop_loss")] = stop_loss
        out.iloc[i, out.columns.get_loc("pattern_tag")] = tag
        out.iloc[i, out.columns.get_loc("confidence")] = 1 - self._pct_diff(
            neckline, stop_ref
        ) if stop_ref else np.nan
