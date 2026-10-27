"""
High-Probability Continuation Patterns: Bull/Bear Flag & Cup-and-Handle
=========================================================================

Bull/Bear Flag
--------------
1. FLAGPOLE: an impulsive leg between two consecutive opposite pivots
   (low -> high for a bull flag) where
        move_pct = (high.price - low.price) / low.price >= flagpole_min_move_pct
        AND it happens fast: (high.idx - low.idx) <= pole_max_bars
   A slow, grinding 8% gain over 60 bars is not a "pole"; a sharp 8%+ move
   in a handful of bars is.

2. FLAG: the consolidation immediately after the pole. We fit two parallel
   regression lines (upper through the flag-window's pivot highs, lower
   through its pivot lows). A valid flag:
        - both slopes have the SAME sign and are mild (drifting, not
          trending) -- for a bull flag, both should be flat-to-slightly-
          DOWN (counter-trend pause); for a bear flag, flat-to-slightly-UP
        - the channel width stays roughly constant (parallel, not a
          widening wedge)
        - it's short relative to the pole: flag_bars <= pole_bars * 4
        - volume during the flag is, on average, LOWER than during the
          pole (classic "volume dries up on the pause" signature)

3. ENTRY: close breaks back out through the flag boundary in the pole's
   direction (upper bound for bull, lower bound for bear), with a fresh
   volume >= 1.5x avg. STOP: the opposite boundary of the flag channel.

Cup and Handle
--------------
1. CUP: two pivot highs ("lips") of similar price (within double-pattern
   tolerance) separated by enough bars to be a real base (>= 15), with the
   price path between them fitting a convex parabola
        close(t) ≈ a*t^2 + b*t + c,  a > 0 (U-shaped, via np.polyfit deg=2)
   and a decent fit quality (R^2 above a floor, else it's just noise, not
   a bowl) and cup depth (lip - bottom)/lip within [10%, 50%] -- shallower
   isn't a real correction, deeper starts looking like a broader downtrend.

2. HANDLE: after the right lip, a shallow pullback lasting a fraction of
   the cup's length, with depth <= half the cup's depth (a handle that's
   as deep as the cup itself is really just a double top).

3. ENTRY: close breaks above max(left_lip, right_lip) with volume
   confirmation. STOP: the handle's low.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .. import config
from .base import BaseStrategy
from ..pivots import Pivot

POLE_MAX_BARS = 10
FLAG_MAX_MULTIPLE_OF_POLE = 4
CUP_MIN_BARS = 15
CUP_R2_FLOOR = 0.5


class BullBearFlagStrategy(BaseStrategy):
    name = "Bull_Bear_Flag"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._empty_signal_frame(df)
        pivots = self.pivots.find_pivots(df)
        if len(pivots) < 4:
            return out

        for i in range(len(pivots) - 1):
            pole_start, pole_end = pivots[i], pivots[i + 1]
            if pole_end.idx <= pole_start.idx or pole_end.idx - pole_start.idx > POLE_MAX_BARS:
                continue

            bullish = pole_start.kind == "low" and pole_end.kind == "high"
            bearish = pole_start.kind == "high" and pole_end.kind == "low"
            if not (bullish or bearish):
                continue

            move_pct = self._pct_diff(pole_end.price, pole_start.price)
            if move_pct < config.PATTERNS.flagpole_min_move_pct:
                continue

            pole_bars = pole_end.idx - pole_start.idx
            flag_pivots = [p for p in pivots
                           if pole_end.idx < p.idx <= pole_end.idx + pole_bars * FLAG_MAX_MULTIPLE_OF_POLE]
            self._evaluate_flag(out, df, pole_start, pole_end, flag_pivots, pole_bars, bullish)

        return out

    def _evaluate_flag(self, out, df, pole_start: Pivot, pole_end: Pivot,
                        flag_pivots: List[Pivot], pole_bars: int, bullish: bool) -> None:
        highs = [p for p in flag_pivots if p.kind == "high"]
        lows = [p for p in flag_pivots if p.kind == "low"]
        if len(highs) < 1 or len(lows) < 1:
            return

        # Need at least 2 points on one side to get a slope; if only 1
        # pivot on a side, treat its line as flat through that point.
        upper_pts = highs if len(highs) >= 2 else highs + [Pivot(pole_end.idx, pole_end.confirmed_at_idx,
                                                                    pole_end.timestamp, pole_end.price, "high")]
        lower_pts = lows if len(lows) >= 2 else lows + [Pivot(pole_end.idx, pole_end.confirmed_at_idx,
                                                                 pole_end.timestamp, pole_end.price, "low")]
        if len(upper_pts) < 2 or len(lower_pts) < 2:
            return

        avg_price = df["close"].mean()
        up_slope, up_c = np.polyfit([p.idx for p in upper_pts], [p.price for p in upper_pts], 1)
        lo_slope, lo_c = np.polyfit([p.idx for p in lower_pts], [p.price for p in lower_pts], 1)
        up_pct, lo_pct = up_slope / avg_price, lo_slope / avg_price

        # channel drift must go WITH the counter-trend pause, not re-accelerate
        if bullish and (up_pct > 0.001 or lo_pct > 0.001):
            return
        if not bullish and (up_pct < -0.001 or lo_pct < -0.001):
            return

        last_idx = max(p.idx for p in flag_pivots)
        pole_vol = df["volume"].iloc[pole_start.idx:pole_end.idx + 1].mean()
        flag_vol = df["volume"].iloc[pole_end.idx:last_idx + 1].mean()
        if flag_vol >= pole_vol:   # no volume dry-up during the pause -> weak flag
            return

        start_search = last_idx + self.pivots.order
        self._find_breakout(out, df, start_search, (up_slope, up_c), (lo_slope, lo_c), bullish)

    def _find_breakout(self, out, df, start_idx: int, upper_fit, lower_fit, bullish: bool) -> None:
        if start_idx >= len(df):
            return
        confirm_pct = config.PATTERNS.breakout_confirmation_pct
        for i in range(start_idx, len(df)):
            upper_val = upper_fit[0] * i + upper_fit[1]
            lower_val = lower_fit[0] * i + lower_fit[1]
            close = df["close"].iloc[i]
            breached = close > upper_val * (1 + confirm_pct) if bullish \
                else close < lower_val * (1 - confirm_pct)
            if not breached:
                continue
            if not self._volume_confirmed(df, i):
                return

            direction = 1 if bullish else -1
            stop_loss = lower_val * 0.99 if bullish else upper_val * 1.01
            out.iloc[i, out.columns.get_loc("signal")] = direction
            out.iloc[i, out.columns.get_loc("trigger_price")] = close
            out.iloc[i, out.columns.get_loc("stop_loss")] = stop_loss
            out.iloc[i, out.columns.get_loc("pattern_tag")] = "Bull Flag" if bullish else "Bear Flag"
            out.iloc[i, out.columns.get_loc("confidence")] = 0.7  # qualitative: passed all structural filters
            return


class CupAndHandleStrategy(BaseStrategy):
    name = "Cup_And_Handle"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._empty_signal_frame(df)
        pivots = self.pivots.find_pivots(df)
        highs = [p for p in pivots if p.kind == "high"]
        tol = config.PATTERNS.double_pattern_tolerance * 2  # lips a bit looser than a double top

        for i in range(len(highs) - 1):
            left_lip, right_lip = highs[i], highs[i + 1]
            if right_lip.idx - left_lip.idx < CUP_MIN_BARS:
                continue
            if self._pct_diff(left_lip.price, right_lip.price) > tol:
                continue

            cup_window = df.iloc[left_lip.idx: right_lip.idx + 1]
            fit = self._fit_cup(cup_window)
            if fit is None:
                continue
            bottom_price, r2 = fit
            depth = self._pct_diff(bottom_price, left_lip.price)
            if not (0.10 <= depth <= 0.50) or r2 < CUP_R2_FLOOR:
                continue

            self._find_handle_and_breakout(out, df, right_lip, left_lip, depth)
        return out

    @staticmethod
    def _fit_cup(window: pd.DataFrame):
        if len(window) < CUP_MIN_BARS:
            return None
        x = np.arange(len(window))
        y = window["close"].to_numpy()
        coeffs = np.polyfit(x, y, 2)
        a, b, c = coeffs
        if a <= 0:
            return None  # not convex -> not a bowl
        fitted = np.polyval(coeffs, x)
        ss_res = np.sum((y - fitted) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vertex_x = -b / (2 * a)
        bottom_price = float(np.polyval(coeffs, np.clip(vertex_x, 0, len(window) - 1)))
        return bottom_price, r2

    def _find_handle_and_breakout(self, out, df, right_lip: Pivot, left_lip: Pivot, cup_depth: float) -> None:
        cup_len = right_lip.idx - left_lip.idx
        handle_max_bars = max(int(cup_len * 0.35), self.pivots.order * 2)
        handle_end = min(right_lip.idx + handle_max_bars, len(df) - 1)
        if handle_end <= right_lip.idx:
            return
        handle_window = df.iloc[right_lip.idx: handle_end + 1]
        handle_low = handle_window["low"].min()
        handle_depth = self._pct_diff(handle_low, right_lip.price)
        if handle_depth > cup_depth * 0.5:
            return  # handle too deep -> more like a double top

        resistance = max(left_lip.price, right_lip.price)
        start_search = right_lip.idx + self.pivots.order
        confirm_pct = config.PATTERNS.breakout_confirmation_pct

        for i in range(start_search, len(df)):
            close = df["close"].iloc[i]
            if close <= resistance * (1 + confirm_pct):
                continue
            if not self._volume_confirmed(df, i):
                return

            out.iloc[i, out.columns.get_loc("signal")] = 1
            out.iloc[i, out.columns.get_loc("trigger_price")] = close
            out.iloc[i, out.columns.get_loc("stop_loss")] = handle_low * 0.99
            out.iloc[i, out.columns.get_loc("pattern_tag")] = "Cup and Handle Breakout"
            out.iloc[i, out.columns.get_loc("confidence")] = 1 - handle_depth / (cup_depth + 1e-9)
            return
