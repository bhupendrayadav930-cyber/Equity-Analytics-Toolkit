"""
Triangle Patterns: Ascending / Descending / Symmetrical
========================================================

Method
------
Within a rolling lookback window (config.max_pattern_lookback_bars),
fit TWO regression lines via np.polyfit(idx, price, deg=1):
    upper line  through the window's pivot HIGHS
    lower line  through the window's pivot LOWS
each needs >= 2 pivots (ideally triangle_min_touches combined) to fit.

Normalise both slopes to "% of price per bar" (slope / mean_price) so the
classification thresholds are scale-invariant (a ₹50 stock and a ₹5,000
stock with the same % slope are treated the same):

    flat_threshold = 0.0005 (0.05%/bar) -- below this, a line counts as "flat"

    Ascending Triangle   : |upper_slope_pct| < flat_threshold   (flat resistance)
                            AND lower_slope_pct > flat_threshold (rising support)
    Descending Triangle  : |lower_slope_pct| < flat_threshold   (flat support)
                            AND upper_slope_pct < -flat_threshold (falling resistance)
    Symmetrical Triangle : upper_slope_pct < -flat_threshold
                            AND lower_slope_pct > flat_threshold  (converging)

Convergence check: the two lines must actually be narrowing --
upper_slope_pct <= lower_slope_pct (gap shrinking, not widening/parallel)
otherwise it's a channel, not a triangle -- and reject it.

Entry
    Ascending  -> bias long: breakout ABOVE the flat upper line + volume
    Descending -> bias short: breakdown BELOW the flat lower line + volume
    Symmetrical-> whichever line breaks first, in either direction + volume

Stop loss
    the OPPOSITE trendline's value at the breakout bar (a classic
    "structure failure" stop -- if price whipsaws back through the far
    side of the triangle, the breakout has failed).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import config
from .base import BaseStrategy
from ..pivots import Pivot

FLAT_THRESHOLD = 0.0005


class TrianglePatternStrategy(BaseStrategy):
    name = "Triangle_Pattern"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._empty_signal_frame(df)
        pivots = self.pivots.find_pivots(df)
        highs = [p for p in pivots if p.kind == "high"]
        lows = [p for p in pivots if p.kind == "low"]
        lookback = config.PATTERNS.max_pattern_lookback_bars
        min_touches = config.PATTERNS.triangle_min_touches

        if len(highs) < 2 or len(lows) < 2:
            return out

        # Slide a window anchored at each low-side pivot; look at all highs/
        # lows within [anchor.idx - lookback, anchor.idx] to fit both lines.
        checked_windows = set()
        for anchor in sorted(highs + lows, key=lambda p: p.idx):
            win_start = anchor.idx - lookback
            key = anchor.idx // 10   # coarse de-dup so we don't refit every single bar
            if key in checked_windows:
                continue
            checked_windows.add(key)

            win_highs = [p for p in highs if win_start <= p.idx <= anchor.idx]
            win_lows = [p for p in lows if win_start <= p.idx <= anchor.idx]
            if len(win_highs) < 2 or len(win_lows) < 2 or (len(win_highs) + len(win_lows)) < min_touches:
                continue

            fit = self._classify(df, win_highs, win_lows)
            if fit is None:
                continue
            kind, upper_fit, lower_fit = fit
            last_pivot_idx = max(win_highs[-1].idx, win_lows[-1].idx)
            start_search = last_pivot_idx + self.pivots.order  # confirmation lag
            self._find_breakout(out, df, start_search, upper_fit, lower_fit, kind)

        return out

    # ------------------------------------------------------------------ #
    def _classify(self, df: pd.DataFrame, win_highs: List[Pivot], win_lows: List[Pivot]
                   ) -> Optional[Tuple[str, tuple, tuple]]:
        avg_price = df["close"].mean()

        upper_slope, upper_c = np.polyfit([p.idx for p in win_highs],
                                           [p.price for p in win_highs], 1)
        lower_slope, lower_c = np.polyfit([p.idx for p in win_lows],
                                           [p.price for p in win_lows], 1)
        upper_pct = upper_slope / avg_price
        lower_pct = lower_slope / avg_price

        if upper_pct > lower_pct - FLAT_THRESHOLD / 2:
            return None  # not converging -> parallel channel or diverging, reject

        if abs(upper_pct) < FLAT_THRESHOLD and lower_pct > FLAT_THRESHOLD:
            kind = "Ascending Triangle"
        elif abs(lower_pct) < FLAT_THRESHOLD and upper_pct < -FLAT_THRESHOLD:
            kind = "Descending Triangle"
        elif upper_pct < -FLAT_THRESHOLD and lower_pct > FLAT_THRESHOLD:
            kind = "Symmetrical Triangle"
        else:
            return None

        return kind, (upper_slope, upper_c), (lower_slope, lower_c)

    def _find_breakout(self, out: pd.DataFrame, df: pd.DataFrame, start_idx: int,
                        upper_fit: tuple, lower_fit: tuple, kind: str) -> None:
        if start_idx >= len(df):
            return
        confirm_pct = config.PATTERNS.breakout_confirmation_pct
        allow_long = kind in ("Ascending Triangle", "Symmetrical Triangle")
        allow_short = kind in ("Descending Triangle", "Symmetrical Triangle")

        for i in range(start_idx, len(df)):
            upper_val = upper_fit[0] * i + upper_fit[1]
            lower_val = lower_fit[0] * i + lower_fit[1]
            close = df["close"].iloc[i]

            broke_up = allow_long and close > upper_val * (1 + confirm_pct)
            broke_down = allow_short and close < lower_val * (1 - confirm_pct)
            if not (broke_up or broke_down):
                continue
            if not self._volume_confirmed(df, i):
                return

            direction = 1 if broke_up else -1
            stop_loss = lower_val * 0.99 if broke_up else upper_val * 1.01

            out.iloc[i, out.columns.get_loc("signal")] = direction
            out.iloc[i, out.columns.get_loc("trigger_price")] = close
            out.iloc[i, out.columns.get_loc("stop_loss")] = stop_loss
            out.iloc[i, out.columns.get_loc("pattern_tag")] = (
                f"{kind} {'Breakout' if broke_up else 'Breakdown'}"
            )
            width_pct = abs(upper_val - lower_val) / close
            out.iloc[i, out.columns.get_loc("confidence")] = max(0.0, 1 - width_pct * 5)
            return
