"""
Head & Shoulders (bearish) / Inverse Head & Shoulders (bullish)
=================================================================

Definition (H&S)
-----------------
Three consecutive pivot HIGHS: Left Shoulder (LS), Head (H), Right
Shoulder (RS), with two pivot LOWS in between (Trough1 between LS-H,
Trough2 between H-RS), such that:

    H.price > LS.price  AND  H.price > RS.price          (head is tallest)
    |LS.price - RS.price| / LS.price <= symmetry_tol      (shoulders ~equal, default ±8%)
    Trough1, Trough2 roughly comparable in depth (both meaningfully below
    the shoulders -- otherwise it's not a clean "M-with-a-spike" shape)

Neckline: the line joining Trough1 and Trough2 (NOT assumed horizontal --
real necklines slope). We fit it as a straight line through the two
trough points: price(t) = m*t + c, and evaluate it AT the breakout bar's
x-position, so a downward-sloping neckline is correctly a lower trigger
than a flat one.

Entry (short)
    close crosses below neckline(t), t >= RS.confirmed_at_idx, with volume
    confirmation.

Stop loss
    RS.price * 1.01 (a re-break above the right shoulder invalidates the
    pattern).

Target (informational, used by confidence score only -- exits are handled
by the backtest engine's trailing-stop / structure-failure rule)
    measured move = neckline - (H.price - neckline), i.e. project the
    head-to-neckline distance downward from the breakout point.

Inverse H&S mirrors all of this with pivot LOWS for LS/H/RS and pivot
HIGHS for the neckline, breaking out upward.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .. import config
from .base import BaseStrategy
from ..pivots import Pivot


class HeadAndShouldersStrategy(BaseStrategy):
    name = "Head_And_Shoulders"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._empty_signal_frame(df)
        pivots = self.pivots.find_pivots(df)
        if len(pivots) < 5:
            return out

        highs = [p for p in pivots if p.kind == "high"]
        lows = [p for p in pivots if p.kind == "low"]

        self._scan(out, df, highs, lows, bearish=True)   # classic H&S
        self._scan(out, df, lows, highs, bearish=False)  # inverse H&S
        return out

    def _scan(self, out: pd.DataFrame, df: pd.DataFrame,
              shoulder_side: List[Pivot], neck_side: List[Pivot], bearish: bool) -> None:
        tol = config.PATTERNS.shoulder_symmetry_tolerance

        for i in range(len(shoulder_side) - 2):
            ls, h, rs = shoulder_side[i], shoulder_side[i + 1], shoulder_side[i + 2]

            head_is_extreme = (h.price > ls.price and h.price > rs.price) if bearish \
                else (h.price < ls.price and h.price < rs.price)
            if not head_is_extreme:
                continue
            if self._pct_diff(ls.price, rs.price) > tol:
                continue

            t1_candidates = [p for p in neck_side if ls.idx < p.idx < h.idx]
            t2_candidates = [p for p in neck_side if h.idx < p.idx < rs.idx]
            if not t1_candidates or not t2_candidates:
                continue
            t1 = (min(t1_candidates, key=lambda p: p.price) if bearish
                  else max(t1_candidates, key=lambda p: p.price))
            t2 = (min(t2_candidates, key=lambda p: p.price) if bearish
                  else max(t2_candidates, key=lambda p: p.price))

            start_search = max(rs.confirmed_at_idx, t2.confirmed_at_idx)
            self._find_breakout(out, df, start_search, t1, t2, head=h, right_shoulder=rs,
                                 bearish=bearish)

    def _neckline_value_at(self, t1: Pivot, t2: Pivot, x: int) -> float:
        if t2.idx == t1.idx:
            return t1.price
        slope = (t2.price - t1.price) / (t2.idx - t1.idx)
        return t1.price + slope * (x - t1.idx)

    def _find_breakout(self, out: pd.DataFrame, df: pd.DataFrame, start_idx: int,
                        t1: Pivot, t2: Pivot, head: Pivot, right_shoulder: Pivot,
                        bearish: bool) -> None:
        if start_idx >= len(df):
            return
        confirm_pct = config.PATTERNS.breakout_confirmation_pct

        for i in range(start_idx, len(df)):
            neckline_here = self._neckline_value_at(t1, t2, i)
            close = df["close"].iloc[i]

            breached = (close < neckline_here * (1 - confirm_pct)) if bearish \
                else (close > neckline_here * (1 + confirm_pct))
            if not breached:
                continue
            if not self._volume_confirmed(df, i):
                return  # first touch had no volume -- don't keep chasing later touches

            direction = -1 if bearish else 1
            stop_loss = right_shoulder.price * (1.01 if bearish else 0.99)
            measured_move = abs(head.price - neckline_here)
            target = neckline_here - measured_move if bearish else neckline_here + measured_move

            out.iloc[i, out.columns.get_loc("signal")] = direction
            out.iloc[i, out.columns.get_loc("trigger_price")] = close
            out.iloc[i, out.columns.get_loc("stop_loss")] = stop_loss
            out.iloc[i, out.columns.get_loc("pattern_tag")] = (
                "Head & Shoulders" if bearish else "Inverse Head & Shoulders"
            )
            risk = abs(close - stop_loss)
            reward = abs(target - close)
            out.iloc[i, out.columns.get_loc("confidence")] = (
                min(reward / risk, 3) / 3 if risk > 0 else np.nan
            )
            return  # one signal per pattern instance
