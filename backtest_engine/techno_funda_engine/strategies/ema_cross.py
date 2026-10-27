"""
EMA 50/200 Golden/Death Cross + Volume confirmation -- the baseline
momentum strategy every pattern strategy is benchmarked against.

Math
----
EMA_fast = EWMA(close, span=50), EMA_slow = EWMA(close, span=200)
Golden Cross (long entry)  : EMA_fast crosses ABOVE EMA_slow
                             AND volume[t] >= 1.5 * SMA(volume, 20)[t]
Death Cross  (exit / short): EMA_fast crosses BELOW EMA_slow
Exit (long)                : trailing stop = EMA_20, i.e. close < EMA_20
                             after entry, OR a Death Cross, whichever first.

Fully vectorized -- no per-bar Python loop needed for signal generation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .base import BaseStrategy


class EMACrossVolumeStrategy(BaseStrategy):
    name = "EMA_50_200_Cross_Volume"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._empty_signal_frame(df)
        if len(out) < config.STRATEGY.ema_slow + 5:
            return out

        ema_fast = out["close"].ewm(span=config.STRATEGY.ema_fast, adjust=False).mean()
        ema_slow = out["close"].ewm(span=config.STRATEGY.ema_slow, adjust=False).mean()
        ema_trail = out["close"].ewm(span=config.STRATEGY.trailing_stop_ema, adjust=False).mean()
        vol_avg = out["volume"].rolling(config.STRATEGY.volume_avg_window).mean()

        golden_cross = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        death_cross = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
        vol_confirmed = out["volume"] >= config.PATTERNS.volume_surge_multiple * vol_avg

        long_entry = golden_cross & vol_confirmed
        exit_signal = death_cross | (out["close"] < ema_trail)

        out["signal"] = np.where(long_entry, 1, np.where(exit_signal, -1, 0))
        out["trigger_price"] = np.where(long_entry, out["close"], np.nan)
        out["stop_loss"] = np.where(long_entry, ema_trail, np.nan)
        out["pattern_tag"] = np.where(
            long_entry, "Golden Cross + Vol",
            np.where(exit_signal, "Death Cross / EMA20 fail", ""),
        )
        out["confidence"] = np.where(
            long_entry, (out["volume"] / vol_avg).clip(upper=3) / 3, np.nan
        )
        return out
