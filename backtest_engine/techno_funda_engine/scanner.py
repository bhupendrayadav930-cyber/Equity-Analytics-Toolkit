"""
scanner.py -- PHASE 4 (part 3): "what's actionable RIGHT NOW"

Re-runs every strategy's generate_signals() on the latest data (already
computed when reusing the same signal frames from the backtest run) and
surfaces any bar within the last `recency_bars` bars that fired a signal.
The output is a watchlist for manual chart review -- signals are never
auto-executed.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .logging_utils import get_logger
from .strategies.base import BaseStrategy

logger = get_logger(__name__)


class LiveSetupScanner:
    def __init__(self, strategies: List[BaseStrategy], recency_bars: int = 2):
        self.strategies = strategies
        self.recency_bars = recency_bars

    def scan(self, timeframe: str, symbol_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        rows = []
        for strategy in self.strategies:
            for symbol, df in symbol_dfs.items():
                if df is None or df.empty or len(df) < 30:
                    continue
                try:
                    sig_df = strategy.generate_signals(df)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Scanner: %s failed on %s/%s: %s",
                                 strategy.name, symbol, timeframe, exc)
                    continue

                recent = sig_df.tail(self.recency_bars)
                fired = recent[recent["signal"] != 0]
                for ts, row in fired.iterrows():
                    bars_ago = len(sig_df.loc[ts:]) - 1
                    rows.append({
                        "symbol": symbol, "strategy": strategy.name, "timeframe": timeframe,
                        "timestamp": ts, "direction": "LONG" if row["signal"] == 1 else "SHORT/EXIT",
                        "trigger_price": row["trigger_price"], "stop_loss": row["stop_loss"],
                        "pattern_tag": row["pattern_tag"], "confidence": row["confidence"],
                        "bars_ago": bars_ago,
                    })

        out = pd.DataFrame(rows)
        if out.empty:
            logger.info("Live scan (%s): no active setups found.", timeframe)
            return out
        return out.sort_values(["bars_ago", "confidence"], ascending=[True, False]).reset_index(drop=True)

    def scan_all_timeframes(self, tf_symbol_dfs: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        frames = [self.scan(tf, symbol_dfs) for tf, symbol_dfs in tf_symbol_dfs.items()]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
