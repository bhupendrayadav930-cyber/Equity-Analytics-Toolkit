"""
main.py -- orchestrates all four phases end-to-end.

Run:
    python -m techno_funda_engine.main --csv data/ind_nifty500list.csv --limit 15

The default configuration uses YFinanceDataProvider (no credentials needed).
To use broker data instead, construct a DhanDataProvider / UpstoxDataProvider
(credentials are read from environment variables) and pass it to
MultiTimeframeDataManager.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import pandas as pd

from . import config
from .analytics import ComparisonMatrixBuilder, PerformanceAnalytics
from .backtest import BacktestEngine
from .data_providers import MultiTimeframeDataManager, YFinanceDataProvider
from .logging_utils import get_logger
from .scanner import LiveSetupScanner
from .strategies import ALL_STRATEGIES
from .universe import YFinanceFundamentalProvider, build_cleared_universe

logger = get_logger(__name__)


def run_pipeline(csv_path: str, limit: int = 15, lookback_days: int = 720) -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ---------------- Phase 1 ---------------- #
    logger.info("=== PHASE 1: Universe + Fundamental Filter ===")
    cleared = build_cleared_universe(csv_path, provider=YFinanceFundamentalProvider(), limit=limit)
    if cleared.empty:
        logger.error("No stocks passed the fundamental filter -- relaxing criteria or "
                      "checking data source is recommended. Aborting.")
        return
    symbols = cleared["yf_symbol"].tolist()
    logger.info("Cleared universe (%d): %s", len(symbols), symbols)

    # ---------------- Phase 2 ---------------- #
    logger.info("=== PHASE 2: Multi-timeframe data fetch ===")
    provider = YFinanceDataProvider(interval=config.BASE_INTERVAL)
    manager = MultiTimeframeDataManager(provider)
    end = datetime.now()
    start = end - timedelta(days=lookback_days)

    per_symbol_tf = manager.get_universe_timeframes(symbols, start, end)

    # reshape to {timeframe: {symbol: df}}
    tf_symbol_dfs = {tf: {} for tf in config.ALL_TIMEFRAMES}
    for sym, tf_dict in per_symbol_tf.items():
        for tf, df in tf_dict.items():
            tf_symbol_dfs[tf][sym] = df

    # ---------------- Phase 3 & 4 ---------------- #
    logger.info("=== PHASE 3 & 4: Strategy signals -> Backtests ===")
    all_results = []
    analytics = PerformanceAnalytics()

    for timeframe in config.ALL_TIMEFRAMES:
        symbol_dfs = tf_symbol_dfs[timeframe]
        n_with_data = sum(1 for df in symbol_dfs.values() if df is not None and not df.empty)
        logger.info("Timeframe %s: %d/%d symbols have data", timeframe, n_with_data, len(symbols))
        if n_with_data == 0:
            continue

        for strategy_cls in ALL_STRATEGIES:
            strategy = strategy_cls(timeframe=timeframe)
            engine = BacktestEngine(strategy, timeframe)
            result = engine.run(symbol_dfs)
            metrics = analytics.evaluate(result)
            logger.info("  %-30s | trades=%3d win%%=%6.2f CAGR%%=%7.2f Sharpe=%6.2f MaxDD%%=%7.2f",
                        strategy.name, metrics.total_trades, metrics.win_rate_pct,
                        metrics.annualized_return_pct, metrics.sharpe_ratio, metrics.max_drawdown_pct)
            all_results.append(result)

    # ---------------- Comparison Matrix ---------------- #
    matrix_builder = ComparisonMatrixBuilder(analytics)
    matrix = matrix_builder.build(all_results)
    matrix_path = os.path.join(config.OUTPUT_DIR, "comparison_matrix.csv")
    matrix.to_csv(matrix_path, index=False)
    logger.info("Comparison matrix written to %s", matrix_path)

    if not matrix.empty:
        print("\n=== TOP STRATEGY x TIMEFRAME COMBINATIONS (by win rate) ===")
        print(matrix_builder.to_narrative(matrix, metric="win_rate_pct", top_n=10))

    # ---------------- Live Setup Scanner ---------------- #
    logger.info("=== Live Setup Scan ===")
    scanner = LiveSetupScanner([cls(timeframe="1D") for cls in ALL_STRATEGIES], recency_bars=3)
    active_setups = scanner.scan("1D", tf_symbol_dfs.get("1D", {}))
    setups_path = os.path.join(config.OUTPUT_DIR, "active_setups.csv")
    active_setups.to_csv(setups_path, index=False)
    logger.info("Active setups written to %s (%d rows)", setups_path, len(active_setups))

    if not active_setups.empty:
        print("\n=== ACTIVE SETUPS (manual review) ===")
        print(active_setups.to_string(index=False))


def _cli():
    parser = argparse.ArgumentParser(description="Techno-Funda Backtesting Engine")
    parser.add_argument("--csv", default="data/ind_nifty500list.csv", help="Path to the Nifty 500 constituents CSV")
    parser.add_argument("--limit", type=int, default=15, help="Cap universe size for a quick run")
    parser.add_argument("--lookback_days", type=int, default=720)
    args = parser.parse_args()
    run_pipeline(args.csv, limit=args.limit, lookback_days=args.lookback_days)


if __name__ == "__main__":
    _cli()
