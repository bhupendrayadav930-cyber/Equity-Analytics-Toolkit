"""
smoke_test.py -- validates the whole Phase 3 + Phase 4 pipeline against
SYNTHETIC OHLCV data (no network calls), so we can catch import errors,
shape mismatches, and off-by-one bugs in the pattern math / backtest loop
without depending on yfinance/broker network access.
"""
import sys
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from techno_funda_engine import config
from techno_funda_engine.pivots import PivotDetector
from techno_funda_engine.strategies import ALL_STRATEGIES
from techno_funda_engine.backtest import BacktestEngine
from techno_funda_engine.analytics import PerformanceAnalytics, ComparisonMatrixBuilder


def make_synthetic_ohlcv(n=600, seed=0, regime="mixed") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B", tz="Asia/Kolkata")

    # Build a price path with trends, a rounded cup, a double top, and noise
    # so pattern detectors actually have something to find.
    t = np.arange(n)
    trend = np.piecewise(
        t.astype(float), [t < n * 0.3, (t >= n * 0.3) & (t < n * 0.55), t >= n * 0.55],
        [lambda x: 100 + 0.15 * x,
         lambda x: 100 + 0.15 * (n * 0.3) - 0.02 * (x - n * 0.3) ** 1.05 * 0 + 40 * np.sin((x - n*0.3) / (n*0.25) * np.pi),
         lambda x: 100 + 0.25 * (x - n * 0.55) + 60]
    )
    noise = rng.normal(0, 1.2, n).cumsum() * 0.3
    close = trend + noise
    close = np.maximum(close, 5)

    high = close + rng.uniform(0.2, 2.0, n)
    low = close - rng.uniform(0.2, 2.0, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    # inject a few volume spikes so volume-confirmation logic actually fires
    spike_days = rng.choice(n, size=n // 15, replace=False)
    volume[spike_days] *= rng.uniform(1.8, 3.0, len(spike_days))

    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


def main():
    failures = []

    # 1) Pivot detector sanity
    df = make_synthetic_ohlcv(seed=1)
    pivots = PivotDetector(timeframe="1D").find_pivots(df)
    print(f"[pivots] found {len(pivots)} pivots on {len(df)}-bar synthetic series")
    assert len(pivots) > 0, "PivotDetector found zero pivots -- something is wrong"

    # 2) Each strategy: generate_signals must not raise, must return correct columns
    expected_cols = {"signal", "trigger_price", "stop_loss", "pattern_tag", "confidence"}
    symbol_dfs = {}
    for i in range(6):
        symbol_dfs[f"TEST{i}.NS"] = make_synthetic_ohlcv(seed=i, n=650)

    signal_counts = {}
    for strategy_cls in ALL_STRATEGIES:
        strat = strategy_cls(timeframe="1D")
        try:
            sig = strat.generate_signals(df)
            missing = expected_cols - set(sig.columns)
            assert not missing, f"{strat.name} missing columns {missing}"
            assert len(sig) == len(df), f"{strat.name} changed row count"
            n_signals = int((sig["signal"] != 0).sum())
            signal_counts[strat.name] = n_signals
            print(f"[strategy] {strat.name:30s} OK  ({n_signals} signals fired)")
        except Exception as exc:
            failures.append((strat.name, "generate_signals", exc))
            print(f"[strategy] {strat.name:30s} FAILED: {exc}")
            traceback.print_exc()

    # 3) Full BacktestEngine run per strategy across the mini "universe"
    results = []
    analytics = PerformanceAnalytics()
    for strategy_cls in ALL_STRATEGIES:
        strat = strategy_cls(timeframe="1D")
        engine = BacktestEngine(strat, "1D")
        try:
            result = engine.run(symbol_dfs)
            metrics = analytics.evaluate(result)
            print(f"[backtest] {strat.name:30s} trades={metrics.total_trades:3d} "
                  f"win%={metrics.win_rate_pct:6.2f} CAGR%={metrics.annualized_return_pct:8.2f} "
                  f"Sharpe={metrics.sharpe_ratio:6.2f} MaxDD%={metrics.max_drawdown_pct:7.2f}")
            results.append(result)
        except Exception as exc:
            failures.append((strat.name, "backtest", exc))
            print(f"[backtest] {strat.name:30s} FAILED: {exc}")
            traceback.print_exc()

    # 4) Comparison matrix assembly
    try:
        matrix = ComparisonMatrixBuilder(analytics).build(results)
        print("\n[matrix] shape:", matrix.shape)
        print(matrix[["strategy", "timeframe", "total_trades", "win_rate_pct",
                       "sharpe_ratio", "max_drawdown_pct"]].to_string(index=False))
    except Exception as exc:
        failures.append(("ComparisonMatrixBuilder", "build", exc))
        traceback.print_exc()

    print("\n" + "=" * 60)
    if failures:
        print(f"SMOKE TEST: {len(failures)} FAILURE(S)")
        for name, stage, exc in failures:
            print(f"  - {name} @ {stage}: {exc}")
        sys.exit(1)
    else:
        print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
