"""
analytics.py -- PHASE 4 (part 2): the CFA-style performance layer

Metrics implemented per (Strategy, Timeframe) BacktestResult:
    - Total Cumulative Return, Annualized Return (CAGR)
    - Maximum Drawdown
    - Win Rate / Win-Loss Ratio / pattern "accuracy" (% of closed trades
      that were profitable)
    - Sharpe Ratio, using risk-free rate = 7% p.a. and a TIMEFRAME-AWARE
      annualization factor (sqrt of bars/year from config.BARS_PER_YEAR) --
      using a flat sqrt(252) for a 1W equity curve would badly overstate
      Sharpe, and using it for a 1H curve would badly understate it.

ComparisonMatrixBuilder assembles all combinations into one summary table,
e.g. "Double Bottom on 1D -> 65% accuracy, EMA Cross on 4H -> 55%".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from . import config
from .backtest import BacktestResult


@dataclass
class PerformanceMetrics:
    strategy: str
    timeframe: str
    total_trades: int
    win_rate_pct: float
    win_loss_ratio: float
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float

    def to_dict(self) -> dict:
        return self.__dict__


class PerformanceAnalytics:
    def __init__(self, risk_free_rate: float = config.CAPITAL.risk_free_rate):
        self.risk_free_rate = risk_free_rate

    def evaluate(self, result: BacktestResult) -> PerformanceMetrics:
        trades_df = result.trades_df()
        equity = result.equity_series()

        if equity.empty or len(equity) < 2:
            return self._empty_metrics(result)

        total_return_pct = (equity.iloc[-1] / result.starting_capital - 1) * 100

        years = self._elapsed_years(equity)
        annualized_return_pct = (
            ((equity.iloc[-1] / result.starting_capital) ** (1 / years) - 1) * 100
            if years > 0 and equity.iloc[-1] > 0 else 0.0
        )

        max_dd_pct = self._max_drawdown(equity)
        sharpe = self._sharpe(equity, result.timeframe)

        if trades_df.empty:
            wins = losses = win_rs = loss_rs = pd.Series(dtype=float)
        else:
            closed = trades_df.dropna(subset=["exit_price"])
            wins = closed.loc[closed["pnl"] > 0, "pnl_pct"]
            losses = closed.loc[closed["pnl"] <= 0, "pnl_pct"]
            win_rs = closed.loc[closed["pnl"] > 0, "pnl"]
            loss_rs = closed.loc[closed["pnl"] <= 0, "pnl"]

        total_closed = len(wins) + len(losses)
        win_rate = (len(wins) / total_closed * 100) if total_closed else 0.0
        avg_win = wins.mean() * 100 if len(wins) else 0.0
        avg_loss = losses.mean() * 100 if len(losses) else 0.0
        win_loss_ratio = (abs(avg_win / avg_loss) if avg_loss != 0 else np.inf) if len(losses) else np.inf
        # Profit factor = gross rupee profit / gross rupee loss
        gross_profit = win_rs.sum() if len(win_rs) else 0.0
        gross_loss = abs(loss_rs.sum()) if len(loss_rs) else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf

        return PerformanceMetrics(
            strategy=result.strategy, timeframe=result.timeframe, total_trades=total_closed,
            win_rate_pct=round(win_rate, 2), win_loss_ratio=round(win_loss_ratio, 2)
            if np.isfinite(win_loss_ratio) else float("inf"),
            total_return_pct=round(total_return_pct, 2),
            annualized_return_pct=round(annualized_return_pct, 2),
            max_drawdown_pct=round(max_dd_pct, 2), sharpe_ratio=round(sharpe, 2),
            avg_win_pct=round(avg_win, 2), avg_loss_pct=round(avg_loss, 2),
            profit_factor=round(profit_factor, 2) if np.isfinite(profit_factor) else float("inf"),
        )

    @staticmethod
    def _empty_metrics(result: BacktestResult) -> PerformanceMetrics:
        return PerformanceMetrics(result.strategy, result.timeframe, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    @staticmethod
    def _elapsed_years(equity: pd.Series) -> float:
        span_days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
        return max(span_days / 365.25, 1e-6)

    @staticmethod
    def _max_drawdown(equity: pd.Series) -> float:
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        return float(drawdown.min() * 100)

    def _sharpe(self, equity: pd.Series, timeframe: str) -> float:
        returns = equity.pct_change().dropna()
        if returns.empty or returns.std() < 1e-12:
            return 0.0
        bars_per_year = config.BARS_PER_YEAR.get(timeframe, 252)
        rf_per_bar = self.risk_free_rate / bars_per_year
        excess = returns - rf_per_bar
        return float(excess.mean() / returns.std() * np.sqrt(bars_per_year))


class ComparisonMatrixBuilder:
    """Builds the flagship output: a Strategy x Timeframe performance grid."""

    def __init__(self, analytics: PerformanceAnalytics = None):
        self.analytics = analytics or PerformanceAnalytics()

    def build(self, results: List[BacktestResult]) -> pd.DataFrame:
        rows = [self.analytics.evaluate(r).to_dict() for r in results]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values(
            ["sharpe_ratio", "annualized_return_pct"], ascending=False
        ).reset_index(drop=True)

    @staticmethod
    def pivot_by_metric(matrix: pd.DataFrame, metric: str = "win_rate_pct") -> pd.DataFrame:
        """Strategy (rows) x Timeframe (cols) view of a single metric -- the
        "Double Bottom on 1D = 65% accuracy, EMA Cross on 4H = 55%" style table.
        """
        if matrix.empty:
            return matrix
        return matrix.pivot(index="strategy", columns="timeframe", values=metric)

    @staticmethod
    def to_narrative(matrix: pd.DataFrame, metric: str = "win_rate_pct", top_n: int = 5) -> str:
        if matrix.empty:
            return "No trades were generated across any Strategy/Timeframe combination."
        top = matrix.sort_values(metric, ascending=False).head(top_n)
        lines = [
            f"{row.strategy} on {row.timeframe}: {row[metric]:.1f}% "
            f"(Sharpe {row.sharpe_ratio:.2f}, {row.total_trades} trades, "
            f"MaxDD {row.max_drawdown_pct:.1f}%)"
            for _, row in top.iterrows()
        ]
        return "\n".join(lines)
