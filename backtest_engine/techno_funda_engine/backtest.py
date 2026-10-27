"""
backtest.py -- PHASE 4 (part 1): trade simulation

Each (Strategy, Timeframe) combination is backtested
INDEPENDENTLY with its own fresh ₹10,00,000 capital pool (so the
comparison matrix in analytics.py is apples-to-apples: "Double Bottom on
1D" and "EMA Cross on 4H" each start from the same base). Within one such
run, ONE shared PortfolioManager is used across every stock in the
filtered universe -- this is what prevents the classic bug of 200 stocks
each pretending they had their own independent ₹10L (which would silently
let the backtest deploy 200x the real capital).

Long-only by default (config.ALLOW_SHORT_TRADES). Indian cash-market swing
trading typically can't hold overnight shorts without F&O margin, so
bearish pattern signals (-1) are treated as EXIT triggers for existing
longs, not fresh short entries, unless shorting is explicitly enabled.

Exit rule (applies uniformly on top of whatever entry strategy fired):
    effective_stop = max(pattern's initial stop_loss, EMA-20)   [long]
    exit if bar's low <= effective_stop, OR the strategy fires an
    opposite (-1) signal on that symbol, OR data simply ends.
This "trailing 20-EMA or structure failure" exit is applied identically
across all 7 strategies so returns are comparable and not an artifact of
one strategy having a looser exit.

Costs: every entry and exit pays config.CAPITAL.transaction_cost_pct of
traded value, and idle cash accrues the risk-free rate per bar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from . import config
from .logging_utils import get_logger
from .strategies.base import BaseStrategy

logger = get_logger(__name__)

ALLOW_SHORT_TRADES = False


@dataclass
class Trade:
    symbol: str
    strategy: str
    timeframe: str
    direction: int
    entry_time: pd.Timestamp
    entry_price: float
    qty: int
    capital_allocated: float
    stop_loss: float
    pattern_tag: str
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.qty * self.direction

    @property
    def pnl_pct(self) -> float:
        if self.exit_price is None or self.entry_price == 0:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price * self.direction


class PortfolioManager:
    def __init__(self, starting_capital: float = config.CAPITAL.starting_capital,
                 allocation_pct: float = config.CAPITAL.allocation_per_trade,
                 max_positions: int = config.CAPITAL.max_concurrent_positions,
                 cost_pct: float = config.CAPITAL.transaction_cost_pct):
        self.cash = starting_capital
        self.cost_pct = cost_pct
        self.total_costs = 0.0
        self.allocation_pct = allocation_pct
        self.max_positions = max_positions
        self.open_positions: Dict[str, Trade] = {}
        self.trade_log: List[Trade] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []

    def _mark_to_market(self, last_close: Dict[str, float]) -> float:
        """Cash + mark-to-market value of open positions. For a long, that's
        simply qty * current_price. For a short, cash already reflects the
        short-sale proceeds, so its mark-to-market contribution is the
        capital originally committed plus/minus the move against it.
        """
        open_value = 0.0
        for sym, pos in self.open_positions.items():
            price = last_close.get(sym, pos.entry_price)
            if pos.direction == 1:
                open_value += pos.qty * price
            else:  # short: unrealised P&L relative to entry, on top of capital already held as cash
                open_value += pos.capital_allocated + (pos.entry_price - price) * pos.qty
        return self.cash + open_value

    def current_equity(self, last_close: Dict[str, float]) -> float:
        return self._mark_to_market(last_close)

    def can_open(self) -> bool:
        return len(self.open_positions) < self.max_positions

    def open_position(self, symbol: str, strategy: str, timeframe: str,
                       ts: pd.Timestamp, price: float, stop_loss: float,
                       pattern_tag: str, last_close: Dict[str, float],
                       direction: int = 1) -> bool:
        if symbol in self.open_positions or not self.can_open() or price <= 0:
            return False

        equity = self.current_equity(last_close)
        capital_to_deploy = equity * self.allocation_pct
        capital_to_deploy = min(capital_to_deploy, self.cash)  # never exceed actual free cash
        qty = int(capital_to_deploy // (price * (1 + self.cost_pct)))
        if qty <= 0:
            return False

        cost = qty * price
        fees = cost * self.cost_pct
        self.cash -= cost + fees
        self.total_costs += fees
        trade = Trade(
            symbol=symbol, strategy=strategy, timeframe=timeframe, direction=direction,
            entry_time=ts, entry_price=price, qty=qty, capital_allocated=cost,
            stop_loss=stop_loss, pattern_tag=pattern_tag,
        )
        self.open_positions[symbol] = trade
        return True

    def close_position(self, symbol: str, ts: pd.Timestamp, price: float, reason: str) -> None:
        pos = self.open_positions.pop(symbol, None)
        if pos is None:
            return
        pos.exit_time = ts
        pos.exit_price = price
        pos.exit_reason = reason
        proceeds = pos.qty * price
        fees = proceeds * self.cost_pct
        self.cash += proceeds - fees
        self.total_costs += fees
        self.trade_log.append(pos)

    def close_all_remaining(self, last_close: Dict[str, float], ts: pd.Timestamp) -> None:
        for symbol in list(self.open_positions.keys()):
            price = last_close.get(symbol, self.open_positions[symbol].entry_price)
            self.close_position(symbol, ts, price, "backtest_end")

    def accrue_interest(self, rate_per_bar: float) -> None:
        """Idle cash earns the risk-free rate (e.g. liquid fund / T-bill)."""
        if self.cash > 0:
            self.cash *= 1 + rate_per_bar

    def snapshot_equity(self, ts: pd.Timestamp, last_close: Dict[str, float]) -> None:
        point = (ts, self.current_equity(last_close))
        if self.equity_curve and self.equity_curve[-1][0] == ts:
            self.equity_curve[-1] = point   # never record the same bar twice
        else:
            self.equity_curve.append(point)


@dataclass
class BacktestResult:
    strategy: str
    timeframe: str
    trades: List[Trade]
    equity_curve: List[Tuple[pd.Timestamp, float]]
    starting_capital: float

    def equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float)
        idx, vals = zip(*self.equity_curve)
        return pd.Series(vals, index=pd.DatetimeIndex(idx)).sort_index()

    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            "symbol": t.symbol, "direction": t.direction, "entry_time": t.entry_time,
            "entry_price": t.entry_price, "exit_time": t.exit_time, "exit_price": t.exit_price,
            "qty": t.qty, "capital_allocated": t.capital_allocated, "pnl": t.pnl,
            "pnl_pct": t.pnl_pct, "pattern_tag": t.pattern_tag, "exit_reason": t.exit_reason,
        } for t in self.trades])


class BacktestEngine:
    def __init__(self, strategy: BaseStrategy, timeframe: str,
                 allow_short: bool = ALLOW_SHORT_TRADES):
        self.strategy = strategy
        self.timeframe = timeframe
        self.allow_short = allow_short

    def _prepare(self, symbol_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        prepared = {}
        for sym, df in symbol_dfs.items():
            if df is None or df.empty or len(df) < 30:
                continue
            try:
                sig_df = self.strategy.generate_signals(df)
            except Exception as exc:  # noqa: BLE001
                logger.error("Signal generation failed for %s/%s/%s: %s",
                             self.strategy.name, self.timeframe, sym, exc)
                continue
            sig_df["ema_trail"] = df["close"].ewm(
                span=config.STRATEGY.trailing_stop_ema, adjust=False
            ).mean()
            prepared[sym] = sig_df
        return prepared

    def _check_exit(self, pos: Trade, row: pd.Series) -> Tuple[Optional[float], Optional[str]]:
        if pos.direction == 1:
            effective_stop = max(pos.stop_loss, row["ema_trail"])
            if row["low"] <= effective_stop:
                exit_price = min(row["open"], effective_stop) if row["open"] < effective_stop else effective_stop
                return exit_price, "stop_loss_or_trailing_ema"
            if row["signal"] == -1:
                return row["close"], "opposite_signal"
        else:  # short
            effective_stop = min(pos.stop_loss, row["ema_trail"])
            if row["high"] >= effective_stop:
                exit_price = max(row["open"], effective_stop) if row["open"] > effective_stop else effective_stop
                return exit_price, "stop_loss_or_trailing_ema"
            if row["signal"] == 1:
                return row["close"], "opposite_signal"
        return None, None

    def run(self, symbol_dfs: Dict[str, pd.DataFrame]) -> BacktestResult:
        prepared = self._prepare(symbol_dfs)
        portfolio = PortfolioManager()

        if not prepared:
            logger.warning("No usable data for %s / %s -- empty backtest.",
                            self.strategy.name, self.timeframe)
            return BacktestResult(self.strategy.name, self.timeframe, [], [],
                                   config.CAPITAL.starting_capital)

        all_ts = sorted(set().union(*[df.index for df in prepared.values()]))
        last_close: Dict[str, float] = {}
        rf_per_bar = (
            config.CAPITAL.risk_free_rate / config.BARS_PER_YEAR.get(self.timeframe, 252)
            if config.CAPITAL.cash_earns_risk_free else 0.0
        )

        for ts in all_ts:
            portfolio.accrue_interest(rf_per_bar)

            # 1) exits first -- never let a new entry mask a stop that hit today
            for sym in list(portfolio.open_positions.keys()):
                df = prepared.get(sym)
                if df is None or ts not in df.index:
                    continue
                row = df.loc[ts]
                exit_price, reason = self._check_exit(portfolio.open_positions[sym], row)
                if exit_price is not None:
                    portfolio.close_position(sym, ts, float(exit_price), reason)

            # 2) refresh mark-to-market prices
            for sym, df in prepared.items():
                if ts in df.index:
                    last_close[sym] = float(df.loc[ts, "close"])

            # 3) new entries
            if portfolio.can_open():
                for sym, df in prepared.items():
                    if sym in portfolio.open_positions or ts not in df.index:
                        continue
                    row = df.loc[ts]
                    sig = row["signal"]
                    if sig == 1:
                        price = row["trigger_price"] if pd.notna(row["trigger_price"]) else row["close"]
                        portfolio.open_position(sym, self.strategy.name, self.timeframe, ts,
                                                 float(price), float(row["stop_loss"]),
                                                 row["pattern_tag"], last_close, direction=1)
                    elif sig == -1 and self.allow_short:
                        price = row["trigger_price"] if pd.notna(row["trigger_price"]) else row["close"]
                        portfolio.open_position(sym, self.strategy.name, self.timeframe, ts,
                                                 float(price), float(row["stop_loss"]),
                                                 row["pattern_tag"], last_close, direction=-1)
                    if not portfolio.can_open():
                        break

            # 4) equity snapshot
            portfolio.snapshot_equity(ts, last_close)

        portfolio.close_all_remaining(last_close, all_ts[-1])
        portfolio.snapshot_equity(all_ts[-1], last_close)

        return BacktestResult(
            strategy=self.strategy.name, timeframe=self.timeframe,
            trades=portfolio.trade_log, equity_curve=portfolio.equity_curve,
            starting_capital=config.CAPITAL.starting_capital,
        )
