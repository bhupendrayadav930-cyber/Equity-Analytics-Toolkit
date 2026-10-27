"""
config.py
---------
Single source of truth for every tunable constant in the engine.
Nothing else in the codebase should hard-code a number that belongs here.
"""
from dataclasses import dataclass, field
from typing import Dict, List


# --------------------------------------------------------------------------- #
# Capital / Risk Management
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CapitalConfig:
    starting_capital: float = 1_000_000.0     # ₹10,00,000
    allocation_per_trade: float = 0.05        # 5% of *current* equity per trade
    max_concurrent_positions: int = 20        # hard cap -> prevents >100% deployment
    risk_free_rate: float = 0.07              # for Sharpe Ratio (7% p.a.)
    # Round-trip friction applied on BOTH entry and exit: approx. brokerage +
    # STT + exchange/SEBI charges + stamp duty + slippage for NSE delivery.
    transaction_cost_pct: float = 0.0015      # 0.15% per side
    # Idle cash earns the risk-free rate, so the equity curve and the Sharpe
    # ratio's risk-free hurdle are measured on the same basis.
    cash_earns_risk_free: bool = True


# --------------------------------------------------------------------------- #
# Fundamental Screening Criteria (Phase 1)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FundamentalCriteria:
    min_roe: float = 15.0            # Return on Equity, %
    max_debt_to_equity: float = 1.0  # Debt/Equity ratio
    min_eps_growth: float = 0.0      # YoY EPS growth, % (0 = must be growing)
    min_market_cap_cr: float = 500.0 # ₹ Crore, filters out illiquid micro-caps


# --------------------------------------------------------------------------- #
# Timeframes
# --------------------------------------------------------------------------- #
# Everything is fetched at the BASE_INTERVAL and resampled locally to the
# others. This means exactly one API call pattern per stock instead of four.
BASE_INTERVAL: str = "1h"
DERIVED_TIMEFRAMES: List[str] = ["2H", "4H", "1D", "1W"]
ALL_TIMEFRAMES: List[str] = [BASE_INTERVAL.upper()] + DERIVED_TIMEFRAMES

# NSE cash market session, IST. Used to anchor resample bin edges so a "2H"
# bar starts at 09:15 rather than at midnight (pandas' default origin).
NSE_SESSION_START = "09:15"
NSE_SESSION_END = "15:30"

# Approx. number of *bars per calendar year* for each timeframe, used to
# annualize returns / Sharpe correctly. NSE ~ 252 trading days/year,
# 6.25 trading hours/day.
BARS_PER_YEAR: Dict[str, float] = {
    "1H": 252 * 6,      # 6 x 1H bars/day (approx, last bar is a 15-min remainder)
    "2H": 252 * 3,
    "4H": 252 * 1.5,
    "1D": 252,
    "1W": 52,
}


# --------------------------------------------------------------------------- #
# Pivot / Pattern Detection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PatternConfig:
    # scipy.signal.argrelextrema "order" = number of bars on each side a
    # point must be higher/lower than to qualify as a pivot. Larger order =
    # fewer, more significant pivots. Kept per-timeframe because 500 bars on
    # 1D is very different in "noise" terms from 500 bars on 1H.
    pivot_order: Dict[str, int] = field(default_factory=lambda: {
        "1H": 12, "2H": 8, "4H": 6, "1D": 5, "1W": 3,
    })

    double_pattern_tolerance: float = 0.015     # ±1.5% for double top/bottom
    shoulder_symmetry_tolerance: float = 0.08   # ±8% between H&S shoulders
    triangle_min_touches: int = 4               # min pivot touches to confirm a triangle
    breakout_confirmation_pct: float = 0.003    # 0.3% beyond level = real breakout, not noise
    volume_surge_multiple: float = 1.5          # x of 20-period avg volume
    flagpole_min_move_pct: float = 0.08         # min 8% impulsive move to call it a flagpole
    max_pattern_lookback_bars: int = 120        # don't search patterns older than this


# --------------------------------------------------------------------------- #
# Strategy-specific defaults
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StrategyConfig:
    ema_fast: int = 50
    ema_slow: int = 200
    trailing_stop_ema: int = 20
    volume_avg_window: int = 20
    trend_confirmation_ema_short: int = 50


CAPITAL = CapitalConfig()
FUNDAMENTALS = FundamentalCriteria()
PATTERNS = PatternConfig()
STRATEGY = StrategyConfig()

LOG_DIR = "logs"
CACHE_DIR = "data_cache"
OUTPUT_DIR = "output"
