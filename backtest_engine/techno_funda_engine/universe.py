"""
universe.py  --  PHASE 1: Data Ingestion & Fundamental Filtration
===================================================================

Pipeline
--------
1. UniverseBuilder            -> reads ind_nifty500list.csv, appends ".NS"
2. FundamentalDataProvider(s) -> builds the fundamental dataset from free
                                  public sources:
                                    a) YFinanceFundamentalProvider (default,
                                       no auth, decent coverage)
                                    b) ScreenerInFundamentalProvider (richer
                                       ratios, scrapes screener.in, use
                                       responsibly / rate-limited)
3. FundamentalFilter          -> merges universe + fundamentals, applies the
                                  quality screen (ROE, D/E, EPS growth, mcap)

Design note
-----------
FundamentalDataProvider is an ABC so a third provider can be added (e.g. a
paid Tijori/Trendlyne feed) later without touching FundamentalFilter.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Iterable, List, Optional

import pandas as pd
import requests

from . import config
from .logging_utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# 1. Universe
# --------------------------------------------------------------------------- #
class UniverseBuilder:
    """Reads the official NSE Nifty500 constituent list."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df.columns = [c.strip() for c in df.columns]
        if "Symbol" not in df.columns:
            raise ValueError(
                f"Expected a 'Symbol' column in {self.csv_path}, "
                f"found: {list(df.columns)}"
            )
        df["yf_symbol"] = df["Symbol"].astype(str).str.strip() + ".NS"
        logger.info("Loaded %d Nifty500 constituents from %s", len(df), self.csv_path)
        return df


# --------------------------------------------------------------------------- #
# 2. Fundamental data providers
# --------------------------------------------------------------------------- #
class FundamentalDataProvider(ABC):
    """Common interface: given a list of yfinance-style symbols (XXXX.NS),
    return a DataFrame indexed by symbol with at least these columns:
        roe (%), debt_to_equity, eps_growth (%), market_cap_cr (₹ Cr)
    """

    REQUIRED_COLUMNS = ["roe", "debt_to_equity", "eps_growth", "market_cap_cr"]

    @abstractmethod
    def fetch_one(self, symbol: str) -> Optional[dict]:
        ...

    def fetch_many(self, symbols: Iterable[str], throttle_sec: float = 0.0) -> pd.DataFrame:
        rows = []
        for sym in symbols:
            try:
                row = self.fetch_one(sym)
                if row:
                    row["symbol"] = sym
                    rows.append(row)
                else:
                    logger.warning("No fundamental data for %s", sym)
            except Exception as exc:  # noqa: BLE001 - keep the batch alive
                logger.error("Fundamental fetch failed for %s: %s", sym, exc)
            if throttle_sec:
                time.sleep(throttle_sec)

        df = pd.DataFrame(rows)
        if df.empty:
            logger.error("Fundamental provider returned NO rows at all.")
            return pd.DataFrame(columns=["symbol"] + self.REQUIRED_COLUMNS)
        df = df.set_index("symbol")
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        return df


class YFinanceFundamentalProvider(FundamentalDataProvider):
    """Free, no auth. Pulls ratios out of yfinance's `Ticker.info` /
    `Ticker.get_info()`. Coverage on Indian mid/small caps is imperfect —
    treat NaNs as 'unknown, exclude' rather than 'fails the filter'.
    """

    def fetch_one(self, symbol: str) -> Optional[dict]:
        import yfinance as yf

        info = yf.Ticker(symbol).get_info()
        if not info:
            return None

        roe = info.get("returnOnEquity")
        de = info.get("debtToEquity")
        eps_growth = info.get("earningsGrowth")
        mcap = info.get("marketCap")

        return {
            "roe": None if roe is None else roe * 100,          # fraction -> %
            "debt_to_equity": None if de is None else de / 100,  # yfinance reports as %, normalise to ratio
            "eps_growth": None if eps_growth is None else eps_growth * 100,
            "market_cap_cr": None if mcap is None else mcap / 1e7,  # ₹ -> ₹ Crore
        }


class ScreenerInFundamentalProvider(FundamentalDataProvider):
    """Scrapes the public 'quick ratios' section of screener.in for richer,
    India-specific fundamentals (this is what most Indian retail quants
    use in practice since NSE/BSE don't expose ROE/D-E directly).

    IMPORTANT: screener.in's ToS restricts automated/bulk scraping for
    commercial redistribution. This is intended for personal research use
    only, with polite rate-limiting (throttle_sec). If a
    screener.in "Premium" export or API access, prefer that over scraping.
    """

    BASE_URL = "https://www.screener.in/company/{symbol}/"
    HEADERS = {"User-Agent": "Mozilla/5.0 (research use)"}

    def fetch_one(self, symbol: str) -> Optional[dict]:
        from bs4 import BeautifulSoup  # local import: optional dependency

        nse_code = symbol.replace(".NS", "")
        resp = requests.get(self.BASE_URL.format(symbol=nse_code),
                             headers=self.HEADERS, timeout=15)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        ratios = {}
        for li in soup.select("#top-ratios li"):
            name_el = li.select_one(".name")
            val_el = li.select_one(".value")
            if name_el and val_el:
                ratios[name_el.text.strip()] = val_el.text.strip()

        def _num(s: Optional[str]) -> Optional[float]:
            if not s:
                return None
            s = s.replace(",", "").replace("%", "").replace("₹", "").replace("Cr.", "").strip()
            try:
                return float(s)
            except ValueError:
                return None

        return {
            "roe": _num(ratios.get("ROE")),
            "debt_to_equity": _num(ratios.get("Debt to equity")),
            # The top-ratios panel shows absolute EPS, not growth, so growth is
            # left unknown rather than mislabelled. FundamentalFilter skips any
            # criterion whose column is entirely missing.
            "eps_growth": None,
            "market_cap_cr": _num(ratios.get("Market Cap")),
        }


# --------------------------------------------------------------------------- #
# 3. Filter
# --------------------------------------------------------------------------- #
class FundamentalFilter:
    def __init__(self, criteria: config.FundamentalCriteria = config.FUNDAMENTALS):
        self.criteria = criteria

    def apply(self, universe_df: pd.DataFrame, fundamentals_df: pd.DataFrame) -> pd.DataFrame:
        merged = universe_df.merge(
            fundamentals_df, left_on="yf_symbol", right_index=True, how="inner"
        )
        before = len(merged)

        rules = {
            "roe": lambda s: s >= self.criteria.min_roe,
            "debt_to_equity": lambda s: s <= self.criteria.max_debt_to_equity,
            "eps_growth": lambda s: s >= self.criteria.min_eps_growth,
            "market_cap_cr": lambda s: s >= self.criteria.min_market_cap_cr,
        }
        mask = pd.Series(True, index=merged.index)
        for col, rule in rules.items():
            values = pd.to_numeric(merged[col], errors="coerce")
            if values.isna().all():
                logger.warning("No data for '%s' from this provider; criterion skipped.", col)
                continue
            # A missing value for an individual stock = unknown = excluded.
            mask &= rule(values).fillna(False)
        passed = merged[mask].reset_index(drop=True)
        logger.info(
            "Fundamental filter: %d/%d stocks passed (ROE>=%.1f, D/E<=%.1f, "
            "EPS growth>=%.1f%%, MCap>=₹%.0fCr)",
            len(passed), before, self.criteria.min_roe, self.criteria.max_debt_to_equity,
            self.criteria.min_eps_growth, self.criteria.min_market_cap_cr,
        )
        return passed


# --------------------------------------------------------------------------- #
# Convenience orchestrator
# --------------------------------------------------------------------------- #
def build_cleared_universe(
    nifty500_csv_path: str,
    provider: Optional[FundamentalDataProvider] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """One-call helper: CSV -> fundamentals -> filtered list of yf_symbols."""
    provider = provider or YFinanceFundamentalProvider()

    universe_df = UniverseBuilder(nifty500_csv_path).load()
    symbols: List[str] = universe_df["yf_symbol"].tolist()
    if limit:
        symbols = symbols[:limit]
        universe_df = universe_df[universe_df["yf_symbol"].isin(symbols)]

    fundamentals_df = provider.fetch_many(symbols, throttle_sec=0.2)
    return FundamentalFilter().apply(universe_df, fundamentals_df)
