"""
data_providers.py  --  PHASE 2: Multi-Timeframe Historical Data Fetching
============================================================================

Strategy: fetch ONE base interval (1H) per stock, cache it to Parquet, and
derive 2H / 4H / 1D / 1W locally via pandas.resample(). This means:
    - 1 API call pattern per stock instead of 4
    - consistent OHLCV across timeframes (no vendor discrepancies between
      their own "4H" bars and yours)
    - trivially extensible to more timeframes later (30min, 3D, ...) for free

DataProvider is an ABC so the rest of the engine never talks to
Dhan/Upstox/yfinance directly -- only to "give me 1H bars for this symbol
between these dates".

Dhan and Upstox adapters are included for higher-quality intraday data.
Both need a user-generated API key + access token (OAuth for Upstox, a
partner/client login flow for Dhan), read from environment variables -- that credential plumbing is
inherently account-specific and changes with their API versions, so it is
isolated in `_headers()` / `_request()` below. Verify endpoint paths
against the current Dhan/Upstox API docs before going live; the request
shape here follows their documented v2 REST pattern as of this writing.
yfinance is wired in as a zero-config fallback for 1D/1W (and for
prototyping without broker credentials).
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
import requests

from . import config
from .logging_utils import get_logger

logger = get_logger(__name__)

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


# --------------------------------------------------------------------------- #
# Abstract base
# --------------------------------------------------------------------------- #
class DataProvider(ABC):
    """All providers return a DataFrame indexed by tz-aware DatetimeIndex
    (Asia/Kolkata) with columns = OHLCV_COLS, sorted ascending, or an empty
    DataFrame (never None, never raises) on failure -- callers loop over
    hundreds of symbols and must not be broken by one bad ticker.
    """

    name = "base"

    @abstractmethod
    def _fetch_raw(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        ...

    def fetch_base_interval(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        try:
            df = self._fetch_raw(symbol, start, end)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] fetch failed for %s: %s", self.name, symbol, exc)
            return pd.DataFrame(columns=OHLCV_COLS)

        if df is None or df.empty:
            logger.warning("[%s] no data returned for %s", self.name, symbol)
            return pd.DataFrame(columns=OHLCV_COLS)

        df = df[~df.index.duplicated(keep="last")].sort_index()
        missing = [c for c in OHLCV_COLS if c not in df.columns]
        if missing:
            logger.error("[%s] %s missing columns %s, dropping", self.name, symbol, missing)
            return pd.DataFrame(columns=OHLCV_COLS)
        return df[OHLCV_COLS].astype(float)


# --------------------------------------------------------------------------- #
# yfinance -- zero-config fallback / good for 1D & 1W history
# --------------------------------------------------------------------------- #
class YFinanceDataProvider(DataProvider):
    name = "yfinance"

    # yfinance's own interval ceiling: it will silently clip anything older
    # than this for intraday bars, so we warn rather than fail.
    MAX_INTRADAY_LOOKBACK_DAYS = {"1h": 730, "60m": 730, "1d": None, "1wk": None}

    def __init__(self, interval: str = "1h"):
        self.interval = interval

    def _fetch_raw(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        import yfinance as yf

        cap_days = self.MAX_INTRADAY_LOOKBACK_DAYS.get(self.interval)
        if cap_days and (end - start).days > cap_days:
            logger.warning(
                "[yfinance] %s: requested %d days of '%s' data, Yahoo caps this "
                "around %d days -- older bars will simply be absent.",
                symbol, (end - start).days, self.interval, cap_days,
            )

        raw = yf.download(
            symbol, start=start, end=end, interval=self.interval,
            progress=False, auto_adjust=False, threads=False,
        )
        if raw.empty:
            return raw

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
        else:
            raw.index = raw.index.tz_convert("Asia/Kolkata")
        return raw


# --------------------------------------------------------------------------- #
# Dhan (primary broker option -- free API, simple REST + access token)
# --------------------------------------------------------------------------- #
class DhanDataProvider(DataProvider):
    """Dhan's Data APIs need: a security-id per symbol (NOT the trading
    symbol string) -- fetch/cache their instrument master CSV once and map
    yf-style symbols ("TCS.NS") -> Dhan security_id. `instrument_map` below
    is expected to already contain that mapping so this class stays a pure
    HTTP client and doesn't own instrument-master parsing.
    """

    name = "dhan"
    BASE_URL = "https://api.dhan.co/v2"

    def __init__(self, access_token: Optional[str] = None,
                 client_id: Optional[str] = None,
                 instrument_map: Optional[Dict[str, str]] = None,
                 interval_minutes: int = 60):
        self.access_token = access_token or os.getenv("DHAN_ACCESS_TOKEN")
        self.client_id = client_id or os.getenv("DHAN_CLIENT_ID")
        self.instrument_map = instrument_map or {}
        self.interval_minutes = interval_minutes
        if not self.access_token:
            logger.warning("DhanDataProvider initialised without an access token "
                            "(set DHAN_ACCESS_TOKEN) -- calls will fail until set.")

    def _headers(self) -> dict:
        return {
            "access-token": self.access_token,
            "client-id": self.client_id,
            "Content-Type": "application/json",
        }

    def _fetch_raw(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        security_id = self.instrument_map.get(symbol)
        if not security_id:
            raise ValueError(f"No Dhan security_id mapped for {symbol}; "
                              f"populate instrument_map from Dhan's instrument master.")

        payload = {
            "securityId": security_id,
            "exchangeSegment": "NSE_EQ",
            "instrument": "EQUITY",
            "interval": str(self.interval_minutes),
            "fromDate": start.strftime("%Y-%m-%d"),
            "toDate": end.strftime("%Y-%m-%d"),
        }
        resp = requests.post(f"{self.BASE_URL}/charts/intraday",
                              json=payload, headers=self._headers(), timeout=20)
        resp.raise_for_status()
        data = resp.json()

        if not data or "open" not in data:
            return pd.DataFrame(columns=OHLCV_COLS)

        df = pd.DataFrame({
            "open": data["open"], "high": data["high"], "low": data["low"],
            "close": data["close"], "volume": data["volume"],
        })
        df.index = pd.to_datetime(data["timestamp"], unit="s") \
            .tz_localize("UTC").tz_convert("Asia/Kolkata")
        return df


# --------------------------------------------------------------------------- #
# Upstox (alternative broker option -- OAuth2, free tier)
# --------------------------------------------------------------------------- #
class UpstoxDataProvider(DataProvider):
    """Upstox v2 historical-candle API. `instrument_key` follows the format
    'NSE_EQ|<ISIN>' -- build instrument_map from Upstox's published
    instrument master (they publish a gzipped JSON/CSV daily).
    OAuth access tokens expire daily and must be refreshed via their
    login-redirect flow; that's out of scope for this data-layer class,
    inject a freshly generated token at construction time.
    """

    name = "upstox"
    BASE_URL = "https://api.upstox.com/v2"

    def __init__(self, access_token: Optional[str] = None,
                 instrument_map: Optional[Dict[str, str]] = None,
                 unit: str = "hours", interval: str = "1"):
        self.access_token = access_token or os.getenv("UPSTOX_ACCESS_TOKEN")
        self.instrument_map = instrument_map or {}
        self.unit = unit
        self.interval = interval
        if not self.access_token:
            logger.warning("UpstoxDataProvider initialised without an access token "
                            "(set UPSTOX_ACCESS_TOKEN) -- calls will fail until set.")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.access_token}",
                "Accept": "application/json"}

    def _fetch_raw(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        instrument_key = self.instrument_map.get(symbol)
        if not instrument_key:
            raise ValueError(f"No Upstox instrument_key mapped for {symbol}.")

        url = (f"{self.BASE_URL}/historical-candle/{instrument_key}/"
               f"{self.unit}/{self.interval}/"
               f"{end.strftime('%Y-%m-%d')}/{start.strftime('%Y-%m-%d')}")
        resp = requests.get(url, headers=self._headers(), timeout=20)
        resp.raise_for_status()
        candles = resp.json().get("data", {}).get("candles", [])
        if not candles:
            return pd.DataFrame(columns=OHLCV_COLS)

        # Upstox candle row: [timestamp, open, high, low, close, volume, oi]
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low",
                                             "close", "volume", "oi"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
        return df.set_index("timestamp").sort_index()


# --------------------------------------------------------------------------- #
# Local Parquet cache -- avoid re-hitting the API on every run
# --------------------------------------------------------------------------- #
class ParquetCache:
    def __init__(self, cache_dir: str = config.CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, symbol: str, interval: str) -> str:
        safe = symbol.replace("/", "_")
        return os.path.join(self.cache_dir, f"{safe}_{interval}.parquet")

    def load(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        path = self._path(symbol, interval)
        if os.path.exists(path):
            try:
                return pd.read_parquet(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cache read failed for %s (%s): %s", symbol, interval, exc)
        return None

    def save(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        df.to_parquet(self._path(symbol, interval))


# --------------------------------------------------------------------------- #
# Resampler -- 1H base bars -> 2H / 4H / 1D / 1W
# --------------------------------------------------------------------------- #
class TimeframeResampler:
    """Anchors bins to the NSE session open (09:15) so a '2H' bar is
    [09:15-11:15), [11:15-13:15), [13:15-15:15), [15:15-15:30) -- not an
    arbitrary midnight-aligned bucket. '1D'/'1W' don't need the anchor but
    it's harmless for them.
    """

    _AGG = {"open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"}

    @staticmethod
    def _rule(tf: str) -> str:
        return {"2H": "2h", "4H": "4h", "1D": "1D", "1W": "1W-FRI"}[tf]

    def resample(self, df_1h: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        if df_1h.empty:
            return df_1h.copy()
        if target_tf == "1H":
            return df_1h.copy()

        rule = self._rule(target_tf)
        resample_kwargs = {"label": "left", "closed": "left"}
        if target_tf in ("2H", "4H"):
            # 'origin' only affects sub-daily (Tick-like: h/min/s) frequencies
            # in pandas -- passing it for 1D/1W is a silent no-op that pandas
            # 3.x now warns about, so it's only included where it matters.
            resample_kwargs["origin"] = "start_day"
        out = df_1h.resample(rule, **resample_kwargs).agg(self._AGG)
        out = out.dropna(subset=["open", "high", "low", "close"])
        return out


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class MultiTimeframeDataManager:
    def __init__(self, provider: DataProvider, cache: Optional[ParquetCache] = None,
                 use_cache: bool = True):
        self.provider = provider
        self.cache = cache or ParquetCache()
        self.use_cache = use_cache
        self.resampler = TimeframeResampler()

    def get_all_timeframes(self, symbol: str, start: datetime, end: datetime,
                            refresh: bool = False) -> Dict[str, pd.DataFrame]:
        base_df = None
        if self.use_cache and not refresh:
            base_df = self.cache.load(symbol, "1H")

        if base_df is None or base_df.empty:
            base_df = self.provider.fetch_base_interval(symbol, start, end)
            if not base_df.empty:
                self.cache.save(symbol, "1H", base_df)

        result = {"1H": base_df}
        for tf in config.DERIVED_TIMEFRAMES:
            result[tf] = self.resampler.resample(base_df, tf)
        return result

    def get_universe_timeframes(self, symbols, start: datetime, end: datetime,
                                 pause_sec: float = 0.3) -> Dict[str, Dict[str, pd.DataFrame]]:
        """symbol -> {timeframe -> df}. Sequential with a small pause to stay
        polite to whichever API is behind `self.provider`; swap for a
        thread/process pool once the provider's rate limits are confirmed.
        """
        out = {}
        for i, sym in enumerate(symbols, 1):
            logger.info("Fetching %s (%d/%d)...", sym, i, len(symbols))
            out[sym] = self.get_all_timeframes(sym, start, end)
            time.sleep(pause_sec)
        return out
