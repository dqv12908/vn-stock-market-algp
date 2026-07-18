"""Data access layer over vnstock (VCI/TCBS public endpoints).

Three data granularities:
  1. Daily OHLCV        — quote.history            (years of history)
  2. Intraday ticks     — quote.intraday           (match-by-match with Buy/Sell
                          aggressor side; only recent sessions are served)
  3. Order book depth   — trading.price_board      (live top-10 bid/ask levels,
                          foreign flow, ATO/ATC prints; snapshot-only, so we
                          persist snapshots ourselves — see orderbook_collector)
"""

from __future__ import annotations

import os
import sys
import time
import logging
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get("VNMM_CACHE_DIR", "data/cache"))


@contextmanager
def _quiet():
    """Suppress vnstock's promotional banners on stdout/stderr."""
    devnull = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = devnull, devnull
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        devnull.close()


def _retry(fn, tries=3, delay=2.0):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — upstream raises bare Exception
            last = e
            time.sleep(delay * (2**i))
    raise last


def daily_history(symbol: str, start: str, end: str | None = None,
                  use_cache: bool = True) -> pd.DataFrame:
    """Daily OHLCV. Columns: time, open, high, low, close, volume."""
    end = end or date.today().isoformat()
    cache = CACHE_DIR / f"daily_{symbol}_{start}_{end}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    with _quiet():
        from vnstock import Vnstock
        q = Vnstock().stock(symbol=symbol, source="VCI").quote
        df = _retry(lambda: q.history(start=start, end=end, interval="1D"))
    df["time"] = pd.to_datetime(df["time"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def intraday_ticks(symbol: str, page_size: int = 30000) -> pd.DataFrame:
    """Most recent session's match-by-match prints.

    Columns: time, price, volume, match_type (Buy/Sell/ATO/ATC), id.
    match_type is the aggressor side: 'Buy' = executed against the ask
    (buyer lifted the offer), 'Sell' = hit the bid.
    """
    with _quiet():
        from vnstock import Vnstock
        q = Vnstock().stock(symbol=symbol, source="VCI").quote
        df = _retry(lambda: q.intraday(symbol=symbol, page_size=page_size))
    df["time"] = pd.to_datetime(df["time"])
    return df


def price_board(symbols: list[str]) -> pd.DataFrame:
    """Live snapshot: top-10 depth, foreign flow, ceiling/floor, session stats.

    Returns a flattened single-level-column DataFrame (group_field naming),
    e.g. listing_symbol, match_match_price, bid_ask_bid_1_price ...
    """
    with _quiet():
        from vnstock import Trading
        tr = Trading(source="VCI")
        df = _retry(lambda: tr.price_board(symbols))
    df.columns = ["_".join(c) if isinstance(c, tuple) else c for c in df.columns]
    return df


def index_history(start: str, end: str | None = None,
                  symbol: str = "VNINDEX", use_cache: bool = True) -> pd.DataFrame:
    """Daily index OHLCV (VNINDEX by default) for market-adjusting returns."""
    return daily_history(symbol, start, end, use_cache=use_cache)


def all_symbols(exchanges: tuple[str, ...] = ("HSX", "HNX")) -> pd.DataFrame:
    """Listing universe with exchange tags."""
    with _quiet():
        from vnstock import Listing
        df = _retry(lambda: Listing(source="VCI").symbols_by_exchange())
    return df[df["exchange"].isin(exchanges)].reset_index(drop=True)


def fetch_universe_daily(symbols: list[str], start: str,
                         end: str | None = None,
                         pause: float = 0.6) -> dict[str, pd.DataFrame]:
    """Bulk daily history download with polite rate limiting."""
    out: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        try:
            out[sym] = daily_history(sym, start, end)
        except Exception as e:  # noqa: BLE001
            log.warning("skip %s: %s", sym, e)
        if i % 10 == 9:
            log.info("fetched %d/%d", i + 1, len(symbols))
        time.sleep(pause)
    return out
