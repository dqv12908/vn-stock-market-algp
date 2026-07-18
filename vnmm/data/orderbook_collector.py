"""Order book snapshot collector.

Historical depth data for HOSE/HNX is not publicly available — brokers sell
it, exchanges don't. But the live top-10 book IS free via VCI's price board.
So we build our own history: poll during trading hours, append to daily
parquet files, and after a few weeks you have the dataset every depth
detector in vnmm.signals.orderbook needs.

Run:  python scripts/collect_orderbook.py --watchlist config.yaml
Storage: data/orderbook/YYYY-MM-DD/SYMBOL.parquet  (one row per snapshot)

VN trading hours (ICT): 09:00-11:30, 13:00-14:45 (HOSE ATC 14:30-14:45).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from . import loader

log = logging.getLogger(__name__)
ICT = ZoneInfo("Asia/Ho_Chi_Minh")

SESSIONS = [(dtime(9, 0), dtime(11, 30)), (dtime(13, 0), dtime(14, 46))]

KEEP_PREFIXES = ("bid_ask_", )
KEEP_COLS = (
    "listing_symbol", "listing_ceiling", "listing_floor", "listing_ref_price",
    "match_match_price", "match_match_vol", "match_accumulated_volume",
    "match_accumulated_value", "match_avg_match_price",
    "match_foreign_buy_volume", "match_foreign_sell_volume",
    "match_total_buy_orders", "match_total_sell_orders",
    "match_highest", "match_lowest",
)


def in_session(now: datetime | None = None) -> bool:
    now = now or datetime.now(ICT)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return any(a <= t <= b for a, b in SESSIONS)


def snapshot(symbols: list[str]) -> pd.DataFrame:
    df = loader.price_board(symbols)
    cols = [c for c in df.columns
            if c in KEEP_COLS or c.startswith(KEEP_PREFIXES)]
    out = df[cols].copy()
    out["ts"] = datetime.now(ICT).isoformat()
    return out


def append_snapshot(df: pd.DataFrame, root: Path = Path("data/orderbook")) -> None:
    day = datetime.now(ICT).date().isoformat()
    for _, row in df.iterrows():
        sym = row["listing_symbol"]
        path = root / day / f"{sym}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = row.to_frame().T
        if path.exists():
            rec = pd.concat([pd.read_parquet(path), rec], ignore_index=True)
        rec.to_parquet(path, index=False)


def load_day(symbol: str, day: str,
             root: Path = Path("data/orderbook")) -> pd.DataFrame:
    path = root / day / f"{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts").sort_index()


def run(symbols: list[str], interval_s: float = 15.0,
        root: Path = Path("data/orderbook")) -> None:
    """Poll forever during market hours; sleep outside them."""
    log.info("collecting %d symbols every %.0fs", len(symbols), interval_s)
    while True:
        if not in_session():
            time.sleep(60)
            continue
        t0 = time.time()
        try:
            append_snapshot(snapshot(symbols), root)
        except Exception as e:  # noqa: BLE001
            log.warning("snapshot failed: %s", e)
        time.sleep(max(1.0, interval_s - (time.time() - t0)))
