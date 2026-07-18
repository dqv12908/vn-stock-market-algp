"""Watchlist scanner: fetch data, compute scores, emit alerts.

Two passes:
  * daily pass  — accumulation footprint on daily bars (run after close, or
                  anytime; uses ~1.5y of history per symbol)
  * tick pass   — today's tape microstructure (run near/after close)
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import pandas as pd

from .backtest import align_index
from .data import loader
from .signals.accumulation import accumulation_features
from .signals.tickflow import tick_features
from .signals.composite import daily_score, tick_score

log = logging.getLogger(__name__)


def scan_daily(symbols: list[str], lookback_days: int = 550,
               use_cache: bool = True, pause: float = 2.0) -> pd.DataFrame:
    # cache files are keyed by (start, end=today), so cached reads are at
    # most one session old — fine for a daily scan and safe from the
    # ~60 req/min limit that kills the process on repeated full fetches.
    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    try:
        index_df = loader.index_history(start=start, use_cache=use_cache)
    except Exception as e:  # noqa: BLE001
        log.warning("VNINDEX fetch failed (%s); scoring without benchmark", e)
        index_df = None
    rows = []
    for sym in symbols:
        try:
            cached = loader.CACHE_DIR / f"daily_{sym}_{start}_{date.today().isoformat()}.parquet"
            if not cached.exists():
                time.sleep(pause)
            df = loader.daily_history(sym, start, use_cache=use_cache)
            if len(df) < 120:
                continue
            # skip suspended/delisted symbols serving stale bars
            if (pd.Timestamp.today() - df["time"].iloc[-1]).days > 7:
                log.info("skip %s: stale (last bar %s)", sym, df["time"].iloc[-1].date())
                continue
            feats = accumulation_features(
                df.reset_index(drop=True),
                benchmark=align_index(df.reset_index(drop=True), index_df))
            score = daily_score(feats)
            last = feats.iloc[-1]
            rows.append({
                "symbol": sym,
                "date": df["time"].iloc[-1].date(),
                "close": df["close"].iloc[-1],
                "score": round(float(score.iloc[-1]), 3),
                "score_ma5": round(float(score.rolling(5).mean().iloc[-1]), 3),
                **{k: round(float(last[k]), 3) for k in feats.columns},
            })
        except Exception as e:  # noqa: BLE001
            log.warning("daily scan %s failed: %s", sym, e)
    out = pd.DataFrame(rows)
    return out.sort_values("score", ascending=False).reset_index(drop=True) if len(out) else out


def scan_ticks(symbols: list[str], pause: float = 2.0) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        try:
            time.sleep(pause)
            ticks = loader.intraday_ticks(sym)
            feats = tick_features(ticks)
            if not feats:
                continue
            rows.append({"symbol": sym,
                         "tick_score": round(tick_score(feats), 3),
                         **{k: round(v, 4) for k, v in feats.items()}})
        except Exception as e:  # noqa: BLE001
            log.warning("tick scan %s failed: %s", sym, e)
    out = pd.DataFrame(rows)
    return out.sort_values("tick_score", ascending=False).reset_index(drop=True) if len(out) else out
