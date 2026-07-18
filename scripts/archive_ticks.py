#!/usr/bin/env python3
"""Archive today's full match-by-match tape for the watchlist.

Run once after each close (e.g. 15:15 ICT via cron). Over weeks this
builds the historical tick dataset that cannot be bought cheaply —
enabling backtests of the tape features (rush bursts, wash battery,
OFI persistence) instead of single-day snapshots.

Storage: data/ticks/YYYY-MM-DD/SYMBOL.parquet
Cron:    15 15 * * 1-5  cd /path/to/repo && python scripts/archive_ticks.py
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vnmm.data import loader  # noqa: E402

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="data/ticks")
    ap.add_argument("--pause", type=float, default=3.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    day = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).date().isoformat()
    root = Path(args.out) / day
    root.mkdir(parents=True, exist_ok=True)

    ok = 0
    for sym in cfg["watchlist"]:
        out = root / f"{sym}.parquet"
        if out.exists():
            continue
        try:
            ticks = loader.intraday_ticks(sym)
            if len(ticks):
                # only archive if the tape is actually from `day`
                tape_day = ticks["time"].dt.date.iloc[-1].isoformat()
                if tape_day == day:
                    ticks.to_parquet(out, index=False)
                    ok += 1
                else:
                    log.info("%s tape is from %s, skipping", sym, tape_day)
        except Exception as e:  # noqa: BLE001
            log.warning("%s failed: %s", sym, e)
        time.sleep(args.pause)
    log.info("archived %d/%d symbols -> %s", ok, len(cfg["watchlist"]), root)


if __name__ == "__main__":
    main()
