#!/usr/bin/env python3
"""Continuously snapshot the top-10 order book for the watchlist.

Leave this running on a server during trading hours (it sleeps outside
09:00-11:30 / 13:00-14:45 ICT). Snapshots land in data/orderbook/DATE/SYM.parquet.

Usage:
    python scripts/collect_orderbook.py [--config config.yaml] [--interval 15]
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vnmm.data import orderbook_collector  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--interval", type=float, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    interval = args.interval or cfg.get("collector", {}).get("interval_seconds", 15)
    orderbook_collector.run(cfg["watchlist"], interval_s=float(interval))


if __name__ == "__main__":
    main()
