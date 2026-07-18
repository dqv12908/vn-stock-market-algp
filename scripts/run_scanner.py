#!/usr/bin/env python3
"""Run the daily + tick scan over the watchlist and print/send alerts.

Usage:
    python scripts/run_scanner.py [--config config.yaml] [--no-ticks]
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vnmm import scanner, alerts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-ticks", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    syms = cfg["watchlist"]

    daily = scanner.scan_daily(syms)
    print("\n=== DAILY ACCUMULATION SCORES ===")
    if len(daily):
        cols = ["symbol", "date", "close", "score", "score_ma5",
                "obv_div", "absorption", "cmf", "vol_z"]
        if "p_onset" in daily.columns:
            cols.insert(3, "p_onset")
        print(daily[cols].to_string(index=False))

    tick = None
    if not args.no_ticks:
        tick = scanner.scan_ticks(syms)
        print("\n=== TAPE (TICK) SCORES ===")
        if len(tick):
            print(tick.to_string(index=False))

    msg = alerts.format_alerts(daily, cfg["thresholds"]["daily_score"],
                               tick, cfg["thresholds"]["tick_score"])
    print("\n=== ALERTS ===")
    alerts.send(msg)


if __name__ == "__main__":
    main()
