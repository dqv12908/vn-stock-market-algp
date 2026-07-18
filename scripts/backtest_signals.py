#!/usr/bin/env python3
"""Event-study backtest of the daily accumulation score.

Usage:
    python scripts/backtest_signals.py [--config config.yaml] \
        [--start 2024-01-01] [--threshold 0.8]
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vnmm.data import loader  # noqa: E402
from vnmm import backtest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--show-events", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    data = loader.fetch_universe_daily(cfg["watchlist"], start=args.start)
    print(f"loaded {len(data)} symbols from {args.start}")

    summary = backtest.event_study(
        data, threshold=args.threshold,
        big_move=cfg["thresholds"].get("backtest_big_move", 0.15))
    backtest.print_summary(summary)
    if args.show_events and summary.get("n_events"):
        ev = summary["events"].copy()
        ev["date"] = ev["date"].dt.date
        print(ev.to_string(index=False))


if __name__ == "__main__":
    main()
