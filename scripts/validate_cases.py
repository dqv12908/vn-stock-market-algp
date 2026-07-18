#!/usr/bin/env python3
"""Validate the detector against DOCUMENTED manipulation cases.

For each episode in vnmm.cases (criminal convictions, UBCKNN fines,
documented run-ups), fetch history around the documented markup window
and answer: did the composite score (and model, if trained) fire in the
30 sessions BEFORE the documented markup start?

This is the strongest available ground truth — labels come from
prosecutions, not from price patterns, so there is no circularity with
the price-based labeling used in training.

Usage:
    python scripts/validate_cases.py [--pre 30] [--threshold 0.5]
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vnmm.data import loader                                  # noqa: E402
from vnmm.backtest import align_index                         # noqa: E402
from vnmm.cases import CASES                                  # noqa: E402
from vnmm.signals.accumulation import accumulation_features   # noqa: E402
from vnmm.signals.composite import daily_score                # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", type=int, default=30,
                    help="sessions before markup start to look for a signal")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--pause", type=float, default=5.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    rows = []
    for sym, markup_start, peak, tier, note in CASES:
        m0 = datetime.fromisoformat(markup_start)
        start = (m0 - timedelta(days=550)).date().isoformat()
        end = (datetime.fromisoformat(peak) + timedelta(days=30)).date().isoformat()
        try:
            df = loader.daily_history(sym, start, end)
            idx = loader.daily_history("VNINDEX", start, end)
        except Exception as e:  # noqa: BLE001
            rows.append({"symbol": sym, "tier": tier, "status": f"no data ({e})"})
            continue
        df = df.reset_index(drop=True)
        if len(df) < 150:
            rows.append({"symbol": sym, "tier": tier, "status": "short history"})
            continue
        bench = align_index(df, idx)
        feats = accumulation_features(df, benchmark=bench)
        score = daily_score(feats).rolling(5).mean()
        dates = pd.to_datetime(df["time"])
        pre_mask = (dates < pd.Timestamp(m0)) & \
                   (dates >= pd.Timestamp(m0) - pd.Timedelta(days=args.pre * 1.6))
        pre = score[pre_mask].tail(args.pre)
        if not len(pre):
            rows.append({"symbol": sym, "tier": tier, "status": "no pre-window"})
            continue
        rows.append({
            "symbol": sym, "tier": tier,
            "markup_start": markup_start,
            "pre_max_score": round(float(pre.max()), 2),
            "fired": bool(pre.max() >= args.threshold),
            "days_before": int((pre >= args.threshold).values[::-1].argmax())
                           if pre.max() >= args.threshold else None,
            "note": note, "status": "ok",
        })
        import time as _t
        _t.sleep(args.pause)

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    ok = out[out["status"] == "ok"]
    if len(ok):
        print(f"\ndetector fired before documented markup: "
              f"{ok['fired'].sum()}/{len(ok)} cases "
              f"({ok['fired'].mean():.0%}) at threshold {args.threshold}")
        for tier in ("criminal", "admin", "runup"):
            t = ok[ok["tier"] == tier]
            if len(t):
                print(f"  {tier:9s}: {t['fired'].sum()}/{len(t)}")


if __name__ == "__main__":
    main()
