#!/usr/bin/env python3
"""Precision/recall evaluation of alert policies vs labeled markup onsets.

Policies compared:
  * threshold:      alert when 5d-mean score >= t (per symbol, absolute)
  * topk:           alert when score is in the day's cross-sectional top-k
                    AND above a soft floor (relative — adapts to regimes)
  * gated:          threshold AND breakout_prox > 0 (coiled near the lid)

Metrics: precision (alerts followed by an onset within `lead` sessions),
recall (onsets preceded by an alert), median lead time in sessions.

Usage:
    python scripts/evaluate_alerts.py [--start 2024-06-01] [--lead 12]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vnmm.data import loader                                  # noqa: E402
from vnmm.backtest import align_index                         # noqa: E402
from vnmm.labels import markup_onsets, alert_pr               # noqa: E402
from vnmm.signals.accumulation import accumulation_features   # noqa: E402
from vnmm.signals.composite import daily_score                # noqa: E402

WARMUP = 120


def build(data, index_df):
    """Per-symbol: smoothed score series, feature frame, onset indices."""
    out = {}
    for sym, df in data.items():
        df = df.reset_index(drop=True)
        if len(df) < WARMUP + 40:
            continue
        bench = align_index(df, index_df)
        feats = accumulation_features(df, benchmark=bench)
        score = daily_score(feats).rolling(5).mean()
        onsets = [i for i in markup_onsets(df, bench) if i >= WARMUP]
        out[sym] = {"df": df, "score": score, "feats": feats, "onsets": onsets}
    return out


def pooled_pr(univ, alert_fn, lead):
    """Aggregate TP/FP/FN across symbols for a per-symbol alert index fn."""
    tp_a = fp_a = caught = total_on = 0
    leads = []
    for sym, u in univ.items():
        alerts = alert_fn(sym, u)
        r = alert_pr(alerts, u["onsets"], lead=lead)
        n_tp = round((r["precision"] if r["precision"] == r["precision"] else 0)
                     * r["n_alerts"])
        tp_a += n_tp
        fp_a += r["n_alerts"] - n_tp
        caught += round((r["recall"] if r["recall"] == r["recall"] else 0)
                        * r["n_onsets"])
        total_on += r["n_onsets"]
        if r["median_lead"] == r["median_lead"]:
            leads.append(r["median_lead"])
    n_alerts = tp_a + fp_a
    return {"alerts": n_alerts, "onsets": total_on,
            "precision": tp_a / n_alerts if n_alerts else float("nan"),
            "recall": caught / total_on if total_on else float("nan"),
            "median_lead": float(np.median(leads)) if leads else float("nan")}


def dedup(idx, cooldown=10):
    keep, last = [], -10**9
    for i in idx:
        if i - last >= cooldown:
            keep.append(i)
            last = i
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--start", default="2024-06-01")
    ap.add_argument("--lead", type=int, default=12)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    cfg = yaml.safe_load(Path(args.config).read_text())
    data = loader.fetch_universe_daily(cfg["watchlist"], start=args.start)
    index_df = loader.index_history(start=args.start)
    univ = build(data, index_df)
    n_onsets = sum(len(u["onsets"]) for u in univ.values())
    print(f"universe: {len(univ)} symbols, {n_onsets} labeled markup onsets\n")

    # assemble cross-sectional score panel for topk policy
    panel = pd.DataFrame({s: u["score"] for s, u in univ.items()})

    def thresh(t):
        return lambda sym, u: dedup(
            [i for i in range(WARMUP, len(u["score"]))
             if u["score"][i] == u["score"][i] and u["score"][i] >= t])

    def topk(k, floor):
        rank = panel.rank(axis=1, ascending=False)
        return lambda sym, u: dedup(
            [i for i in range(WARMUP, len(u["score"]))
             if u["score"][i] == u["score"][i] and u["score"][i] >= floor
             and sym in rank.columns and rank[sym].iloc[i] <= k])

    def gated(t):
        base = thresh(t)
        return lambda sym, u: [i for i in base(sym, u)
                               if u["feats"]["breakout_prox"].iloc[i] > 0]

    print(f"{'policy':22s} {'alerts':>7s} {'precision':>10s} {'recall':>7s} {'lead':>5s}")
    for name, fn in [
        ("threshold 0.5", thresh(0.5)), ("threshold 0.6", thresh(0.6)),
        ("threshold 0.7", thresh(0.7)),
        ("top3 floor 0.3", topk(3, 0.3)), ("top2 floor 0.4", topk(2, 0.4)),
        ("gated 0.5", gated(0.5)), ("gated 0.6", gated(0.6)),
    ]:
        r = pooled_pr(univ, fn, args.lead)
        print(f"{name:22s} {r['alerts']:7d} {r['precision']:10.0%} "
              f"{r['recall']:7.0%} {r['median_lead']:5.1f}")


if __name__ == "__main__":
    main()
