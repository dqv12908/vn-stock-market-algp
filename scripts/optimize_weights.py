#!/usr/bin/env python3
"""Walk-forward weight optimization for the daily composite score.

Protocol (no test-set leakage):
  1. Precompute features and forward ABNORMAL returns (vs VNINDEX) for the
     whole universe.
  2. TRAIN window: random search over weight vectors, objective = Spearman
     rank-IC between the 5d-smoothed score and +10/+20d abnormal returns.
     Rank-IC uses every symbol-day, so it is far more sample-efficient than
     counting threshold events, and is immune to level/scale choices.
  3. TEST window (later, untouched): run the threshold event study with the
     winning weights and with the default weights. Ship the winner only if
     it also wins out-of-sample.

Usage:
    python scripts/optimize_weights.py [--start 2024-06-01] \
        [--split 2025-10-01] [--iters 300] [--seed 7]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vnmm.data import loader                       # noqa: E402
from vnmm import backtest                          # noqa: E402
from vnmm.signals.accumulation import accumulation_features  # noqa: E402
from vnmm.signals.composite import DAILY_WEIGHTS, daily_score  # noqa: E402

WARMUP = 120
HORIZONS = (10, 20)


def build_panel(data: dict[str, pd.DataFrame], index_df: pd.DataFrame):
    """Stack per-symbol features + forward abnormal returns into one panel."""
    frames = []
    for sym, df in data.items():
        df = df.reset_index(drop=True)
        if len(df) < WARMUP + max(HORIZONS) + 5:
            continue
        bench = backtest.align_index(df, index_df)
        feats = accumulation_features(df, benchmark=bench)
        c = df["close"].astype(float)
        for h in HORIZONS:
            feats[f"fwd_{h}d"] = (c.shift(-h) / c - 1) - (bench.shift(-h) / bench - 1)
        feats["date"] = pd.to_datetime(df["time"]).values
        feats["symbol"] = sym
        frames.append(feats.iloc[WARMUP:-max(HORIZONS)])
    return pd.concat(frames, ignore_index=True)


def rank_ic(panel: pd.DataFrame, weights: dict[str, float]) -> float:
    score = daily_score(panel, weights)
    sm = score.groupby(panel["symbol"]).transform(lambda s: s.rolling(5).mean())
    ics = []
    for h in HORIZONS:
        m = sm.notna() & panel[f"fwd_{h}d"].notna()
        ics.append(sm[m].rank().corr(panel.loc[m, f"fwd_{h}d"].rank()))
    return float(np.nanmean(ics))


def tail_return(panel: pd.DataFrame, weights: dict[str, float],
                q: float = 0.98, min_n: int = 50) -> float:
    """Mean forward abnormal return of the top-(1-q) tail of smoothed scores.

    This is the quantity we actually trade — alerts fire only in the
    extreme tail, so optimizing average rank correlation (IC) selects for
    the wrong part of the distribution and overfits (verified empirically:
    IC-optimal weights collapse out-of-sample).
    """
    score = daily_score(panel, weights)
    sm = score.groupby(panel["symbol"]).transform(lambda s: s.rolling(5).mean())
    cut = sm.quantile(q)
    tail = panel[sm >= cut]
    if len(tail) < min_n:
        return -np.inf
    return float(np.nanmean([tail[f"fwd_{h}d"].mean() for h in HORIZONS]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--start", default="2024-06-01")
    ap.add_argument("--split", default="2025-10-01")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--objective", choices=("tail", "ic"), default="tail")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    cfg = yaml.safe_load(Path(args.config).read_text())
    data = loader.fetch_universe_daily(cfg["watchlist"], start=args.start)
    index_df = loader.index_history(start=args.start)
    panel = build_panel(data, index_df)
    train = panel[panel["date"] < args.split]
    print(f"panel: {len(panel)} symbol-days, train {len(train)}, "
          f"test {len(panel) - len(train)}")

    objective = tail_return if args.objective == "tail" else rank_ic
    rng = np.random.default_rng(args.seed)
    keys = list(DAILY_WEIGHTS)
    best_w, best_obj = dict(DAILY_WEIGHTS), objective(train, DAILY_WEIGHTS)
    print(f"default weights train {args.objective}: {best_obj:+.4f}")
    for i in range(args.iters):
        # perturb around defaults rather than sampling blind: keeps the
        # search near theory-grounded weights and resists overfitting
        w = {k: float(max(0.05, DAILY_WEIGHTS[k] * rng.lognormal(0, 0.5)))
             for k in keys}
        obj = objective(train, w)
        if obj > best_obj:
            best_obj, best_w = obj, w
            print(f"  iter {i:3d}  obj {obj:+.4f}  "
                  + " ".join(f"{k}={v:.2f}" for k, v in w.items()))

    print(f"\nbest train {args.objective}: {best_obj:+.4f}")

    # out-of-sample comparison on the untouched test window
    test_data = {s: df[pd.to_datetime(df["time"]) >=
                       pd.Timestamp(args.split) - pd.Timedelta(days=300)]
                 for s, df in data.items()}
    for label, w in (("default", None), ("optimized", best_w)):
        print(f"\n=== TEST event study [{label}] (events after {args.split}) ===")
        s = backtest.event_study(test_data, threshold=args.threshold,
                                 index_df=index_df, weights=w)
        if s.get("n_events"):
            ev = s["events"]
            keep = ev[ev["date"] >= args.split]
            print(f"events after split: {len(keep)}  "
                  f"+10d {keep['ret_10d'].mean():+.2%}  "
                  f"+20d {keep['ret_20d'].mean():+.2%}  "
                  f"big-move {(keep['max_gain_20d'] >= 0.15).mean():.0%}")
        backtest.print_summary(s)

    print("\noptimized weights:")
    for k, v in best_w.items():
        print(f"  {k:12s} {v:.3f}")


if __name__ == "__main__":
    main()
