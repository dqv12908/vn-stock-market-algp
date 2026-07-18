"""Event-study backtest: does a high accumulation score precede big moves?

Method (standard event study, fully causal):
  1. For every symbol, compute the daily composite score each day.
  2. An EVENT fires when the 5-day mean score crosses above `threshold`
     (with a `cooldown` so one campaign isn't counted ten times).
  3. Measure forward returns at +5/+10/+20 sessions from the event close.
  4. Compare against the unconditional (baseline) forward returns of the
     same universe — the lift over baseline is the information content.

Also reports hit-rate on "big moves" (forward max-gain >= big_move).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .signals.accumulation import accumulation_features
from .signals.composite import daily_score

log = logging.getLogger(__name__)

HORIZONS = (5, 10, 20)


def _events(score: pd.Series, threshold: float, cooldown: int) -> list[int]:
    ma = score.rolling(5).mean()
    idx, last = [], -10**9
    for i, v in enumerate(ma):
        if v is not None and not np.isnan(v) and v >= threshold and i - last >= cooldown:
            idx.append(i)
            last = i
    return idx


def event_study(data: dict[str, pd.DataFrame], threshold: float = 0.8,
                cooldown: int = 20, big_move: float = 0.15,
                warmup: int = 120) -> dict:
    ev_rows, base_rows = [], []
    for sym, df in data.items():
        if len(df) < warmup + max(HORIZONS):
            continue
        c = df["close"].astype(float).reset_index(drop=True)
        feats = accumulation_features(df.reset_index(drop=True))
        score = daily_score(feats)

        def fwd(i: int) -> dict | None:
            if i + max(HORIZONS) >= len(c):
                return None
            row = {f"ret_{h}d": c[i + h] / c[i] - 1 for h in HORIZONS}
            row["max_gain_20d"] = c[i + 1: i + 21].max() / c[i] - 1
            row["max_loss_20d"] = c[i + 1: i + 21].min() / c[i] - 1
            return row

        for i in _events(score.iloc[warmup:], threshold, cooldown):
            r = fwd(i + warmup)
            if r:
                ev_rows.append({"symbol": sym,
                                "date": df["time"].iloc[i + warmup],
                                "score": score.iloc[i + warmup], **r})
        # baseline: every 5th day after warmup
        for i in range(warmup, len(c) - max(HORIZONS), 5):
            r = fwd(i)
            if r:
                base_rows.append(r)

    ev = pd.DataFrame(ev_rows)
    base = pd.DataFrame(base_rows)
    if not len(ev):
        return {"n_events": 0}

    summary = {"n_events": len(ev), "n_baseline": len(base), "events": ev}
    for h in HORIZONS:
        summary[f"event_mean_{h}d"] = float(ev[f"ret_{h}d"].mean())
        summary[f"base_mean_{h}d"] = float(base[f"ret_{h}d"].mean())
        summary[f"event_median_{h}d"] = float(ev[f"ret_{h}d"].median())
        summary[f"win_rate_{h}d"] = float((ev[f"ret_{h}d"] > 0).mean())
    summary["big_move_hit"] = float((ev["max_gain_20d"] >= big_move).mean())
    summary["big_move_base"] = float((base["max_gain_20d"] >= big_move).mean())
    return summary


def print_summary(s: dict) -> None:
    if not s.get("n_events"):
        print("no events fired")
        return
    print(f"events: {s['n_events']}   baseline samples: {s['n_baseline']}")
    for h in HORIZONS:
        print(f"  +{h:>2}d  event mean {s[f'event_mean_{h}d']:+.2%}  "
              f"median {s[f'event_median_{h}d']:+.2%}  "
              f"win {s[f'win_rate_{h}d']:.0%}   baseline {s[f'base_mean_{h}d']:+.2%}")
    print(f"  big-move (>=15% max gain in 20d): events {s['big_move_hit']:.0%} "
          f"vs baseline {s['big_move_base']:.0%}")
