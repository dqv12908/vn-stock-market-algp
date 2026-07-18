"""Event-study backtest: does a high accumulation score precede big moves
IN EXCESS OF THE MARKET?

Method (standard abnormal-return event study, fully causal):
  1. For every symbol, compute the daily composite score each day —
     with price features measured relative to VNINDEX when supplied.
  2. An EVENT fires when the 5-day mean score crosses above `threshold`
     (with a `cooldown` so one campaign isn't counted ten times).
  3. Measure ABNORMAL forward returns at +5/+10/+20 sessions:
     stock return minus VNINDEX return over the same calendar window.
     This removes the "everything went up" confound — a broad rally
     contributes identically to event and benchmark legs and cancels.
  4. Compare against the unconditional abnormal forward returns of the
     same universe (baseline). Any residual baseline drift (small-cap
     beta vs the index) is visible in the table, and the reported lift
     is event-minus-baseline, not event-minus-zero.

Also reports hit-rate on "big moves" (forward max ABNORMAL gain >= big_move).
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


def align_index(df: pd.DataFrame, index_df: pd.DataFrame | None) -> pd.Series | None:
    """Index close aligned to the stock's rows by calendar date."""
    if index_df is None:
        return None
    ic = index_df.set_index(pd.to_datetime(index_df["time"]).dt.normalize())["close"]
    ic = ic[~ic.index.duplicated()]
    dates = pd.to_datetime(df["time"]).dt.normalize()
    return pd.Series(ic.reindex(dates).ffill().values, index=df.index, dtype=float)


def event_study(data: dict[str, pd.DataFrame], threshold: float = 0.6,
                cooldown: int = 20, big_move: float = 0.15,
                warmup: int = 120, index_df: pd.DataFrame | None = None,
                weights: dict[str, float] | None = None) -> dict:
    ev_rows, base_rows = [], []
    for sym, df in data.items():
        if len(df) < warmup + max(HORIZONS):
            continue
        df = df.reset_index(drop=True)
        c = df["close"].astype(float)
        bench = align_index(df, index_df)
        feats = accumulation_features(df, benchmark=bench)
        score = daily_score(feats, weights)

        def fwd(i: int) -> dict | None:
            if i + max(HORIZONS) >= len(c):
                return None
            row = {}
            for h in HORIZONS:
                r = c[i + h] / c[i] - 1
                if bench is not None:
                    r -= bench[i + h] / bench[i] - 1
                row[f"ret_{h}d"] = r
            gain = c[i + 1: i + 21].max() / c[i] - 1
            loss = c[i + 1: i + 21].min() / c[i] - 1
            if bench is not None:
                mret = bench[i + 20] / bench[i] - 1
                gain, loss = gain - mret, loss - mret
            row["max_gain_20d"], row["max_loss_20d"] = gain, loss
            return row

        for i in _events(score.iloc[warmup:], threshold, cooldown):
            r = fwd(i + warmup)
            if r:
                ev_rows.append({"symbol": sym,
                                "date": df["time"].iloc[i + warmup],
                                "score": score.iloc[i + warmup], **r})
        # baseline: every 5th day after warmup, same abnormalization
        for i in range(warmup, len(c) - max(HORIZONS), 5):
            r = fwd(i)
            if r:
                base_rows.append(r)

    ev = pd.DataFrame(ev_rows)
    base = pd.DataFrame(base_rows)
    if not len(ev):
        return {"n_events": 0}

    summary = {"n_events": len(ev), "n_baseline": len(base), "events": ev,
               "abnormal": index_df is not None}
    for h in HORIZONS:
        summary[f"event_mean_{h}d"] = float(ev[f"ret_{h}d"].mean())
        summary[f"base_mean_{h}d"] = float(base[f"ret_{h}d"].mean())
        summary[f"event_median_{h}d"] = float(ev[f"ret_{h}d"].median())
        summary[f"win_rate_{h}d"] = float((ev[f"ret_{h}d"] > 0).mean())
        # t-stat of event mean vs baseline mean (independent, unequal var)
        se = np.sqrt(ev[f"ret_{h}d"].var() / len(ev) +
                     base[f"ret_{h}d"].var() / len(base))
        summary[f"tstat_{h}d"] = float(
            (ev[f"ret_{h}d"].mean() - base[f"ret_{h}d"].mean()) / (se + 1e-12))
    summary["big_move_hit"] = float((ev["max_gain_20d"] >= big_move).mean())
    summary["big_move_base"] = float((base["max_gain_20d"] >= big_move).mean())
    return summary


def print_summary(s: dict) -> None:
    if not s.get("n_events"):
        print("no events fired")
        return
    kind = "ABNORMAL (vs VNINDEX)" if s.get("abnormal") else "RAW"
    print(f"returns: {kind}   events: {s['n_events']}   "
          f"baseline samples: {s['n_baseline']}")
    for h in HORIZONS:
        print(f"  +{h:>2}d  event mean {s[f'event_mean_{h}d']:+.2%}  "
              f"median {s[f'event_median_{h}d']:+.2%}  "
              f"win {s[f'win_rate_{h}d']:.0%}   "
              f"baseline {s[f'base_mean_{h}d']:+.2%}   "
              f"t={s[f'tstat_{h}d']:+.1f}")
    print(f"  big-move (>=15% max abnormal gain in 20d): events {s['big_move_hit']:.0%} "
          f"vs baseline {s['big_move_base']:.0%}")
