"""Ground-truth labeling: markup onsets.

An "opportunity" is the START of a sustained abnormal run — the moment
phase 3 (markup) begins. We label it mechanically so precision/recall of
the alert system can be measured:

  Onset at bar i if:
    * forward max abnormal gain within `horizon` sessions >= `min_gain`
      (computed close-to-max, market-adjusted), AND
    * the run had not already started: trailing `quiet` -session abnormal
      return < `pre_run_cap` (we only credit catching the BEGINNING —
      alerting mid-pump is chasing, not predicting), AND
    * first bar of the episode (no other onset in the previous `horizon`).

This is the same labeling logic used by the crypto pump-detection
literature, adapted to daily bars and VN price bands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def markup_onsets(df: pd.DataFrame, bench: pd.Series | None = None,
                  min_gain: float = 0.20, horizon: int = 15,
                  quiet: int = 10, pre_run_cap: float = 0.06) -> list[int]:
    c = df["close"].astype(float).reset_index(drop=True)
    if bench is not None:
        b = bench.reset_index(drop=True)
        rel = c / b
    else:
        rel = c
    n = len(c)
    onsets, last = [], -10**9
    for i in range(quiet, n - horizon):
        fwd_max = rel[i + 1: i + 1 + horizon].max() / rel[i] - 1
        trail = rel[i] / rel[i - quiet] - 1
        if fwd_max >= min_gain and trail < pre_run_cap and i - last >= horizon:
            onsets.append(i)
            last = i
    return onsets


def alert_pr(alert_idx: list[int], onset_idx: list[int],
             lead: int = 12, n_bars: int | None = None) -> dict:
    """Precision / recall / lead-time of alerts vs onsets.

    An alert is a TRUE POSITIVE if an onset occurs within [0, lead]
    sessions after it. An onset is CAUGHT if any alert fired within the
    `lead` sessions before it (or on the day).
    """
    alerts, onsets = sorted(alert_idx), sorted(onset_idx)
    tp_alerts = [a for a in alerts if any(0 <= o - a <= lead for o in onsets)]
    caught = [o for o in onsets if any(0 <= o - a <= lead for a in alerts)]
    leads = [min(o - a for a in alerts if 0 <= o - a <= lead) for o in caught]
    return {
        "n_alerts": len(alerts), "n_onsets": len(onsets),
        "precision": len(tp_alerts) / len(alerts) if alerts else np.nan,
        "recall": len(caught) / len(onsets) if onsets else np.nan,
        "median_lead": float(np.median(leads)) if leads else np.nan,
    }
