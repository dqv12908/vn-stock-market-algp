"""Composite market-maker accumulation score.

One number per symbol per day in roughly [-3, +3] units (weighted z-space).
The score is deliberately a *linear* combination of interpretable features —
when an alert fires you can read exactly which distortion drove it, which
matters more than a few points of AUC from a black box.

Weights below were sanity-tuned on the 2025-2026 event study in
scripts/backtest_signals.py; retune with your own collected data.
"""

from __future__ import annotations

import pandas as pd

# feature -> weight. Positive = more accumulation-like.
DAILY_WEIGHTS: dict[str, float] = {
    "obv_div":     1.00,   # volume flowing in while price flat — core signal
    "absorption":  0.90,   # effort without result
    "cmf":         0.70,   # closes near highs on volume
    "vol_z":       0.50,   # volume waking up
    "turnover_z":  0.50,   # float churn
    "coil":        0.40,   # rising lows under flat lid
    "bb_squeeze":  0.40,   # compression before markup
    "shakeout":    0.60,   # engineered flush marker
}

TICK_WEIGHTS: dict[str, float] = {
    "ofi":                  1.00,
    "flow_persistence":     0.80,
    "block_share":          0.50,
    "auction_share":        0.40,
    "atc_gap":              0.60,
    # inverted features: LOW entropy / HIGH repeated-size share = suspicious
    "size_entropy":        -0.70,
    "top_size_share":       0.70,
}


def daily_score(features: pd.DataFrame,
                weights: dict[str, float] | None = None) -> pd.Series:
    w = weights or DAILY_WEIGHTS
    cols = [c for c in w if c in features.columns]
    score = sum(features[c].clip(-3, 3).fillna(0) * w[c] for c in cols)
    return score / sum(abs(w[c]) for c in cols)


def tick_score(feats: dict[str, float],
               weights: dict[str, float] | None = None) -> float:
    w = weights or TICK_WEIGHTS
    used = [k for k in w if k in feats]
    if not used:
        return 0.0
    # entropy is in [0,1]; center it so 'normal' tape ~0.85 scores ~0
    adj = dict(feats)
    if "size_entropy" in adj:
        adj["size_entropy"] = adj["size_entropy"] - 0.85
    raw = sum(max(-3.0, min(3.0, adj[k])) * w[k] for k in used)
    return raw / sum(abs(w[k]) for k in used)
