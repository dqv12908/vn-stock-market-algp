"""Order-book (depth) detectors.

Works on snapshot rows produced by `orderbook_collector` — top-10 bid/ask
levels plus session stats from VCI's price board, sampled every N seconds.
With a few days of collected snapshots per symbol these become time series
and the interesting dynamics (walls appearing/vanishing, absorption) fall
out of first differences between snapshots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12
LEVELS = 10


def _side(df: pd.DataFrame, side: str, field: str) -> pd.DataFrame:
    cols = [f"bid_ask_{side}_{i}_{field}" for i in range(1, LEVELS + 1)]
    have = [c for c in cols if c in df.columns]
    return df[have].astype(float)


def depth_features(snaps: pd.DataFrame) -> pd.DataFrame:
    """Per-snapshot depth features for one symbol's snapshot time series."""
    out = pd.DataFrame(index=snaps.index)
    bidv = _side(snaps, "bid", "volume")
    askv = _side(snaps, "ask", "volume")
    bidp = _side(snaps, "bid", "price")
    askp = _side(snaps, "ask", "price")

    tb, ta = bidv.sum(axis=1), askv.sum(axis=1)

    # --- 1. Depth imbalance -------------------------------------------------
    # DI in [-1, 1]. Persistently positive DI while price is flat = support
    # being built; a sudden flip negative after a run = distribution.
    out["depth_imbalance"] = (tb - ta) / (tb + ta + EPS)

    # --- 2. Near-touch imbalance (levels 1-3 weighted) ----------------------
    # What actually moves the next print. Exponential weights favor L1.
    w = np.exp(-0.5 * np.arange(LEVELS))
    nb = (bidv * w).sum(axis=1)
    na = (askv * w).sum(axis=1)
    out["touch_imbalance"] = (nb - na) / (nb + na + EPS)

    # --- 3. Wall detection --------------------------------------------------
    # A single level holding an outsized share of its side's volume.
    out["bid_wall"] = bidv.max(axis=1) / (tb + EPS)
    out["ask_wall"] = askv.max(axis=1) / (ta + EPS)

    # --- 4. Book slope ------------------------------------------------------
    # Liquidity decay away from touch. A crew supporting price makes the bid
    # side unusually deep/flat; a thin ask slope means cheap to mark up.
    out["bid_slope"] = bidv.mean(axis=1) / (bidv.iloc[:, 0] + EPS)
    out["ask_slope"] = askv.mean(axis=1) / (askv.iloc[:, 0] + EPS)

    # --- 5. Spread in ticks -------------------------------------------------
    out["spread_pct"] = (askp.iloc[:, 0] - bidp.iloc[:, 0]) / (bidp.iloc[:, 0] + EPS)

    if "match_match_price" in snaps.columns:
        out["price"] = snaps["match_match_price"].astype(float)
    if "match_accumulated_volume" in snaps.columns:
        out["cum_volume"] = snaps["match_accumulated_volume"].astype(float)
    return out


def dynamic_features(feat: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Second-order signals from the *evolution* of the book across snapshots.

    window is in snapshots (e.g. 60 snapshots @ 15s = 15 minutes).
    """
    out = pd.DataFrame(index=feat.index)

    # --- 1. Spoofing / vanishing-wall score ---------------------------------
    # Big ask walls that disappear WITHOUT the cumulative volume advancing
    # (i.e. cancelled, not eaten) while price then rises: layering to scare
    # sellers out, then pulling. We approximate "eaten" via traded volume.
    cum = feat.get("cum_volume", pd.Series(0.0, index=feat.index))
    dvol = cum.diff().fillna(0)
    ask_wall_drop = (-feat["ask_wall"].diff()).clip(lower=0)
    vol_norm = dvol / (dvol.rolling(window).mean() + EPS)
    out["spoof_score"] = (ask_wall_drop * np.exp(-vol_norm)).rolling(window).sum()

    # --- 2. Absorption score ------------------------------------------------
    # Heavy traded volume while price stands still and the ask side refills:
    # someone is eating everything offered. Precedes markup.
    if "price" in feat.columns:
        px_still = np.exp(-200 * feat["price"].pct_change().abs().fillna(0))
        out["absorption_score"] = (dvol * px_still).rolling(window).sum() / \
                                  (dvol.rolling(window).sum() + EPS)
    # --- 3. Imbalance trend -------------------------------------------------
    out["di_trend"] = feat["depth_imbalance"].rolling(window).mean()
    out["di_flip"] = feat["depth_imbalance"].rolling(window).mean().diff(window)
    return out
