"""Daily-bar detectors for the accumulation phase of a driven move.

Theory
------
A Vietnamese "đội lái" (pump crew / market maker) campaign has a canonical
lifecycle borrowed straight from Wyckoff, compressed by the ±7%/±10%/±15%
daily bands and the T+2.5 settlement rule:

  1. ACCUMULATION  — weeks of quiet buying. Price is pinned in a range,
     volume migrates from sellers to the operator. Float is absorbed.
  2. SHAKEOUT      — one or two engineered flushes (often limit-down spikes)
     to trigger retail stops and margin calls, collecting the last loose shares.
  3. MARKUP        — the visible 20-50% run, often ceiling-limit chains
     (trần) where volume *vanishes* because nobody sells.
  4. DISTRIBUTION  — huge volume near highs, price stalls, closes weaken.

The tradeable information lives in phase 1-2, and it is measurable because
the operator cannot accumulate a meaningful % of the free float without
distorting volume/price mechanics. Every detector below scores one such
distortion on the *daily* tape. All features are causal (rolling, no
lookahead) and returned as columns aligned to the input index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _z(s: pd.Series, win: int) -> pd.Series:
    m = s.rolling(win, min_periods=win // 2).mean()
    sd = s.rolling(win, min_periods=win // 2).std()
    return (s - m) / (sd + EPS)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff().fillna(0)) * volume).cumsum()


def accumulation_features(df: pd.DataFrame, base_win: int = 60,
                          slope_win: int = 20,
                          benchmark: pd.Series | None = None) -> pd.DataFrame:
    """Compute the accumulation footprint feature set.

    Input df needs columns: open, high, low, close, volume (daily bars).

    benchmark: optional index close series aligned to df's rows (e.g.
    VNINDEX). When given, the price-driven features (OBV direction, OBV
    divergence, absorption "result") are computed on the stock's return IN
    EXCESS of the index, so a broad market rally does not read as
    stock-specific accumulation. Bar-structure features (CLV, coil,
    squeeze, shakeout) stay on raw bars — they describe the stock's own
    tape shape.
    """
    o, h, l, c, v = (df[k].astype(float) for k in
                     ("open", "high", "low", "close", "volume"))
    out = pd.DataFrame(index=df.index)

    # market-relative close: stock priced in units of the index. All
    # trend/divergence math below uses c_rel so index beta cancels out.
    if benchmark is not None:
        bench = benchmark.astype(float).reindex(df.index).ffill()
        c_rel = c / (bench + EPS)
    else:
        c_rel = c

    # --- 1. Volume anomaly -------------------------------------------------
    # z-score of log volume vs trailing 60d. Log because VN small-cap volume
    # is heavy-tailed; a raw z-score saturates on one big day.
    out["vol_z"] = _z(np.log1p(v), base_win)

    # --- 2. OBV divergence -------------------------------------------------
    # Operator accumulates: OBV trends up while price goes nowhere.
    # Slope of normalized OBV minus slope of normalized price over slope_win.
    obv_ = obv(c_rel, v)
    obv_n = (obv_ - obv_.rolling(base_win).mean()) / (obv_.rolling(base_win).std() + EPS)
    px_n = (c_rel - c_rel.rolling(base_win).mean()) / (c_rel.rolling(base_win).std() + EPS)
    t = np.arange(len(df), dtype=float)
    def _slope(s: pd.Series) -> pd.Series:
        # rolling OLS slope of s on time
        x = pd.Series(t, index=s.index)
        cov = s.rolling(slope_win).cov(x)
        var = x.rolling(slope_win).var()
        return cov / (var + EPS)
    out["obv_div"] = _slope(obv_n) - _slope(px_n)

    # --- 3. Effort vs Result (absorption) ---------------------------------
    # Wyckoff: big volume ("effort") that produces no price change ("result")
    # means someone is absorbing supply. result = |ret| in range units;
    # effort = volume vs average. High effort_z with low result => absorption.
    ret = c_rel.pct_change().abs()
    atr_pct = ((h - l) / c.shift(1)).rolling(base_win).mean()
    result = ret / (atr_pct + EPS)
    effort = v / (v.rolling(base_win).mean() + EPS)
    raw_absorb = effort * np.exp(-2.0 * result)   # decays fast once price moves
    out["absorption"] = _z(raw_absorb, base_win)

    # --- 4. Close location value on volume ---------------------------------
    # Accumulation days close near the high of the bar. CLV in [-1, 1],
    # volume-weighted and smoothed => Chaikin-style money flow.
    clv = ((c - l) - (h - c)) / (h - l + EPS)
    out["cmf"] = (clv * v).rolling(slope_win).sum() / (v.rolling(slope_win).sum() + EPS)

    # --- 5. Volatility compression (pre-markup coil) ------------------------
    # Bollinger bandwidth percentile: crews mark up from tight coils because
    # a tight float needs less cash to move. Low percentile = compressed.
    bw = c.rolling(slope_win).std() / (c.rolling(slope_win).mean() + EPS)
    out["bb_squeeze"] = 1.0 - bw.rolling(252, min_periods=base_win).rank(pct=True)

    # --- 6. Float turnover --------------------------------------------------
    # Cumulative 20d volume as multiple of its own 1y norm. Crews must churn
    # the float; sustained elevated turnover in a flat price is the tell.
    turn20 = v.rolling(slope_win).sum()
    out["turnover_z"] = _z(np.log1p(turn20), 252)

    # --- 7. Higher-low structure under a flat lid ---------------------------
    # Classic absorption picture: resistance flat, lows rising.
    hi_slope = _slope((h - h.rolling(base_win).mean()) / (h.rolling(base_win).std() + EPS))
    lo_slope = _slope((l - l.rolling(base_win).mean()) / (l.rolling(base_win).std() + EPS))
    out["coil"] = lo_slope - hi_slope.clip(lower=0)

    # --- 8. Shakeout marker -------------------------------------------------
    # Engineered flush: an outsized down day on high volume that closes well
    # off its low, inside a period of otherwise compressed volatility.
    down_spike = (-c.pct_change()) / (atr_pct + EPS)
    recover = (c - l) / (h - l + EPS)
    out["shakeout"] = ((down_spike > 1.5) & (recover > 0.5) &
                       (out["vol_z"] > 1.0)).astype(float)

    return out
