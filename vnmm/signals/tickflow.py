"""Intraday tick-level detectors.

Input is the match-by-match tape from VCI (`intraday_ticks`): every print
carries price, volume and the aggressor side (Buy = lifted the ask,
Sell = hit the bid, plus ATO/ATC auction prints). This is where operator
behaviour is most visible, because a crew has to *execute* — and execution
leaves shape on the tape that organic retail flow does not have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def tick_features(ticks: pd.DataFrame) -> dict[str, float]:
    """Summarize one session's tape into a feature dict.

    Expects columns: time, price, volume, match_type.
    """
    df = ticks.copy()
    df = df[df["volume"] > 0]
    if len(df) < 30:
        return {}

    is_buy = df["match_type"].eq("Buy")
    is_sell = df["match_type"].eq("Sell")
    is_auction = df["match_type"].isin(("ATO", "ATC"))
    total_vol = float(df["volume"].sum())

    feats: dict[str, float] = {}

    # --- 1. Order-flow imbalance -------------------------------------------
    # OFI = (aggressive buy vol - aggressive sell vol) / matched vol.
    # Sustained positive OFI with flat price = someone absorbing offers
    # without letting the price run yet.
    bvol = float(df.loc[is_buy, "volume"].sum())
    svol = float(df.loc[is_sell, "volume"].sum())
    feats["ofi"] = (bvol - svol) / (bvol + svol + EPS)

    # --- 2. Block-trade share ----------------------------------------------
    # Fraction of volume in prints >= 95th percentile print size. Retail flow
    # is many small prints; an operator working size shows up in the tail.
    p95 = df["volume"].quantile(0.95)
    feats["block_share"] = float(df.loc[df["volume"] >= p95, "volume"].sum()) / (total_vol + EPS)

    # --- 3. Print-size entropy (wash-trade / painting signature) ------------
    # Crews cross volume with themselves to fake liquidity, and automation
    # reuses the same lot sizes. Organic tape has high size diversity.
    # Normalized Shannon entropy of the print-size histogram: LOW = suspicious.
    sizes = df["volume"].value_counts(normalize=True)
    ent = float(-(sizes * np.log(sizes + EPS)).sum())
    feats["size_entropy"] = ent / (np.log(len(sizes)) + EPS)

    # --- 4. Repeated-size dominance ----------------------------------------
    # Share of total volume carried by the single most common print size
    # (excluding the 100-share odd lot). Direct wash-trade fingerprint.
    by_size = df.groupby("volume")["volume"].sum()
    by_size = by_size[by_size.index > 100]
    feats["top_size_share"] = float(by_size.max()) / (total_vol + EPS) if len(by_size) else 0.0

    # --- 5. Auction share (marking the close) -------------------------------
    # ATC is the cheapest place to set the reference price for tomorrow's
    # band. Outsized ATC volume, especially with an ATC price jump, is the
    # classic mark. Normal ATC share on HOSE is ~3-8% of the day.
    feats["auction_share"] = float(df.loc[is_auction, "volume"].sum()) / (total_vol + EPS)
    cont = df[~is_auction]
    atc = df[df["match_type"].eq("ATC")]
    if len(atc) and len(cont):
        last_cont = float(cont["price"].iloc[-1])
        feats["atc_gap"] = (float(atc["price"].iloc[-1]) - last_cont) / (last_cont + EPS)
    else:
        feats["atc_gap"] = 0.0

    # --- 6. Buy-pressure persistence ----------------------------------------
    # Autocorrelation of signed volume in 5-minute buckets. Operator programs
    # execute steadily; organic flow mean-reverts. High persistence = working
    # a parent order.
    df5 = df.set_index("time")
    signed = (df5["volume"] * np.where(df5["match_type"].eq("Buy"), 1,
              np.where(df5["match_type"].eq("Sell"), -1, 0))).resample("5min").sum()
    signed = signed[signed != 0]
    feats["flow_persistence"] = float(signed.autocorr(1)) if len(signed) > 8 else 0.0

    # --- 7. Intraday effort/result ------------------------------------------
    px = df["price"].astype(float)
    rng = (px.max() - px.min()) / (px.iloc[0] + EPS)
    feats["intraday_absorption"] = float(np.log1p(total_vol)) * float(np.exp(-50 * rng))

    return feats
