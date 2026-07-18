#!/usr/bin/env python3
"""Walk-forward training/evaluation of the onset model, then full fit.

Reports out-of-sample precision/recall at several probability cutoffs and
compares against the composite-score baseline evaluated on the SAME rows,
so the comparison is apples-to-apples.

Usage:
    python scripts/train_model.py [--start 2019-01-01] \
        [--splits 2023-01-01,2024-01-01,2025-01-01,2026-01-01]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vnmm.data import loader          # noqa: E402
from vnmm import model as M           # noqa: E402


def pr_at(panel, col, cuts):
    rows = []
    sub = panel[panel[col].notna()]
    pos = sub["label"].sum()
    for cut in cuts:
        sel = sub[sub[col] >= cut]
        if not len(sel):
            continue
        rows.append({
            "cutoff": cut, "alerts": len(sel),
            "precision": sel["label"].mean(),
            "recall": sel["label"].sum() / pos if pos else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--splits",
                    default="2023-01-01,2024-01-01,2025-01-01,2026-01-01")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    cfg = yaml.safe_load(Path(args.config).read_text())
    data = loader.fetch_universe_daily(cfg["watchlist"], start=args.start)
    index_df = loader.index_history(start=args.start)
    panel = M.build_dataset(data, index_df)
    print(f"panel: {len(panel)} rows, positives {panel['label'].mean():.1%} "
          f"({int(panel['label'].sum())})")

    panel = M.walk_forward(panel, args.splits.split(","))
    oos = panel[panel["p_oos"].notna()]
    print(f"\nOOS rows: {len(oos)}  base positive rate {oos['label'].mean():.1%}")

    print("\n=== model (walk-forward OOS) ===")
    print(pr_at(oos, "p_oos", [0.5, 0.6, 0.7, 0.8, 0.9]).to_string(index=False))

    print("\n=== composite baseline (same OOS rows) ===")
    print(pr_at(oos, "composite", [0.4, 0.5, 0.6, 0.7]).to_string(index=False))

    M.fit_full(panel)
    print(f"\nfull model saved -> {M.MODEL_PATH}")

    # feature importances via permutation on the last split's model
    from sklearn.inspection import permutation_importance
    m = M.make_model().fit(panel[M.FEATURES], panel["label"])
    imp = permutation_importance(m, panel[M.FEATURES], panel["label"],
                                 n_repeats=3, random_state=7)
    order = np.argsort(-imp.importances_mean)
    print("\nfeature importance (permutation, in-sample):")
    for i in order:
        print(f"  {M.FEATURES[i]:14s} {imp.importances_mean[i]:+.4f}")


if __name__ == "__main__":
    main()
