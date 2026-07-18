"""Supervised markup-onset predictor.

Learns P(markup onset within the next `lead` sessions) from the daily
feature panel, using labeled onsets (vnmm.labels). Compared to the fixed
composite weights this lets feature interactions matter (e.g. absorption
only predicts when volatility is compressed), while walk-forward
evaluation keeps us honest about overfitting — the earlier random-search
experiment showed exactly how easy that is here.

Model: HistGradientBoostingClassifier — handles NaNs natively, robust to
feature scaling, strong on tabular data with ~10^4 rows. Class imbalance
(~5% positives) is handled via class_weight='balanced'.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import align_index
from .labels import markup_onsets
from .signals.accumulation import accumulation_features
from .signals.composite import daily_score

log = logging.getLogger(__name__)

WARMUP = 120
LEAD = 12
MODEL_PATH = Path("data/onset_model.joblib")

# feature columns fed to the model (everything accumulation_features emits)
FEATURES = [
    "vol_z", "obv_div", "absorption", "cmf", "bb_squeeze", "turnover_z",
    "coil", "shakeout", "ceil_touches", "illiq_drop", "acc_streak",
    "breakout_prox",
]


def build_dataset(data: dict[str, pd.DataFrame],
                  index_df: pd.DataFrame | None,
                  lead: int = LEAD) -> pd.DataFrame:
    """Panel of features + binary label (onset within next `lead` sessions).

    Rows within `lead` sessions BEFORE an onset are positive; rows during
    the 15 sessions after an onset are dropped (mid-pump, neither a clean
    positive nor negative). Everything is causal: features at bar i use
    data <= i, labels use only the [i+1, i+lead] window.
    """
    frames = []
    for sym, df in data.items():
        df = df.reset_index(drop=True)
        if len(df) < WARMUP + 40:
            continue
        bench = align_index(df, index_df)
        feats = accumulation_features(df, benchmark=bench)
        feats["composite"] = daily_score(feats).rolling(5).mean()
        onsets = markup_onsets(df, bench)
        n = len(df)
        label = np.zeros(n, dtype=int)
        drop = np.zeros(n, dtype=bool)
        for o in onsets:
            label[max(0, o - lead): o + 1] = 1
            drop[o + 1: o + 16] = True
        feats["label"] = label
        feats["drop"] = drop
        feats["symbol"] = sym
        feats["date"] = pd.to_datetime(df["time"]).values
        frames.append(feats.iloc[WARMUP:])
    panel = pd.concat(frames, ignore_index=True)
    return panel[~panel["drop"]].drop(columns="drop")


def make_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_depth=3, max_iter=150, learning_rate=0.05,
        min_samples_leaf=40, l2_regularization=1.0,
        class_weight="balanced", random_state=7)


def walk_forward(panel: pd.DataFrame, splits: list[str]) -> pd.DataFrame:
    """Expanding-window walk-forward: for each split date, train on all
    rows strictly before it, predict until the next split. Returns the
    panel with an out-of-sample probability column `p_oos`."""
    panel = panel.sort_values("date").reset_index(drop=True)
    panel["p_oos"] = np.nan
    bounds = [pd.Timestamp(s) for s in splits] + [panel["date"].max()
                                                  + pd.Timedelta(days=1)]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        train = panel[panel["date"] < lo]
        test_m = (panel["date"] >= lo) & (panel["date"] < hi)
        if len(train) < 2000 or not test_m.any():
            continue
        m = make_model()
        m.fit(train[FEATURES], train["label"])
        panel.loc[test_m, "p_oos"] = m.predict_proba(
            panel.loc[test_m, FEATURES])[:, 1]
    return panel


def fit_full(panel: pd.DataFrame, path: Path = MODEL_PATH):
    import joblib
    m = make_model()
    m.fit(panel[FEATURES], panel["label"])
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": m, "features": FEATURES, "lead": LEAD}, path)
    return m


def load(path: Path = MODEL_PATH):
    import joblib
    if not path.exists():
        return None
    return joblib.load(path)
