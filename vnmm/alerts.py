"""Alert sinks: console always; Telegram if configured.

Telegram setup: create a bot with @BotFather, put the token and your chat id
in env vars TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

log = logging.getLogger(__name__)


def format_alerts(daily: pd.DataFrame, threshold: float = 0.8,
                  tick: pd.DataFrame | None = None,
                  tick_threshold: float = 0.5) -> str:
    """Tier policy validated walk-forward OOS (2023-2026, 39k symbol-days):
    STRONG = composite>=0.6 AND p_onset>=0.6  -> 24.7% precision, 3.7x base
    WATCH  = composite>=0.5 AND p_onset>=0.5  -> 14.7% precision, 2.2x base
    (base rate 6.6%; see scripts/train_model.py + README)."""
    lines: list[str] = []
    if len(daily) and "p_onset" in daily.columns:
        strong_m = (daily["score_ma5"] >= 0.6) & (daily["p_onset"] >= 0.6)
        watch_m = ((daily["score_ma5"] >= 0.5) & (daily["p_onset"] >= 0.5)
                   & ~strong_m)
        for tier, rows_ in (("🚨 STRONG", daily[strong_m]),
                            ("👀 WATCH", daily[watch_m])):
            for _, r in rows_.iterrows():
                lines.append(
                    f"{tier} {r['symbol']}  P(onset<=12d)={r['p_onset']:.0%}  "
                    f"score={r['score_ma5']:.2f}  close={r['close']}")
    hot = daily[daily["score_ma5"] >= threshold] if len(daily) else daily
    for _, r in hot.iterrows():
        drivers = []
        for k, label in (("obv_div", "OBV-divergence"), ("absorption", "absorption"),
                         ("cmf", "money-flow"), ("vol_z", "volume"),
                         ("shakeout", "shakeout"), ("bb_squeeze", "squeeze")):
            if k in r and r[k] > 0.8:
                drivers.append(label)
        lines.append(f"🔎 {r['symbol']}  score={r['score']:.2f} (5d avg {r['score_ma5']:.2f}) "
                     f"close={r['close']}  drivers: {', '.join(drivers) or 'broad'}")
    if tick is not None and len(tick):
        for _, r in tick[tick["tick_score"] >= tick_threshold].iterrows():
            lines.append(f"⚡ {r['symbol']}  tape score={r['tick_score']:.2f} "
                         f"OFI={r.get('ofi', 0):+.2f} blocks={r.get('block_share', 0):.0%} "
                         f"ATC={r.get('auction_share', 0):.0%} entropy={r.get('size_entropy', 0):.2f}")
    return "\n".join(lines)


def send(message: str) -> None:
    if not message:
        log.info("no alerts today")
        return
    print(message)
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat:
        import urllib.parse
        import urllib.request
        url = (f"https://api.telegram.org/bot{token}/sendMessage?"
               + urllib.parse.urlencode({"chat_id": chat, "text": message}))
        try:
            urllib.request.urlopen(url, timeout=10)
        except Exception as e:  # noqa: BLE001
            log.warning("telegram send failed: %s", e)
