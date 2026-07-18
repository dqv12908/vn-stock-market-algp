# vn-stock-market-algp

**Detecting market-maker ("đội lái") campaigns in the Vietnamese stock market
before the markup — from public data.**

## The thesis

Big moves in VN small/mid caps (±20–50% in weeks) are rarely organic: free
float is thin, retail dominates, and a crew with a few hundred billion VND can
control a tape. But a campaign **cannot be invisible**, because the operator
must physically accumulate a large share of the float through the same public
order book everyone sees. Every phase leaves measurable distortions:

| Phase | What the operator does | What the tape shows |
|---|---|---|
| 1. Accumulation | Buys quietly for weeks, caps rallies so price stays cheap | OBV rises while price is flat; big volume with no price result; closes near bar highs; rising lows under a flat lid |
| 2. Shakeout | Engineered flush (often to the floor band) to strip weak hands | Outsized down-spike on high volume that recovers intra-day |
| 3. Markup | The visible run, often ceiling-limit (trần) chains | Volume *dries up* at ceiling — nobody left to sell |
| 4. Distribution | Sells into euphoria | Record volume near highs, weakening closes, OFI flips |

The tradeable information is in phases 1–2. This repo measures it at three
granularities, all from free public endpoints (via `vnstock` → VCI):

1. **Daily bars** (years of history) — accumulation footprint
2. **Tick-by-tick tape** with aggressor side Buy/Sell — execution footprint
3. **Top-10 order book depth** (live only — we collect our own history)

## The signals (the math)

### Daily accumulation features — `vnmm/signals/accumulation.py`

All causal (rolling windows, no lookahead), emitted in z-score-like units:

- **`obv_div`** — rolling OLS slope of normalized OBV minus slope of
  normalized price. Volume flowing in while price goes nowhere is the single
  strongest accumulation tell.
- **`absorption`** — Wyckoff effort-vs-result: `z( (V/μV) · exp(−2·|ret|/ATR%) )`.
  High volume producing no move = supply being absorbed.
- **`cmf`** — Chaikin money flow (volume-weighted close-location value, 20d):
  accumulation days close near their highs.
- **`vol_z`** — z-score of log volume vs 60d.
- **`turnover_z`** — 20d cumulative volume vs its own 1y norm (float churn).
- **`coil`** — slope(lows) − slope(highs)⁺: rising lows under a flat lid.
- **`bb_squeeze`** — 1 − percentile rank of 20d volatility: crews mark up
  from compressed coils (cheaper to move).
- **`shakeout`** — flag: down-spike > 1.5 ATR on vol_z > 1 closing in the
  upper half of its range.

**Composite** (`vnmm/signals/composite.py`): weighted linear sum, clipped to
±3 per feature, normalized to ~[−3, 3]. Linear on purpose — every alert is
fully attributable to named distortions.

### Tick features — `vnmm/signals/tickflow.py`

From the match-by-match tape (aggressor side included):

- **`ofi`** — order-flow imbalance: (aggr. buy vol − aggr. sell vol)/matched vol.
- **`flow_persistence`** — lag-1 autocorrelation of 5-min signed volume.
  Parent-order execution programs are persistent; organic flow mean-reverts.
- **`block_share`** — share of volume in prints ≥ its own 95th-pct size.
- **`size_entropy`** — normalized Shannon entropy of the print-size
  distribution. **Low entropy = wash-trade / painting signature** (crews
  reuse lot sizes; organic tape is size-diverse).
- **`top_size_share`** — share of volume in the single most repeated size.
- **`auction_share` / `atc_gap`** — ATC volume share and ATC-vs-last price
  gap: *marking the close* to set tomorrow's ±7% band favorably. Normal ATC
  share is ~3–8%; campaigns show 20–60%.

### Order book features — `vnmm/signals/orderbook.py`

From collected top-10 depth snapshots:

- **`depth_imbalance` / `touch_imbalance`** — (Σbid − Σask)/(Σbid + Σask),
  full book and exp-weighted near-touch.
- **`bid_wall` / `ask_wall`** — max single-level share of its side.
- **`spoof_score`** — ask walls that vanish *without* trade volume advancing
  (cancelled, not eaten) — layering to shake out sellers.
- **`absorption_score`** — heavy traded volume with a still price: offers
  being eaten ahead of markup.
- **`di_trend` / `di_flip`** — imbalance regime and its reversal (a flip
  after a run = distribution starting).

## Does it work — and is it just "everything went up"? (event study, real data)

The evaluation is deliberately **market-adjusted**: every return below is the
stock's forward return **minus VNINDEX's return over the identical calendar
window** (abnormal return), so a broad rally contributes equally to both legs
and cancels. The features themselves are also market-relative — OBV
direction, OBV/price divergence and absorption are computed on the stock's
return in excess of the index (`accumulation_features(benchmark=...)`), so an
index melt-up cannot masquerade as stock-specific accumulation.

50-symbol speculative watchlist, daily bars 2024-06 → 2026-07, event = 5-day
mean composite crossing a threshold, 20-session cooldown, vs. unconditional
abnormal returns of the same universe (4,200 samples):

| Threshold | Events | +10d abn. | +20d abn. | +20d t-stat | ≥15% abn. max-gain in 20d |
|---|---|---|---|---|---|
| baseline | — | −0.60% | −1.23% | — | 10% |
| 0.5 | 90 | +0.94% | +1.10% | +1.6 | 19% |
| 0.6 | 38 | +1.84% | +3.35% | **+2.0** | 26% |
| 0.7 | 13 | +1.94% | +5.23% | **+2.6** | **31% (3x)** |

Two things worth noticing. First, the *baseline is negative*: this universe
of speculative names loses to the index unconditionally — the signal's job is
picking the exceptions, and the spread (event minus baseline) is ~4.5–6.5pp
at +20d. Second, the lift survives market adjustment but is roughly half the
raw-return lift, which is exactly the uptrend effect this adjustment exists
to remove (run `--raw` to see the difference).

### Why the weights are not curve-fitted

`scripts/optimize_weights.py` runs a walk-forward search (train
2024-06→2025-09, test 2025-10→now; 20,800 symbol-days) under two objectives:
rank-IC over all days, and mean abnormal return of the top-2% score tail.
Both found weight vectors that beat the defaults in-sample and **both lost to
the theory-derived defaults out-of-sample** (e.g. tail-objective winner:
+2.0% vs defaults' +8.1% at +20d on the test window). With this sample size
the tail is too thin to fit on, so the shipped weights stay theory-grounded —
re-run the optimizer once you have a bigger universe or your own collected
depth history.

Remaining caveats: one two-year window, no transaction costs, no band-limit
fill modeling (you often *can't buy* a ceiling-locked stock). Re-run
`scripts/backtest_signals.py` on your own universe before trusting anything.

## Usage

```bash
pip install -r requirements.txt

# 1. Daily + tape scan over the watchlist, prints alerts
#    (Telegram: export TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)
python scripts/run_scanner.py

# 2. Order book history collector — leave running during trading hours
#    (09:00-11:30 / 13:00-14:45 ICT). Snapshots -> data/orderbook/DATE/SYM.parquet
python scripts/collect_orderbook.py --interval 15

# 3. Event-study backtest
python scripts/backtest_signals.py --start 2024-01-01 --threshold 0.5 --show-events
```

Watchlist and thresholds live in `config.yaml`.

**Rate limits:** the free VCI endpoints allow ~60 req/min and vnstock's
limiter kills the process when exceeded. The loader caches daily history in
`data/cache/`; for large universes pre-fetch slowly (`fetch_universe_daily`
with `pause>=3`) or register a free API key at vnstocks.com.

## Ground truth: documented đội lái cases

`vnmm/cases.py` encodes 19 documented manipulation episodes (2016–2024)
compiled from prosecutions and UBCKNN decisions — the FLC group (Trịnh Văn
Quyết, 500 accounts, wash trading + ATO/ATC domination + spoofing), Louis
Holdings (TGG 37x, BII 10x, 17 accounts + social pump groups), APEC
(API/APS/IDJ +372–581%, 40 accounts, "set new closing prices"), CMS
(Zalo/Telegram pump), GKM, DST, FTM, L14 and more. Each has an approximate
markup-start date, so `scripts/validate_cases.py` can ask the only question
that matters: **did the detector fire before the documented markup?**
Labels come from prosecutions, not price patterns — no circularity with the
price-based training labels.

The prosecution records also validate the feature set directly: every
indictment describes account-cluster wash trading (→ print-size entropy,
churn, side-alternation), ATO/ATC domination (→ auction share, ATC gap),
place-then-cancel spoofing (→ vanishing-wall score), and ceiling-chain
engineering (→ ceiling-touch magnet count).

## What the detection literature says (and what we adopted)

Key findings from the pump-and-dump / manipulation-detection literature that
shaped this system (full citations in the papers noted):

- **Rush-order burstiness** — dispersion of aggressive-buy volume across
  short time chunks — is the single strongest validated pre-pump feature
  (~37% of model importance in La Morgia et al., arXiv:2105.00733). Adopted
  as `rush_std`/`rush_mean` in the tick features.
- **~70% of pump events show a detectable accumulation phase** before markup
  (arXiv:2504.15790) — validating the two-timescale design: daily
  accumulation flag → intraday confirmation.
- **Wash-trade statistics from anonymous prints** (Cong et al., NBER
  w30783): Benford first-digit deviation, round-size share, size-entropy.
  Adopted as `benford_mad`/`round_share`. True circular-trade detection
  needs account graphs regulators have and we don't — these are suspicion
  proxies, stated as such.
- **Magnet effect at price limits** (PLOS ONE 2015, Chinese A-shares — the
  closest market structure to VN): ceiling proximity attracts follow-through.
  Adopted as `ceil_touches` + `breakout_prox`.
- **Manipulation concentrates in small, illiquid, volatile stocks**
  (Aggarwal & Wu) — the watchlist is that cross-sectional prior.
- **Realistic performance bar**: with regulator-grade labels and rare
  events, published systems reach high recall but base-rate-limited
  precision (e.g. recall 0.89 / precision <0.1, MDPI Math 12(9):1336).
  Precision@top-N per day plus post-alert abnormal returns is the honest
  reporting standard — which is what `scripts/evaluate_alerts.py` and the
  event study report.

## Data sources — what exists and what doesn't

| Data | Source | History |
|---|---|---|
| Daily OHLCV | vnstock → VCI, free | years |
| Tick tape w/ Buy/Sell side | vnstock → VCI `intraday`, free | recent session(s) only — **archive it daily if you want history** |
| Top-10 order book depth | vnstock → VCI `price_board`, free | **live only — no public history anywhere; this repo's collector builds your own** |
| Full order-event feed (adds/cancels) | SSI FastConnect / DNSE KRX APIs (registration) | live streaming |
| Foreign buy/sell flow | included in price board & TCBS endpoints | daily history available |

Upgrade path, in order of value (researched July 2026):

1. **Self-record the tape** — run `scripts/archive_ticks.py` after each
   close and `scripts/collect_orderbook.py` during sessions. Free, and after
   a few weeks you own the tick/depth history nobody sells cheaply.
2. **SSI FastConnect Data** (free with an SSI account): streaming `X`
   channel = full 10-level book + trades with aggressor side + foreign room
   in real time. DNSE LightSpeed KRX (MQTT websocket) as a free redundant
   feed. Note: VN exchanges only disseminate aggregated N-level depth —
   nobody gets per-order (L3) events, so cancel-inference from snapshots is
   as good as it gets outside the regulator.
3. **Foreign + tự doanh (prop) daily history** — free JSON endpoints at
   cafef (`lich-su-giao-dich-api-3/-4.chn`) and `foreign_trade` in vnstock.
4. **WiFeed API** (cheap, à la carte): major-shareholder changes and
   free-float per ticker — float concentration is the strongest structural
   predictor of pump-able tickers.
5. **FiinQuant** (paid, 14-day trial): the only KRX-connected historical
   tick archive if you need history you didn't record yourself.
6. **F319/F247 forum post-velocity** (scrape) and FireAnt posts API — pump
   crews recruit retail there; thread-velocity spikes lead markups.

## Repo layout

```
vnmm/
  data/loader.py                 # daily / tick / price-board access + cache
  data/orderbook_collector.py    # depth snapshot daemon -> parquet
  signals/accumulation.py        # daily-bar footprint features
  signals/tickflow.py            # tape microstructure features
  signals/orderbook.py           # depth + book-dynamics features
  signals/composite.py           # weighted, interpretable scores
  scanner.py                     # watchlist scans
  backtest.py                    # event-study harness
  alerts.py                      # console + Telegram
scripts/                         # runnable entry points
config.yaml                      # watchlist + thresholds
```

## Disclaimer

Research tooling, not investment advice. Detecting manipulation footprints to
inform your own risk is legal; *participating* in coordinated manipulation is
not (see the FLC/Trịnh Văn Quyết and Louis Holdings prosecutions). Driven
stocks routinely retrace 50%+ when the crew exits — the same signals that get
you in are the ones to watch for the flip.
