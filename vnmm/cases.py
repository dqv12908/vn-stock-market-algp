"""Documented manipulation episodes (ground truth for validation).

Compiled from prosecutions (UBCKNN decisions, criminal verdicts) and major
financial press. Tiers: criminal = adjudicated manipulation; admin =
UBCKNN fine; runup = widely documented pattern, no charge. markup_start
is the approximate first session of the visible markup phase — the thing
the detector should fire BEFORE.

Sources: VnExpress, Tuổi Trẻ, Vietstock, tinnhanhchungkhoan, CafeF, Báo
Chính phủ coverage of the FLC/Trịnh Văn Quyết, Louis Holdings/Đỗ Thành
Nhân, APEC/Nguyễn Đỗ Lăng, CMS, GKM, DST, FTM, L14 cases (URLs in README).
"""

CASES = [
    # symbol, markup_start, peak_date, tier, note
    ("HAI", "2017-07-01", "2017-08-31", "criminal", "FLC group, +459% in ~1 month"),
    ("AMD", "2017-04-01", "2017-06-30", "criminal", "FLC group, +70%"),
    ("ART", "2017-09-01", "2017-11-30", "criminal", "FLC group, +330%"),
    ("FLC", "2020-09-01", "2022-01-10", "criminal", "+593%, dump 2022-01-10"),
    ("BII", "2021-02-01", "2021-09-30", "criminal", "Louis, ~10x"),
    ("TGG", "2021-06-01", "2021-09-30", "criminal", "Louis, ~37x"),
    ("AGM", "2021-08-01", "2022-03-31", "criminal", "Louis satellite"),
    ("API", "2021-05-04", "2021-12-31", "criminal", "APEC, +372%"),
    ("APS", "2021-05-04", "2021-12-31", "criminal", "APEC, +581%"),
    ("IDJ", "2021-05-04", "2021-12-31", "criminal", "APEC, +503%"),
    ("CMS", "2023-05-04", "2023-10-31", "criminal", "Zalo/Telegram pump"),
    ("L14", "2021-10-01", "2022-01-12", "admin", "20 accounts, ~4x in 3 months"),
    ("GKM", "2021-08-02", "2022-01-28", "admin", "23 accounts"),
    ("DST", "2020-02-24", "2020-10-01", "admin", "23 accounts"),
    ("FTM", "2019-01-01", "2019-08-14", "admin", "50 accounts, collapse 30 floors"),
    ("PSH", "2023-06-01", "2024-03-31", "admin", "heaviest 2024 fines"),
    ("THD", "2020-10-01", "2021-12-31", "runup", "~14x, low float"),
    ("DIG", "2021-09-01", "2022-01-11", "runup", "property wave"),
    ("CEO", "2021-09-01", "2022-01-07", "runup", "property wave"),
]
