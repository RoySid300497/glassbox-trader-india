"""grading each evidence concept by the scored outcomes of calls citing it"""

import json
from datetime import date
from engine.memory import get_client

SHRINK = 10                  # prior weight pulling small samples to baseline

WINDOW = 500                 # judging over the most recent scored decisions
MIN_CITATIONS = 5            # hiding buckets with too little evidence

BUCKETS = [
    ("market_context", ("market_context", "spy_trend", "sector_strength")),
    ("liquidity_sweep", ("sweep",)),
    ("fvg_order_block", ("fvg", "order_block", "zone")),
    ("market_structure", ("structure",)),
    ("ema_regime", ("ema", "regime", "ma50", "200")),
    ("divergence", ("divergence",)),
    ("vix_fix_capitulation", ("capitulation", "wvf", "vix")),

    ("prior_day_break", ("prior_high", "prior_low", "prior_range",
                         "gap_above", "gap_below", "failed_gap",
                         "close_in_prior")),
    ("range_or_squeeze", ("rang", "squeeze", "position_in_range")),
    ("market_stage", ("stage",)),
    ("connors_rsi2", ("rsi2", "connors")),
    ("first_pullback", ("pullback",)),
    ("volume", ("volume", "poc")),
    ("candles_momentum", ("candle", "engulf", "momentum", "wick",
                          "exhaustion", "asymmetry", "inside",
                          "first_red", "first_green")),
    ("adx_trend", ("adx",)),
    ("overnight_gap", ("gap",)),
    ("insider_activity", ("insider",)),
    ("congress_trading", ("congress", "senator", "stock_act")),
    ("macro_news", ("macro",)),
    ("news_sentiment", ("news", "headline", "sentiment")),
    ("model_signal", ("cnn", "signal", "model")),
    ("track_record", ("track", "past_decision", "record")),
    ("earnings", ("earnings",)),
    ("thesis_lessons", ("thesis", "lesson")),
]


# packet blocks that map to one concept regardless of leaf field
BLOCK_MAP = {
    "macro_news": "macro_news",
    "news": "news_sentiment",
    "market_context": "market_context",
    "congress_trading": "congress_trading",
    "insider_activity": "insider_activity",
    "cnn_signal": "model_signal",
    "earnings": "earnings",
    "overnight_gap": "overnight_gap",
    "track_record": "track_record",
    "past_decisions": "track_record",
    "recent_decisions": "track_record",
    "ticker_track_record": "track_record",
    "days_to_earnings": "earnings",
    "news_sentiment_avg": "news_sentiment",
    "theses": "thesis_lessons",
    "active_thesis": "thesis_lessons",
    "lessons": "thesis_lessons",
}


def _bucket(field):
    # mapping one cited packet field onto its strategy concept: the block
    # prefix decides when it can, otherwise needles match the leaf only
    # (never the block name, which would shadow every structure feature)
    f = str(field).lower()
    head = f.split(".")[0].split("[")[0].strip()
    if head in BLOCK_MAP:
        return BLOCK_MAP[head]
    leaf = f.split(".")[-1]
    for name, needles in BUCKETS:
        if any(n in leaf for n in needles):
            return name
    return "other"


def _cited_buckets(case_blob, side):
    # collecting the distinct concepts one side's cases cited
    if isinstance(case_blob, str):
        try:
            case_blob = json.loads(case_blob)
        except Exception:
            return set()
    if not isinstance(case_blob, dict):
        return set()
    out = set()
    for stage in ("opening", "rebuttal"):
        for case in case_blob.get(stage) or []:
            for kp in (case or {}).get("key_points") or []:
                field = (kp or {}).get("evidence_field")
                if field:
                    out.add(_bucket(field))
    return out


def evidence_report():
    # cross-tabulating cited concepts against real outcomes of directional calls
    rows = get_client().table("decisions") \
        .select("action,bull_case,bear_case,was_correct") \
        .not_.is_("was_correct", "null") \
        .in_("action", ["BUY", "SELL"]) \
        .order("decided_at", desc=True).limit(WINDOW).execute().data or []
    if len(rows) < MIN_CITATIONS:
        print(f"evidence report: only {len(rows)} scored directional "
              f"calls — need more history before grading concepts")
        return {}

    stats = {}
    for r in rows:
        # crediting the side that argued for the action actually taken
        side = "bull_case" if r["action"] == "BUY" else "bear_case"
        for bucket in _cited_buckets(r.get(side), side):
            s = stats.setdefault(bucket, [0, 0])
            s[1] += 1
            s[0] += 1 if r["was_correct"] else 0

    base_n = len(rows)
    base_hit = sum(1 for r in rows if r["was_correct"]) / base_n

    # loading the snapshot from roughly four weeks ago for the trend column
    prior = {}
    try:
        old = get_client().table("evidence_stats") \
            .select("concept,hit_rate,week_ending") \
            .lte("week_ending", str(date.today())) \
            .order("week_ending", desc=True).limit(200).execute().data or []
        weeks = sorted({r["week_ending"] for r in old}, reverse=True)
        if len(weeks) >= 4:
            target = weeks[3]
            prior = {r["concept"]: float(r["hit_rate"])
                     for r in old if r["week_ending"] == target}
    except Exception:
        pass

    # ranking by baseline-shrunk edge so thin samples cannot top the table
    def score(hits, n):
        return (hits + base_hit * SHRINK) / (n + SHRINK) - base_hit

    print(f"\nevidence effectiveness — {base_n} scored directional calls, "
          f"baseline hit rate {base_hit:.0%}")
    print(f"{'concept':<24} {'cited':>6} {'hit rate':>9} "
          f"{'edge*':>7} {'4wk trend':>10}")
    shown = 0
    snapshot = []
    for name, (hits, n) in sorted(stats.items(),
                                  key=lambda kv: -score(*kv[1])):
        if n < MIN_CITATIONS:
            continue
        rate = hits / n
        trend = ""
        if name in prior:
            d = rate - prior[name]
            trend = f"{d:+.0%}"
        print(f"{name:<24} {n:>6} {rate:>8.0%} {score(hits, n):>+6.0%} "
              f"{trend:>10}")
        shown += 1
    if not shown:
        print("no concept has enough citations yet — check back next week")
    print("(*edge shrunk toward baseline; rankings stabilise as n grows)")

    # persisting this week's snapshot so the time series accumulates
    try:
        week = str(date.today())
        for name, (hits, n) in stats.items():
            get_client().table("evidence_stats").upsert(
                {"week_ending": week, "concept": name, "cited": n,
                 "hits": hits, "hit_rate": round(hits / n, 4),
                 "baseline": round(base_hit, 4)},
                on_conflict="week_ending,concept").execute()
        print(f"snapshot stored for week ending {week}")
    except Exception as e:
        print(f"snapshot store failed: {e}")

    return {k: (h / n, n) for k, (h, n) in stats.items()
            if n >= MIN_CITATIONS}
