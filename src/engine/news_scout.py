"""nominating tickers whose news is loud even when their charts are quiet

reads the bulk market feed once, resolves tickers only through finnhub's
structured `related` field (no fuzzy name matching), aggregates finbert
sentiment per ticker, and nominates the most newsworthy names into the
debate queue — the debates and gates still decide everything
"""

import os
from collections import defaultdict

MAX_NEWS_PINS = 5
MIN_ABS_SENTIMENT = 0.35     # ignoring lukewarm coverage
MIN_HEADLINES = 1


def news_pins(exclude=None):
    # proposing up to five validated news-driven nominations
    import requests
    from engine.news_fetcher import score_sentiment
    from engine.macro_scout import _universe

    exclude = exclude or set()
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        return []
    universe = _universe()
    if universe is None:
        print("  [news-scout] no universe available — refusing to nominate")
        return []

    try:
        r = requests.get("https://finnhub.io/api/v1/news",
                         params={"category": "general", "token": key},
                         timeout=15)
        r.raise_for_status()
        articles = r.json() or []
    except Exception as e:
        print(f"  [news-scout] feed unavailable: {e}")
        return []

    # aggregating sentiment per ticker via the structured related field
    per = defaultdict(list)
    for a in articles:
        related = str(a.get("related") or "")
        headline = (a.get("headline") or "").strip()
        if not headline or not related:
            continue
        s = score_sentiment(headline)
        if s is None:
            continue
        for t in related.split(","):
            t = t.strip().upper()
            if t and t in universe and t not in exclude:
                per[t].append(s)

    ranked = []
    for t, scores in per.items():
        if len(scores) < MIN_HEADLINES:
            continue
        agg = sum(scores) / len(scores)
        if abs(agg) < MIN_ABS_SENTIMENT:
            continue
        # more headlines and stronger tone rank higher
        ranked.append((abs(agg) * (1 + 0.2 * min(len(scores), 5)), t, agg,
                       len(scores)))
    ranked.sort(reverse=True)

    out = []
    for _, t, agg, n in ranked[:MAX_NEWS_PINS]:
        out.append((t, f"{n} headline(s), sentiment {agg:+.2f}"))
        print(f"  [news-scout] nominating {t}: {n} headline(s) "
              f"at {agg:+.2f}")
    return out
