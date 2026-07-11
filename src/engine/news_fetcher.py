"""fetching per-ticker news from yahoo rss and finnhub, then archiving it"""

import os
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from engine.memory import insert_news, validate_ticker

load_dotenv()

YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}"
FINNHUB_NEWS = "https://finnhub.io/api/v1/company-news"
FINNHUB_MACRO = "https://finnhub.io/api/v1/news"


def fetch_yahoo_rss(ticker):
    # pulling the latest headlines from the free yahoo rss feed
    import feedparser
    ticker = validate_ticker(ticker)
    feed = feedparser.parse(YAHOO_RSS.format(ticker=ticker))
    items = []
    for e in feed.entries[:15]:
        items.append({"ticker": ticker,
                      "published_at": getattr(e, "published", None),
                      "source": "yahoo_rss",
                      "headline": getattr(e, "title", "")[:500],
                      "summary": getattr(e, "summary", "")[:2000],
                      "url": getattr(e, "link", None)})
    return items


_macro_cache = None


def fetch_macro_news(limit=5):
    # pulling market-wide headlines once per run: geopolitics, fed, macro
    global _macro_cache
    if _macro_cache is not None:
        return _macro_cache
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        _macro_cache = []
        return _macro_cache
    try:
        r = requests.get(FINNHUB_MACRO,
                         params={"category": "general", "token": key},
                         timeout=15)
        r.raise_for_status()
        items = []
        for a in (r.json() or [])[:limit * 3]:
            headline = (a.get("headline") or "").strip()
            if not headline:
                continue
            items.append({
                "headline": headline,
                "source": a.get("source") or "finnhub",
                "published_at": datetime.fromtimestamp(
                    a.get("datetime", 0),
                    tz=timezone.utc).isoformat(),
                "sentiment": score_sentiment(headline),
                "url": a.get("url") or "",
            })
            if len(items) >= limit:
                break
        _macro_cache = items
        print(f"  [news] {len(items)} macro headlines fetched")
    except Exception as e:
        print(f"  [news] macro fetch failed: {e}")
        _macro_cache = []
    return _macro_cache


def archive_macro_news():
    # storing macro headlines under the MARKET pseudo-ticker for the site
    try:
        from engine.memory import insert_news
        for it in fetch_macro_news():
            insert_news(ticker="MARKET",
                        published_at=it["published_at"],
                        source=it["source"], headline=it["headline"],
                        summary="", url=it["url"],
                        sentiment=it["sentiment"])
    except Exception as e:
        print(f"  [news] macro archive failed: {e}")


def fetch_finnhub(ticker, days_back=3):
    # pulling structured company news from the finnhub free tier
    ticker = validate_ticker(ticker)
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        return []
    to_d = datetime.now(timezone.utc).date()
    from_d = to_d - timedelta(days=days_back)
    try:
        r = requests.get(FINNHUB_NEWS,
                         params={"symbol": ticker, "from": str(from_d),
                                 "to": str(to_d), "token": key},
                         timeout=15)
        r.raise_for_status()
        items = []
        for a in r.json()[:15]:
            items.append({"ticker": ticker,
                          "published_at": datetime.fromtimestamp(
                              a.get("datetime", 0),
                              tz=timezone.utc).isoformat(),
                          "source": str(a.get("source", "finnhub"))[:100],
                          "headline": str(a.get("headline", ""))[:500],
                          "summary": str(a.get("summary", ""))[:2000],
                          "url": a.get("url")})
        return items
    except Exception:
        return []


def fetch_next_earnings(ticker, days_ahead=30):
    # finding how many days until the next scheduled earnings report
    ticker = validate_ticker(ticker)
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        return None
    today = datetime.now(timezone.utc).date()
    try:
        r = requests.get("https://finnhub.io/api/v1/calendar/earnings",
                         params={"from": str(today),
                                 "to": str(today + timedelta(days=days_ahead)),
                                 "symbol": ticker, "token": key},
                         timeout=15)
        r.raise_for_status()
        events = r.json().get("earningsCalendar", [])
        dates = sorted(e["date"] for e in events if e.get("date"))
        if not dates:
            return None
        next_date = datetime.strptime(dates[0], "%Y-%m-%d").date()
        return (next_date - today).days
    except Exception:
        return None


def dedupe(items):
    # dropping items whose headlines are near-duplicates of earlier ones
    seen, out = set(), []
    for it in items:
        key = it["headline"].lower().strip()[:80]
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


_analyzer = None
_finbert = None
_finbert_dead = False

# finance sentiment ladder: env override first, then candidates in order
# — first one that loads wins, and every run on ci pulls its latest head
FINBERT_CANDIDATES = [m for m in [
    os.environ.get("FINBERT_MODEL"),
    "ProsusAI/finbert",
    "yiyanghkust/finbert-tone",
] if m]


def _finbert_score(text):
    # scoring with the best available finance-tuned model
    global _finbert, _finbert_dead
    if _finbert_dead:
        return None
    try:
        if _finbert is None:
            from transformers import pipeline
            last_err = None
            for name in FINBERT_CANDIDATES:
                try:
                    _finbert = pipeline("text-classification",
                                        model=name, top_k=None)
                    print(f"  [news] sentiment model loaded: {name}")
                    break
                except Exception as e:
                    last_err = e
                    print(f"  [news] {name} failed to load — trying next")
            if _finbert is None:
                raise RuntimeError(last_err)
        scores = {d["label"].lower(): d["score"]
                  for d in _finbert(str(text)[:400])[0]}
        pos = next((v for k, v in scores.items() if k.startswith("pos")), 0.0)
        neg = next((v for k, v in scores.items() if k.startswith("neg")), 0.0)
        return round(pos - neg, 3)
    except Exception as e:
        # falling back to vader for this run rather than failing news
        print(f"  [news] finance models unavailable ({e}) — using vader")
        _finbert_dead = True
        return None


def score_sentiment(text):
    # scoring headline sentiment between -1 and 1 with a finance-aware lexicon
    fb = _finbert_score(text)
    if fb is not None:
        return fb
    global _analyzer
    try:
        if _analyzer is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _analyzer = SentimentIntensityAnalyzer()
            _analyzer.lexicon.update({
                "beats": 2.5, "beat": 2.0, "soars": 3.0, "surges": 3.0,
                "rallies": 2.5, "rally": 2.0, "upgrade": 2.0,
                "upgraded": 2.0, "outperform": 2.0, "buyback": 1.5,
                "raises": 1.5, "record": 1.5, "bullish": 2.0,
                "misses": -2.5, "plunges": -3.0, "tumbles": -2.5,
                "slumps": -2.5, "downgrade": -2.0, "downgraded": -2.0,
                "underperform": -2.0, "cuts": -1.5, "bearish": -2.0,
                "lawsuit": -2.0, "probe": -2.0, "recall": -2.0,
                "bankruptcy": -3.5, "selloff": -2.5, "warns": -1.5})
        return round(_analyzer.polarity_scores(str(text)[:500])["compound"], 3)
    except Exception:
        return None


def fetch_alpaca_news(ticker, limit=8):
    # third news source: alpaca's news api, using the trading keys we
    # already hold — engaged when the primary feeds come back empty
    import os
    import requests
    key = os.environ.get("ALPACA_API_KEY")
    sec = os.environ.get("ALPACA_SECRET_KEY")
    if not (key and sec):
        return []
    try:
        r = requests.get("https://data.alpaca.markets/v1beta1/news",
                         params={"symbols": ticker, "limit": limit},
                         headers={"APCA-API-KEY-ID": key,
                                  "APCA-API-SECRET-KEY": sec}, timeout=10)
        r.raise_for_status()
        out = []
        for a in (r.json() or {}).get("news", []):
            out.append({"ticker": ticker,
                        "headline": (a.get("headline") or "")[:300],
                        "summary": (a.get("summary") or "")[:500],
                        "source": a.get("source") or "alpaca",
                        "url": a.get("url") or "",
                        "published_at": a.get("created_at")})
        return out
    except Exception as e:
        print(f"  [news] alpaca fallback failed: {e}")
        return []


def fetch_and_archive(ticker, top_n=5):
    # combining sources with a fallback ladder so one provider outage or a
    # retired endpoint can never blind the packet: yahoo rss + finnhub
    # first, alpaca news when both come back empty
    items = dedupe(fetch_yahoo_rss(ticker) + fetch_finnhub(ticker))
    if not items:
        items = dedupe(fetch_alpaca_news(ticker))
    for it in items:
        it["sentiment"] = score_sentiment(
            f"{it['headline']} {it.get('summary') or ''}")
        insert_news(**it)
    return items[:top_n]
