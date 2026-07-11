"""storing and retrieving engine memory in supabase with strict validation"""

import os
import re
import json
from datetime import date, datetime, timezone
from dotenv import load_dotenv

load_dotenv()

TICKER_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")
VALID_ACTIONS = {"BUY", "SELL", "NO_TRADE"}

_client = None


def get_client():
    # creating the supabase client once, preferring the server-only secret key
    global _client
    if _client is None:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_KEY") \
            or os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and a supabase key must be set")
        _client = create_client(url, key)
    return _client


def validate_ticker(ticker):
    # rejecting anything that is not a plausible ticker symbol
    ticker = str(ticker).upper().strip()
    if not TICKER_RE.match(ticker):
        raise ValueError(f"invalid ticker: {ticker!r}")
    return ticker


def insert_decision(ticker, action, cnn_direction, cnn_confidence,
                    bull_case, bear_case, judge_votes, risk_gate_note,
                    selection_source=None):
    # recording one decision with its full debate transcript
    ticker = validate_ticker(ticker)
    if action not in VALID_ACTIONS:
        raise ValueError(f"invalid action: {action!r}")
    row = {"ticker": ticker, "action": action,
           "cnn_direction": str(cnn_direction)[:16],
           "cnn_confidence": float(cnn_confidence),
           "bull_case": bull_case, "bear_case": bear_case,
           "judge_votes": judge_votes,
           "risk_gate_note": str(risk_gate_note)[:500]}
    if selection_source:
        row["selection_source"] = str(selection_source)[:32]
    return get_client().table("decisions").insert(row).execute()


def insert_news(ticker, published_at, source, headline, summary, url,
                sentiment=None):
    # archiving one news item, ignoring duplicates by ticker+headline
    ticker = validate_ticker(ticker)
    row = {"ticker": ticker,
           "published_at": str(published_at) if published_at else None,
           "source": str(source)[:100], "headline": str(headline)[:500],
           "summary": str(summary)[:2000] if summary else None,
           "url": str(url)[:1000] if url else None,
           "sentiment": float(sentiment) if sentiment is not None else None}
    try:
        return get_client().table("news_archive").insert(row).execute()
    except Exception:
        return None


def get_recent_news(ticker, limit=5, days=7):
    # fetching recent archived headlines bounded to a freshness window
    from datetime import datetime, timedelta, timezone
    ticker = validate_ticker(ticker)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=int(days))).isoformat()
    res = get_client().table("news_archive").select(
        "published_at,source,headline,summary").eq("ticker", ticker) \
        .gte("published_at", cutoff) \
        .order("published_at", desc=True).limit(int(limit)).execute()
    return res.data or []


def get_recent_decisions(ticker, limit=5, days=30):
    # fetching recent scored decisions bounded to a freshness window
    from datetime import datetime, timedelta, timezone
    ticker = validate_ticker(ticker)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=int(days))).isoformat()
    res = get_client().table("decisions").select(
        "decided_at,action,cnn_direction,outcome_label,was_correct") \
        .eq("ticker", ticker).gte("decided_at", cutoff) \
        .order("decided_at", desc=True).limit(int(limit)).execute()
    return res.data or []


def get_active_lessons(limit=10):
    # fetching the currently active distilled lessons
    res = get_client().table("lessons").select("lesson_text") \
        .eq("active", True).order("created_at", desc=True) \
        .limit(int(limit)).execute()
    return [r["lesson_text"] for r in (res.data or [])]


def get_active_thesis(ticker):
    # fetching the active thesis for one ticker when present
    ticker = validate_ticker(ticker)
    res = get_client().table("theses").select("*").eq("ticker", ticker) \
        .eq("status", "ACTIVE").limit(1).execute()
    return res.data[0] if res.data else None


def upsert_market_context(summary_text):
    # writing today's rolling market narrative
    row = {"date": str(date.today()), "summary_text": str(summary_text)[:4000]}
    return get_client().table("market_context").upsert(row).execute()


def get_market_context():
    # reading the most recent market narrative
    res = get_client().table("market_context").select("summary_text") \
        .order("date", desc=True).limit(1).execute()
    return res.data[0]["summary_text"] if res.data else ""


def get_unscored_decisions(before_days=1):
    # fetching decisions old enough to have a next-day outcome
    res = get_client().table("decisions").select("*") \
        .is_("scored_at", "null").execute()
    return res.data or []


def save_screen_results(results, top_n=20):
    # storing today's top screener rankings for the record and the site
    from datetime import date
    rows = [{"scan_date": str(date.today()),
             "ticker": validate_ticker(r["ticker"]),
             "direction": str(r["direction"])[:16],
             "confidence": float(r["confidence"]),
             "score": float(r["score"])} for r in results[:int(top_n)]]
    if rows:
        return get_client().table("screen_results").upsert(rows).execute()


def get_recent_tickers(days=30):
    # listing tickers that received decisions inside the freshness window
    from datetime import datetime, timedelta, timezone
    cutoff = (datetime.now(timezone.utc) - timedelta(days=int(days))).isoformat()
    res = get_client().table("decisions").select("ticker") \
        .gte("decided_at", cutoff).execute()
    return sorted({r["ticker"] for r in (res.data or [])})


def score_decision(decision_id, ret_1d, ret_5d, outcome_label, was_correct):
    # writing actual outcomes back onto one decision row
    row = {"outcome_return_1d": float(ret_1d) if ret_1d is not None else None,
           "outcome_return_5d": float(ret_5d) if ret_5d is not None else None,
           "outcome_label": str(outcome_label)[:16],
           "was_correct": bool(was_correct),
           "scored_at": datetime.now(timezone.utc).isoformat()}
    return get_client().table("decisions").update(row) \
        .eq("id", int(decision_id)).execute()


def prune_news(years=5):
    # deleting archived news older than the retention window
    from datetime import datetime, timedelta, timezone
    cutoff = (datetime.now(timezone.utc)
              - timedelta(days=int(years * 365))).isoformat()
    return get_client().table("news_archive").delete() \
        .lt("published_at", cutoff).execute()


def get_open_position(ticker):
    # fetching any open position for one ticker for packet awareness
    ticker = validate_ticker(ticker)
    res = get_client().table("positions").select("qty,entry_price,entry_date") \
        .eq("ticker", ticker).eq("status", "OPEN").limit(1).execute()
    return res.data[0] if res.data else None


def get_ticker_stats(ticker):
    # summarising the all-time scored record for one ticker
    ticker = validate_ticker(ticker)
    res = get_client().table("decisions").select("was_correct") \
        .eq("ticker", ticker).not_.is_("scored_at", "null").execute()
    rows = res.data or []
    if not rows:
        return None
    correct = sum(1 for r in rows if r["was_correct"])
    return {"scored": len(rows), "correct": correct}


def save_weekly_report(stats):
    # storing one weekly report row for the site reporting tab
    from datetime import date
    return get_client().table("reports").upsert(
        {"week_of": str(date.today()), "stats": stats}).execute()


def get_watchlist():
    # reading user-pinned tickers and the fill mode from config
    res = get_client().table("config").select("value") \
        .eq("key", "user_watchlist").limit(1).execute().data
    tickers = []
    if res and res[0]["value"]:
        tickers = [t.strip().upper() for t in res[0]["value"].split(",")
                   if t.strip()]
    mode_res = get_client().table("config").select("value") \
        .eq("key", "watchlist_fill").limit(1).execute().data
    fill = mode_res[0]["value"] if mode_res else "screener"
    return tickers, fill


def set_watchlist(tickers, fill="screener"):
    # storing user-pinned tickers and fill mode (screener or empty)
    clean = ",".join(validate_ticker(t) for t in tickers)
    get_client().table("config").upsert(
        {"key": "user_watchlist", "value": clean}).execute()
    get_client().table("config").upsert(
        {"key": "watchlist_fill", "value": fill}).execute()
