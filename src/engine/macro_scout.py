"""nominating tickers materially affected by today's macro headlines

one llm call per day reads the market-wide news and proposes at most two
s&p tickers for the debate queue — the debate itself, the judges, and the
risk gate still decide everything downstream
"""

import os
import json

MAX_MACRO_PINS = 2
_SCOUT_PROVIDERS = ["gemini", "mistral", "groq"]

PROMPT = (
    "You screen macro, geopolitical, policy, and industry headlines for "
    "stock-level impact, including TRANSITIVE and CROSS-DOMAIN chains.\n"
    "Today's market-wide headlines (with sentiment):\n{headlines}\n\n"
    "Reason through impact chains of any depth — for example: conflict -> "
    "commodity supply -> producers or their input-cost victims; policy or "
    "rates -> financing costs -> rate-sensitive sectors; a disaster or "
    "strike -> logistics -> retailers' inventory; regulation -> compliance "
    "costs -> incumbents vs disruptors; a technology shift -> suppliers "
    "several links up the chain.\n"
    "Name at most {k} S&P 500 tickers where such a chain makes a MATERIAL "
    "difference to the business within days-to-weeks. The chain may be "
    "indirect but every link must be stated and plausible — no "
    "broad-market proxies, no vague sector vibes. If nothing clears the "
    "materiality bar, return an empty list; that is a good answer.\n"
    'Respond with ONLY this JSON: {{"picks": [{{"ticker": "XOM", '
    '"why": "the causal chain in one compact sentence"}}]}}'
)


def _universe():
    # the canonical universe is whatever the screener scanned most
    # recently — the database needs no file paths; csv is only a fallback
    try:
        from engine.memory import get_client
        q = get_client().table("screen_results").select("ticker")
        try:
            rows = q.order("id", desc=True).limit(600).execute().data or []
        except Exception:
            rows = get_client().table("screen_results").select("ticker") \
                .limit(600).execute().data or []
        tickers = {str(r["ticker"]).upper() for r in rows
                   if r.get("ticker")}
        if len(tickers) >= 50:
            return tickers
    except Exception as e:
        print(f"  [macro-scout] screen_results unavailable ({e}) — "
              f"trying universe.csv")

    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    candidates = [root / "universe.csv",
                  *sorted(root.glob("*/universe.csv")),
                  *sorted(root.glob("*/*/universe.csv")),
                  Path("universe.csv")]
    base = os.environ.get("STOCK_LENS_BASE")
    if base:
        candidates += [Path(base) / "universe.csv",
                       Path(base) / "data" / "universe.csv"]
    for path in candidates:
        try:
            with open(path) as f:
                return {line.strip().split(",")[0].upper()
                        for line in f if line.strip()}
        except OSError:
            continue
    return None


def macro_pins():
    # proposing validated macro-driven pins, or nothing at all
    from engine.news_fetcher import fetch_macro_news
    from engine.llm_clients import ask, parse_json_reply

    headlines = fetch_macro_news()
    if not headlines:
        return []
    text = "\n".join(f"[{i}] ({h.get('sentiment')}) {h['headline']}"
                     for i, h in enumerate(headlines))
    prompt = PROMPT.format(headlines=text, k=MAX_MACRO_PINS)

    reply = None
    for provider in _SCOUT_PROVIDERS:
        reply = ask(provider, prompt)
        if reply:
            break
    parsed = parse_json_reply(reply, ["picks"])
    if not parsed or not isinstance(parsed.get("picks"), list):
        return []

    universe = _universe()
    if universe is None:
        # failing closed: unvalidated pins can waste debate slots on
        # hallucinated or dead tickers, so no universe means no pins
        print("  [macro-scout] no universe available — refusing to pin")
        return []
    out = []
    for p in parsed["picks"]:
        if len(out) >= MAX_MACRO_PINS:
            break
        ticker = str((p or {}).get("ticker", "")).upper().strip()
        why = str((p or {}).get("why", ""))[:140]
        if not ticker.replace(".", "").replace("-", "").isalpha() \
                or len(ticker) > 6:
            continue
        if ticker not in universe:
            print(f"  [macro-scout] {ticker} not in universe — dropped")
            continue
        out.append((ticker, why))
    return out
