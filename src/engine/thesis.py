"""managing long-horizon theses that daily decisions alone cannot hold"""

import json
from datetime import date, datetime, timedelta, timezone
from engine.llm_clients import ask, parse_json_reply
from engine.memory import (get_client, get_recent_decisions, get_recent_news,
                           get_active_thesis, validate_ticker)

THESIS_SCHEMA = ["thesis_text", "direction", "confidence"]
REVIEW_SCHEMA = ["status", "confidence", "reason"]
AUTO_WEAKEN_MOVE = 0.10          # flagging any thesis fighting a 10% adverse move


def fetch_institutional(ticker):
    # summarising top institutional holders as thesis evidence
    try:
        import yfinance as yf
        df = yf.Ticker(ticker.replace(".", "-")).institutional_holders
        if df is None or df.empty:
            return []
        out = []
        for _, row in df.head(5).iterrows():
            out.append({"holder": str(row.get("Holder", ""))[:80],
                        "pct_out": float(row.get("pctHeld",
                                         row.get("% Out", 0)) or 0)})
        return out
    except Exception:
        return []


def propose_thesis(ticker):
    # asking gemini whether the accumulated evidence supports a thesis
    ticker = validate_ticker(ticker)
    if get_active_thesis(ticker):
        return None
    decisions = get_recent_decisions(ticker, limit=10)
    news = get_recent_news(ticker, limit=10)
    same_dir = [d for d in decisions if d.get("action") == "BUY"]
    if len(same_dir) < 3:
        return None
    institutional = fetch_institutional(ticker)

    prompt = ("You are a long-horizon analyst. Based ONLY on the evidence "
              "below, state whether a structural multi-month thesis exists "
              "for this stock. Institutional positions are facts; treat "
              "opinions in headlines as weak evidence. If evidence is weak, "
              "set confidence below 0.5. Respond with ONLY json: "
              '{"thesis_text": str, "direction": "LONG"|"SHORT", '
              '"confidence": float 0-1}\n\nEVIDENCE:\n'
              + json.dumps({"ticker": ticker, "decisions": decisions,
                            "news": news,
                            "institutional_holders": institutional},
                           default=str)[:5000])
    reply = parse_json_reply(ask("gemini", prompt), THESIS_SCHEMA)
    if not reply or reply.get("confidence", 0) < 0.6 \
            or reply.get("direction") not in ("LONG", "SHORT"):
        return None

    row = {"ticker": ticker, "thesis_text": str(reply["thesis_text"])[:2000],
           "direction": reply["direction"],
           "confidence": float(reply["confidence"]),
           "evidence": {"decisions": decisions[:5], "news": news[:5],
                        "institutional": institutional},
           "review_after": str(date.today() + timedelta(days=7))}
    get_client().table("theses").insert(row).execute()
    return row


def review_theses(price_lookup):
    # re-examining every active thesis weekly with code-enforced honesty
    res = get_client().table("theses").select("*") \
        .eq("status", "ACTIVE").execute()
    for th in (res.data or []):
        ticker = th["ticker"]

        # auto-flagging any thesis the market has moved hard against
        move = price_lookup(ticker)
        adverse = (th["direction"] == "LONG" and move is not None
                   and move < -AUTO_WEAKEN_MOVE) or \
                  (th["direction"] == "SHORT" and move is not None
                   and move > AUTO_WEAKEN_MOVE)
        if adverse:
            get_client().table("theses").update(
                {"status": "WEAKENING",
                 "updated_at": datetime.now(timezone.utc).isoformat()}).eq("id", th["id"]).execute()
            continue

        # asking gemini to reassess against the freshest evidence
        prompt = ("You are reviewing an existing investment thesis against "
                  "new evidence. Respond with ONLY json: "
                  '{"status": "ACTIVE"|"WEAKENING"|"CLOSED", '
                  '"confidence": float 0-1, "reason": str}\n\nTHESIS:\n'
                  + json.dumps(th, default=str)[:2000] + "\nNEW EVIDENCE:\n"
                  + json.dumps({"decisions": get_recent_decisions(ticker),
                                "news": get_recent_news(ticker)},
                               default=str)[:4000])
        reply = parse_json_reply(ask("gemini", prompt), REVIEW_SCHEMA)
        if reply and reply.get("status") in ("ACTIVE", "WEAKENING", "CLOSED"):
            get_client().table("theses").update(
                {"status": reply["status"],
                 "confidence": float(reply.get("confidence", th["confidence"])),
                 "updated_at": datetime.now(timezone.utc).isoformat()}).eq("id", th["id"]).execute()
