"""running the morning decision loop, outcome scoring, and thesis review"""

import os
import argparse
from datetime import datetime, timezone
from engine.news_fetcher import fetch_and_archive
from engine.data_packet import build_packet
from engine.protocol import decide
from engine.risk_gate import apply_gate
from engine.thesis import propose_thesis, review_theses
from engine.lessons import distill_lessons
from engine.champion import elect_champion, get_champion
from engine.screener import select_watchlist
from engine.shadow import (record_predictions,
                           score_model_predictions, model_report)
from engine.execution import (maybe_enter, maybe_exit,
                              sync_positions_table, paper_report, enabled,
                              is_trading_day, manage_positions,
                              ratchet_stops)
from engine.memory import (insert_decision, get_unscored_decisions,
                           score_decision, upsert_market_context,
                           validate_ticker, save_screen_results,
                           get_recent_tickers, prune_news, get_watchlist)

def _envint(name, default):
    # reading an int env var, tolerating empty or malformed values
    raw = os.environ.get(name, "")
    try:
        return int(raw) if raw.strip() else default
    except ValueError:
        return default

DEBATE_BUDGET = _envint("DEBATE_BUDGET", 10)
DEBATE_COOLDOWN_DAYS = _envint("DEBATE_COOLDOWN_DAYS", 2)
DEBATE_PAUSE_SEC = _envint("DEBATE_PAUSE_SEC", 20)
SCAN_LIMIT = os.environ.get("SCAN_LIMIT")   # optional ticker cap for testing
WATCHLIST = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM"]   # fallback only


def run_ticker(ticker, source="technical"):
    # deciding one ticker: news, packet, debate, gate, record
    ticker = validate_ticker(ticker)
    print(f"\n--- {ticker} ---")
    news = fetch_and_archive(ticker)
    packet = build_packet(ticker, news)
    verdict = decide(packet)
    action, note = apply_gate(ticker, verdict)
    print(f"panel: {verdict['decision']} | gate: {action} | {note}")

    sig = packet["cnn_signal"]
    insert_decision(ticker, action, sig.get("direction", "unavailable"),
                    sig.get("confidence", 0.0), verdict["bull_case"],
                    verdict["bear_case"], verdict["judge_votes"], note,
                    selection_source=source)

    # executing on paper only when the flag and keys are present
    if enabled():
        try:
            if action == "BUY":
                maybe_enter(ticker)
            elif action == "SELL":
                maybe_exit(ticker)
        except Exception as e:
            print(f"  [paper] {ticker} execution failed: {e}")
    return action


def market_summary():
    # composing a factual market snapshot from index and volatility data
    import yfinance as yf
    parts = []
    for name, sym in [("S&P 500", "SPY"), ("Nasdaq 100", "QQQ")]:
        try:
            closes = yf.download(sym, period="10d", auto_adjust=True,
                                 progress=False)["Close"].squeeze()
            d1 = closes.pct_change().iloc[-1] * 100
            d5 = closes.pct_change(5).iloc[-1] * 100
            parts.append(f"{name} {d1:+.1f}% last session, {d5:+.1f}% over 5 days")
        except Exception:
            continue
    try:
        vix = float(yf.download("^VIX", period="5d", auto_adjust=True,
                                progress=False)["Close"].squeeze().iloc[-1])
        mood = "calm" if vix < 15 else "elevated" if vix < 25 else "stressed"
        parts.append(f"VIX at {vix:.1f} ({mood})")
    except Exception:
        pass
    return "; ".join(parts) if parts else "market data unavailable"


def pick_watchlist(recently_debated, limit):
    # combining user-pinned tickers with screener picks per the fill mode
    pins, fill = [], "screener"
    try:
        pins, fill = get_watchlist()
    except Exception as e:
        print(f"[picks] watchlist config unavailable: {e}")

    picks = []
    sources = {}
    for t in pins:
        try:
            t = validate_ticker(t)
        except ValueError:
            continue
        if t not in picks:
            picks.append(t)
            sources[t] = "user_pin"
    picks = picks[:DEBATE_BUDGET]

    # letting the macro scout nominate headline-driven names after user
    # pins — automated pins respect the cooldown, user pins never do
    if fill != "empty":
        try:
            from engine.macro_scout import macro_pins
            for t, why in macro_pins():
                if t not in picks and t not in recently_debated \
                        and len(picks) < DEBATE_BUDGET:
                    picks.append(t)
                    sources[t] = "macro_scout"
                    print(f"[picks] macro scout pinned {t}: {why}")
        except Exception as e:
            print(f"[picks] macro scout unavailable: {e}")

    # news scout: names whose coverage is loud even if their charts are
    # quiet — capped slots, cooldown respected, provenance recorded
    if fill != "empty":
        try:
            from engine.news_scout import news_pins
            for t, why in news_pins(exclude=recently_debated | set(picks)):
                if len(picks) < DEBATE_BUDGET:
                    picks.append(t)
                    sources[t] = "news_scout"
                    print(f"[picks] news scout pinned {t}: {why}")
        except Exception as e:
            print(f"[picks] news scout unavailable: {e}")

    # always scanning so screen_results and the site's scan page stay fresh
    need = DEBATE_BUDGET - len(picks)
    k = DEBATE_BUDGET if (fill == "empty" or need <= 0) else need
    exclude = recently_debated | set(picks)
    screener_picks, scan = select_watchlist(k=k, limit=limit, exclude=exclude)

    if fill == "empty" and picks:
        return picks, picks, scan, sources
    watchlist = (picks
                 + [t for t in screener_picks if t not in picks])[:DEBATE_BUDGET]
    for t in watchlist:
        sources.setdefault(t, "technical")
    return watchlist, picks, scan, sources


def run_daily():
    # scanning the universe, then debating only the most interesting tickers
    if not is_trading_day():
        print("market holiday — skipping today's run")
        return

    # exiting when today already ran, making catch-up crons safe
    from datetime import date as _date
    from engine.memory import get_client
    already = get_client().table("decisions").select("id") \
        .gte("decided_at", str(_date.today())).limit(1).execute().data
    if already:
        print("decisions already exist today — skipping duplicate run")
        return

    if enabled():
        from engine.execution import trading_mode, base_url
        print(f"[exec] mode={trading_mode().upper()} endpoint={base_url()}")
    upsert_market_context(market_summary())
    from engine.news_fetcher import archive_macro_news
    archive_macro_news()

    limit = int(SCAN_LIMIT) if SCAN_LIMIT and str(SCAN_LIMIT).strip() else None
    recently_debated = set(get_recent_tickers(days=DEBATE_COOLDOWN_DAYS))
    watchlist, picks, scan, sources = pick_watchlist(recently_debated, limit)
    if scan:
        save_screen_results(scan, top_n=max(40, DEBATE_BUDGET + 10))
    print(f"debating today: {watchlist} "
          f"(pinned: {picks or 'none'}, "
          f"excluded {len(recently_debated)} on cooldown)")

    import time
    results = {}
    for i, ticker in enumerate(watchlist):
        try:
            results[ticker] = run_ticker(
                ticker, source=sources.get(ticker, "technical"))
            record_predictions(ticker)
        except Exception as e:
            print(f"{ticker} failed: {e}")
            results[ticker] = "ERROR"
        # pausing between debates to stay under per-minute rate limits
        if i < len(watchlist) - 1:
            time.sleep(DEBATE_PAUSE_SEC)

    if enabled():
        try:
            sync_positions_table()
        except Exception as e:
            print(f"[paper] position sync failed: {e}")
        from engine.performance import sync_performance
        sync_performance()
        try:
            manage_positions()
        except Exception as e:
            print(f"[paper] position management failed: {e}")
        try:
            moved = ratchet_stops()
            if moved:
                print(f"  [paper] ratcheted {len(moved)} stop(s)")
        except Exception as e:
            print(f"[paper] stop ratchet failed: {e}")
    print(f"\ndaily run complete: {results}")


def run_manage():
    # trailing stops and enforcing exits without scanning or debating
    if not is_trading_day():
        print("market holiday — skipping midday management")
        return
    if not enabled():
        print("trading disabled — nothing to manage")
        return
    try:
        sync_positions_table()
    except Exception as e:
        print(f"[manage] position sync failed: {e}")
    try:
        moved = ratchet_stops()
        print(f"[manage] ratcheted {len(moved)} stop(s)" if moved
              else "[manage] no stops earned a move")
    except Exception as e:
        print(f"[manage] stop ratchet failed: {e}")
    try:
        manage_positions()
    except Exception as e:
        print(f"[manage] position management failed: {e}")
    print("midday management complete")


def latest_prices(tickers):
    # fetching latest closes and returns for scoring and thesis review
    import yfinance as yf
    out = {}
    for t in tickers:
        try:
            hist = yf.download(t.replace(".", "-"), period="10d",
                               auto_adjust=True, progress=False)
            closes = hist["Close"].squeeze()
            out[t] = {"close": float(closes.iloc[-1]),
                      "ret_1d": float(closes.pct_change().iloc[-1]),
                      "ret_5d": float(closes.pct_change(5).iloc[-1])
                      if len(closes) > 5 else None}
        except Exception:
            out[t] = None
    return out


def score_outcomes():
    # scoring each decision against the first trading day after it was made
    import pandas as pd
    import yfinance as yf
    pending = get_unscored_decisions()
    if not pending:
        print("nothing to score")
        return
    for d in pending:
        try:
            decided = pd.Timestamp(d["decided_at"]).tz_localize(None)
            hist = yf.download(d["ticker"].replace(".", "-"), period="1mo",
                               auto_adjust=True, progress=False)
            closes = hist["Close"].squeeze()
            closes.index = pd.to_datetime(closes.index).tz_localize(None)

            # locating the last close at or before the decision moment
            before = closes[closes.index <= decided]
            after = closes[closes.index > decided]
            if before.empty or after.empty:
                print(f"skipping {d['ticker']} #{d['id']}: "
                      f"next trading day not complete yet")
                continue

            base = float(before.iloc[-1])
            ret_1d = float(after.iloc[0]) / base - 1
            ret_5d = (float(after.iloc[4]) / base - 1) if len(after) > 4 else None

            label = ("Up" if ret_1d > 0.01
                     else "Down" if ret_1d < -0.01 else "Neutral")
            correct = (d["action"] == "BUY" and label == "Up") or \
                      (d["action"] == "SELL" and label == "Down") or \
                      (d["action"] == "NO_TRADE" and label == "Neutral")
            score_decision(d["id"], ret_1d, ret_5d, label, correct)
            print(f"scored {d['ticker']} #{d['id']}: {label} "
                  f"({'correct' if correct else 'wrong'})")
        except Exception as e:
            print(f"skipping {d['ticker']} #{d['id']}: {e}")


def performance_report(window=60):
    # summarising panel accuracy and cnn drift over recent scored decisions
    from engine.memory import get_client
    rows = get_client().table("decisions") \
        .select("action,was_correct,cnn_direction,outcome_label") \
        .not_.is_("scored_at", "null") \
        .order("decided_at", desc=True).limit(int(window)).execute().data or []
    if not rows:
        print("performance: no scored decisions yet")
        return
    trades = [r for r in rows if r["action"] != "NO_TRADE"]
    holds = [r for r in rows if r["action"] == "NO_TRADE"]
    cnn_hits = sum(1 for r in rows if r["cnn_direction"] == r["outcome_label"])
    print(f"performance (last {len(rows)} scored):")
    print(f"  trades correct : "
          f"{sum(1 for r in trades if r['was_correct'])}/{len(trades)}")
    print(f"  holds correct  : "
          f"{sum(1 for r in holds if r['was_correct'])}/{len(holds)} "
          f"(missed moves: {sum(1 for r in holds if not r['was_correct'])})")
    print(f"  cnn hit rate   : {cnn_hits}/{len(rows)} "
          f"(random baseline ~{len(rows)//3}) — retrain when this sags")


def write_weekly_report():
    # assembling one machine-readable report row for the site
    from engine.memory import get_client, save_weekly_report
    from datetime import datetime, timedelta, timezone
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    dec = get_client().table("decisions") \
        .select("action,was_correct,scored_at") \
        .gte("decided_at", week_ago).execute().data or []
    scored = [d for d in dec if d["scored_at"]]
    preds = get_client().table("model_predictions") \
        .select("model,was_correct").not_.is_("scored_at", "null") \
        .gte("pred_date", week_ago[:10]).execute().data or []
    models = {}
    for p in preds:
        m = models.setdefault(p["model"], {"correct": 0, "scored": 0})
        m["scored"] += 1
        m["correct"] += 1 if p["was_correct"] else 0
    lessons_new = get_client().table("lessons").select("lesson_text") \
        .gte("created_at", week_ago).execute().data or []
    ninety = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    hist = get_client().table("decisions") \
        .select("action,outcome_return_1d") \
        .not_.is_("scored_at", "null").gte("decided_at", ninety) \
        .execute().data or []
    buy_rets = [h["outcome_return_1d"] for h in hist
                if h["action"] == "BUY" and h["outcome_return_1d"] is not None]
    sell_rets = [h["outcome_return_1d"] for h in hist
                 if h["action"] == "SELL" and h["outcome_return_1d"] is not None]
    stats = {"decisions": len(dec),
             "champion": get_champion(),
             "avg_1d_after_buy": round(sum(buy_rets) / len(buy_rets), 4)
             if buy_rets else None,
             "avg_1d_after_sell": round(sum(sell_rets) / len(sell_rets), 4)
             if sell_rets else None,
             "scored": len(scored),
             "correct": sum(1 for d in scored if d["was_correct"]),
             "trades": sum(1 for d in dec if d["action"] != "NO_TRADE"),
             "models": models,
             "new_lessons": [l["lesson_text"][:200] for l in lessons_new]}
    if enabled():
        try:
            from engine.execution import get_account
            acct = get_account()
            stats["equity"] = float(acct["equity"])
            stats["last_equity"] = float(acct["last_equity"])
        except Exception:
            pass
    save_weekly_report(stats)
    print("weekly report row saved")


def weekly_review():
    # scoring outcomes, reporting performance, reviewing recent tickers
    score_outcomes()
    score_model_predictions()
    performance_report()
    model_report()
    paper_report()
    distill_lessons()
    elect_champion()
    from engine.retrain_trigger import maybe_trigger_retrain
    maybe_trigger_retrain()
    try:
        from engine.evidence_report import evidence_report
        evidence_report()
    except Exception as e:
        print(f"evidence report failed: {e}")
    write_weekly_report()
    if enabled():
        from engine.performance import sync_performance
        sync_performance()
    tickers = get_recent_tickers(days=30) or WATCHLIST
    prices = latest_prices(tickers)
    review_theses(lambda t: (prices.get(t) or {}).get("ret_5d"))
    for ticker in tickers:
        proposed = propose_thesis(ticker)
        if proposed:
            print(f"new thesis for {ticker}: {proposed['thesis_text'][:80]}")
    prune_news(years=5)
    print("weekly review complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="daily",
                        choices=["daily", "score", "weekly", "manage"])
    args = parser.parse_args()
    if args.mode == "daily":
        run_daily()
    elif args.mode == "manage":
        run_manage()
    elif args.mode == "score":
        score_outcomes()
        score_model_predictions()
    else:
        weekly_review()


if __name__ == "__main__":
    main()
