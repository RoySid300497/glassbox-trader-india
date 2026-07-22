"""executing gated decisions in SIMULATION mode for the india/NSE system

indian brokers have no alpaca-style paper sandbox, so this module simulates
a paper portfolio itself: it sizes and "places" bracket orders exactly like
the live logic, records them to supabase, and a next-day scoring pass fills
them against real NSE OHLC (respecting circuit bands). the public interface
matches the original alpaca module so run_daily needs no changes.

set TRADING_MODE=simulate to enable. a future execution_angel.py can swap in
real order placement behind a LIVE flag without touching run_daily.
"""

import os
import math
from datetime import datetime, timezone, date
from dotenv import load_dotenv
from engine.memory import get_client, validate_ticker

load_dotenv()

RISK_PER_TRADE = float(os.environ.get("RISK_PER_TRADE", "0.01"))
STOP_ATR_MULT = float(os.environ.get("STOP_ATR_MULT", "1.5"))
REWARD_RISK = float(os.environ.get("REWARD_RISK", "2.0"))
PARTIAL_R = float(os.environ.get("PARTIAL_R", "1.0"))
MAX_POSITION_FRACTION = float(os.environ.get("MAX_POSITION_FRACTION", "0.10"))
MAX_DRAWDOWN_HALT = float(os.environ.get("MAX_DRAWDOWN_HALT", "0.10"))
MAX_HOLD_DAYS = int(os.environ.get("MAX_HOLD_DAYS", "10"))
EARNINGS_BLACKOUT_DAYS = int(os.environ.get("EARNINGS_BLACKOUT_DAYS", "2"))
START_EQUITY = float(os.environ.get("START_EQUITY", "1000000.0"))        # ₹10 lakh notional paper book
CIRCUIT_BAND = float(os.environ.get("CIRCUIT_BAND", "0.10"))               # cap simulated fills within a 10% daily band


def trading_mode():
    # simulate is the india default; live is reserved for the angel client
    mode = os.environ.get("TRADING_MODE", "").lower()
    if not mode:
        mode = "simulate" if os.environ.get(
            "PAPER_TRADING", "").lower() == "true" else ""
    return mode if mode in ("simulate", "paper", "live") else ""


def enabled():
    # simulation needs no broker keys — only that the mode is set
    return trading_mode() in ("simulate", "paper")


def base_url():
    # no live endpoint in simulation; returned for log/interface compatibility
    return "simulation://nse-paper-book"


# ----- virtual account, backed by supabase -------------------------------

def _sim_state():
    # loading (or seeding) the virtual cash/equity record
    client = get_client()
    rows = client.table("config").select("value").eq(
        "key", "sim_equity").execute().data
    if rows:
        try:
            return float(rows[0]["value"])
        except (TypeError, ValueError):
            pass
    client.table("config").upsert(
        {"key": "sim_equity", "value": str(START_EQUITY)}).execute()
    return START_EQUITY


def _set_equity(value):
    get_client().table("config").upsert(
        {"key": "sim_equity", "value": str(round(float(value), 2))}).execute()


def get_account():
    # mirroring the alpaca account shape the rest of the code expects
    eq = _sim_state()
    rows = get_client().table("config").select("value").eq(
        "key", "sim_last_equity").execute().data
    last = float(rows[0]["value"]) if rows else eq
    return {"equity": eq, "last_equity": last, "cash": eq}


def get_positions():
    # returning open sim positions in the alpaca-like field shape
    rows = get_client().table("positions").select("*").eq(
        "status", "OPEN").execute().data or []
    out = []
    for r in rows:
        entry = float(r.get("entry_price") or 0)
        out.append({"symbol": r["ticker"].replace(".", "-"),
                    "qty": float(r.get("qty") or 0),
                    "avg_entry_price": entry,
                    "unrealized_pl": 0.0})
    return out


# ----- risk levels (same math as live) -----------------------------------

def compute_levels(ticker):
    # deriving entry, atr stop, and 2r target from recent NSE daily bars
    import yfinance as yf
    sym = ticker if ticker.endswith(".NS") else ticker + ".NS"
    hist = yf.download(sym, period="2mo", auto_adjust=True, progress=False)
    if hist.empty or len(hist) < 15:
        return None
    high = hist["High"].squeeze()
    low = hist["Low"].squeeze()
    close = hist["Close"].squeeze()
    prev_close = close.shift(1)
    tr = (high - low).combine((high - prev_close).abs(), max) \
        .combine((low - prev_close).abs(), max)
    atr = float(tr.rolling(14).mean().iloc[-1])
    entry = float(close.iloc[-1])
    stop = round(entry - STOP_ATR_MULT * atr, 2)
    target = round(entry + REWARD_RISK * (entry - stop), 2)
    if stop <= 0 or stop >= entry:
        return None
    return {"entry": entry, "stop": stop, "target": target,
            "atr": round(atr, 2)}


def in_drawdown_halt():
    # refusing new risk when equity has fallen too far from its peak
    peak = _peak_equity()
    current = float(get_account()["equity"])
    return peak > 0 and (current / peak - 1) < -MAX_DRAWDOWN_HALT


def _peak_equity():
    rows = get_client().table("config").select("value").eq(
        "key", "sim_peak_equity").execute().data
    seed = float(rows[0]["value"]) if rows else START_EQUITY
    cur = float(_sim_state())
    if cur > seed:
        get_client().table("config").upsert(
            {"key": "sim_peak_equity", "value": str(cur)}).execute()
        return cur
    return seed


# ----- entry (simulated bracket) -----------------------------------------

def maybe_enter(ticker):
    # opening a simulated long bracket sized to risk one percent of equity
    if not enabled():
        return "simulation disabled"
    ticker = validate_ticker(ticker)
    if in_drawdown_halt():
        return f"{ticker}: drawdown halt active — no new entries"
    if any(p["symbol"] == ticker.replace(".", "-") for p in get_positions()):
        return f"{ticker}: position already open"

    try:
        from engine.news_fetcher import fetch_next_earnings
        days = fetch_next_earnings(ticker)
        if days is not None and int(days) <= EARNINGS_BLACKOUT_DAYS:
            note = f"{ticker}: earnings in {int(days)}d — blackout"
            print(f"  [sim] {note}")
            return note
    except Exception:
        pass

    levels = compute_levels(ticker)
    if levels is None:
        return f"{ticker}: could not compute risk levels"

    equity = float(get_account()["equity"])
    risk = equity * RISK_PER_TRADE
    # performance-based risk scaling: when signal-1 (the model layer) is drifting
    # toward random or the models disagree on this ticker, shrink risk. never
    # amplifies (multiplier <= 1.0); toggle with SIGNAL_DERISK=0 to A/B test.
    if os.environ.get("SIGNAL_DERISK", "1") == "1":
        try:
            from engine.signal_health import signal_risk_multiplier
            _mult, _detail = signal_risk_multiplier(ticker)
            if _mult < 1.0:
                risk *= _mult
                print(f"  [signal-health] {ticker}: risk x{_mult} "
                      f"({_detail.get('drift', {}).get('state', '?')})")
        except Exception as _e:
            print(f"  [signal-health] unavailable: {_e}")
    per_share_risk = levels["entry"] - levels["stop"]
    qty = math.floor(risk / per_share_risk)
    max_qty = math.floor(equity * MAX_POSITION_FRACTION / levels["entry"])
    qty = min(qty, max_qty)
    if qty < 1:
        return f"{ticker}: position size below one share"

    # recording the simulated position; fills are scored next day
    get_client().table("positions").upsert({
        "ticker": ticker,
        "qty": float(qty),
        "entry_price": float(levels["entry"]),
        "entry_date": datetime.now(timezone.utc).isoformat(),
        "status": "OPEN"}).execute()
    # stashing the bracket levels for the scorer
    _save_bracket(ticker, levels, qty)

    note = (f"{ticker}: SIM bought {qty} @ ~{levels['entry']} "
            f"stop {levels['stop']} target {levels['target']} "
            f"(atr {levels['atr']})")
    print(f"  [sim] {note}")
    return note


def _save_bracket(ticker, levels, qty):
    import json
    get_client().table("config").upsert({
        "key": f"sim_bracket:{ticker}",
        "value": json.dumps({"stop": levels["stop"],
                             "target": levels["target"],
                             "entry": levels["entry"], "qty": qty})}).execute()


# ----- exit + next-day fill scoring --------------------------------------

def maybe_exit(ticker):
    # closing a simulated position at the latest close
    if not enabled():
        return "simulation disabled"
    ticker = validate_ticker(ticker)
    px = _latest_close(ticker)
    _close_position(ticker, px, reason="manual")
    return f"{ticker}: SIM closed @ ~{px}"


def score_open_positions():
    # the daily fill simulator: for each open position, pull the latest bar
    # and decide if stop/target hit, capping fills within the circuit band
    import json
    if not enabled():
        return
    client = get_client()
    rows = client.table("positions").select("*").eq(
        "status", "OPEN").execute().data or []
    for r in rows:
        ticker = r["ticker"]
        bar = _latest_bar(ticker)
        if bar is None:
            continue
        bk_rows = client.table("config").select("value").eq(
            "key", f"sim_bracket:{ticker}").execute().data
        if not bk_rows:
            continue
        bk = json.loads(bk_rows[0]["value"])
        entry, qty = bk["entry"], bk["qty"]
        lo = max(bar["low"], entry * (1 - CIRCUIT_BAND))
        hi = min(bar["high"], entry * (1 + CIRCUIT_BAND))
        if lo <= bk["stop"]:
            _close_position(ticker, bk["stop"], "stop", qty, entry)
        elif hi >= bk["target"]:
            _close_position(ticker, bk["target"], "target", qty, entry)
        # else: still open, carry to next day


def _close_position(ticker, exit_px, reason, qty=None, entry=None):
    client = get_client()
    if qty is None or entry is None:
        pos = client.table("positions").select("*").eq(
            "ticker", ticker).eq("status", "OPEN").execute().data
        if not pos:
            return
        qty = float(pos[0]["qty"])
        entry = float(pos[0]["entry_price"])
    pnl = (float(exit_px) - float(entry)) * float(qty)
    # updating virtual equity
    eq = float(_sim_state())
    client.table("config").upsert(
        {"key": "sim_last_equity", "value": str(eq)}).execute()
    _set_equity(eq + pnl)
    client.table("positions").update(
        {"status": "CLOSED"}).eq("ticker", ticker).eq(
        "status", "OPEN").execute()
    client.table("config").delete().eq(
        "key", f"sim_bracket:{ticker}").execute()
    print(f"  [sim] {ticker} closed ({reason}) @ {exit_px} "
          f"pnl {pnl:+,.2f}")


# ----- market data helpers -----------------------------------------------

def _latest_bar(ticker):
    import yfinance as yf
    sym = ticker if ticker.endswith(".NS") else ticker + ".NS"
    h = yf.download(sym, period="5d", auto_adjust=True, progress=False)
    if h.empty:
        return None
    last = h.iloc[-1]
    return {"high": float(last["High"].squeeze()),
            "low": float(last["Low"].squeeze()),
            "close": float(last["Close"].squeeze())}


def _latest_close(ticker):
    bar = _latest_bar(ticker)
    return bar["close"] if bar else None


# ----- position lifecycle (same interface as live) -----------------------

def sync_positions_table():
    # in simulation the positions table IS the source of truth, so this is
    # a no-op kept for interface compatibility
    return


def manage_positions():
    # closing positions past the hold limit unless a thesis backs them
    if not enabled():
        return
    from engine.memory import get_active_thesis
    rows = get_client().table("positions").select(
        "ticker,entry_date").eq("status", "OPEN").execute().data or []
    now = datetime.now(timezone.utc)
    for r in rows:
        try:
            age = (now - datetime.fromisoformat(
                r["entry_date"].replace("Z", "+00:00"))).days
        except Exception:
            continue
        if age <= MAX_HOLD_DAYS:
            continue
        thesis = get_active_thesis(r["ticker"])
        if thesis and thesis["direction"] == "LONG":
            continue
        print(f"  [sim] {r['ticker']}: {age}d exceeds "
              f"{MAX_HOLD_DAYS}d — closing")
        maybe_exit(r["ticker"])


def ratchet_stops():
    # trailing the stop up as price advances, on the stored bracket levels
    import json
    if not enabled():
        return
    client = get_client()
    rows = client.table("positions").select("*").eq(
        "status", "OPEN").execute().data or []
    for r in rows:
        ticker = r["ticker"]
        bar = _latest_bar(ticker)
        bk_rows = client.table("config").select("value").eq(
            "key", f"sim_bracket:{ticker}").execute().data
        if bar is None or not bk_rows:
            continue
        bk = json.loads(bk_rows[0]["value"])
        entry = bk["entry"]
        # once price is up one r, lift the stop to breakeven
        r_dist = entry - bk["stop"]
        if r_dist > 0 and bar["close"] >= entry + r_dist and bk["stop"] < entry:
            bk["stop"] = round(entry, 2)
            client.table("config").upsert(
                {"key": f"sim_bracket:{ticker}",
                 "value": json.dumps(bk)}).execute()
            print(f"  [sim] {ticker}: stop ratcheted to breakeven")


def paper_report():
    # summarising the simulated book for the weekly log
    if not enabled():
        print("simulation: disabled")
        return
    acct = get_account()
    positions = get_positions()
    print(f"sim account: equity ₹{acct['equity']:,.2f} "
          f"(last ₹{acct['last_equity']:,.2f})")
    for p in positions:
        print(f"  open: {p['symbol']} x{p['qty']} "
              f"entry {p['avg_entry_price']}")
    if not positions:
        print("  no open positions")


def is_trading_day():
    # dynamic NSE session check — no hardcoded holiday list to maintain yearly.
    # primary: exchange_calendars (maintained NSE/BSE calendar, auto-updates).
    # fallback: yfinance (did a liquid index actually trade today?).
    # final fallback: weekday check. errs toward OPEN only as a last resort.
    today = date.today()
    if today.weekday() >= 5:
        return False

    # primary — maintained exchange calendar (XBOM covers NSE/BSE holidays)
    try:
        import exchange_calendars as xcals
        import pandas as pd
        cal = xcals.get_calendar("XBOM")
        return bool(cal.is_session(pd.Timestamp(today)))
    except Exception:
        pass

    # fallback — did the Nifty 50 actually trade today? (holidays => no bar)
    try:
        import yfinance as yf
        hist = yf.download("^NSEI", period="5d", auto_adjust=True,
                           progress=False)
        if hist is not None and len(hist):
            import pandas as pd
            last_session = pd.Timestamp(hist.index[-1]).date()
            return last_session == today
    except Exception:
        pass

    # last resort — weekday already passed above; assume open
    return True

