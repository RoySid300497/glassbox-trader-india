"""executing gated decisions on alpaca paper with code-computed risk levels"""

import os
import math
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from engine.memory import get_client, validate_ticker

load_dotenv()

PAPER_URL = "https://paper-api.alpaca.markets/v2"
LIVE_URL = "https://api.alpaca.markets/v2"
LIVE_CONFIRM_SENTINEL = "I_UNDERSTAND_REAL_MONEY"
RISK_PER_TRADE = 0.01        # risking one percent of equity per position
STOP_ATR_MULT = 1.5          # placing the stop this many ATRs below entry
REWARD_RISK = 2.0            # placing the runner half's target at two r
PARTIAL_R = 1.0              # banking the scalp half at one r of profit
MAX_POSITION_FRACTION = 0.10  # capping any position at ten percent of equity
MAX_DRAWDOWN_HALT = 0.10     # halting new entries past this peak-to-now drawdown
MAX_HOLD_DAYS = 10           # closing stale positions unless a thesis backs them
EARNINGS_BLACKOUT_DAYS = 2   # refusing fresh risk right before earnings


def trading_mode():
    # resolving paper or live with a double interlock guarding real money
    mode = os.environ.get("TRADING_MODE", "").lower()
    if not mode:
        mode = "paper" if os.environ.get(
            "PAPER_TRADING", "").lower() == "true" else ""
    if mode == "live" and os.environ.get(
            "LIVE_TRADING_CONFIRM") != LIVE_CONFIRM_SENTINEL:
        print("[exec] TRADING_MODE=live set without LIVE_TRADING_CONFIRM "
              "sentinel — refusing live, staying disabled")
        return ""
    return mode if mode in ("paper", "live") else ""


def base_url():
    # selecting the endpoint strictly from the resolved mode
    return LIVE_URL if trading_mode() == "live" else PAPER_URL


def enabled():
    # trading only when a valid mode and both keys are explicitly present
    return bool(trading_mode()
                and os.environ.get("ALPACA_API_KEY")
                and os.environ.get("ALPACA_SECRET_KEY"))


def _headers():
    return {"APCA-API-KEY-ID": os.environ["ALPACA_API_KEY"],
            "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"]}


def _get(path):
    r = requests.get(f"{base_url()}{path}", headers=_headers(), timeout=20)
    r.raise_for_status()
    return r.json()


def get_account():
    # reading paper equity and buying power
    return _get("/account")


def get_positions():
    # listing open paper positions
    return _get("/positions")


def compute_levels(ticker):
    # deriving entry, atr stop, and 2r target from recent daily bars
    import yfinance as yf
    hist = yf.download(ticker.replace(".", "-"), period="2mo",
                       auto_adjust=True, progress=False)
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
    res = get_client().table("portfolio_history").select("equity") \
        .order("equity", desc=True).limit(1).execute().data
    if not res:
        return False
    peak = float(res[0]["equity"])
    current = float(get_account()["equity"])
    return peak > 0 and (current / peak - 1) < -MAX_DRAWDOWN_HALT


def maybe_enter(ticker):
    # opening a long bracket position sized to risk one percent of equity
    if not enabled():
        return "paper trading disabled"
    ticker = validate_ticker(ticker)
    if in_drawdown_halt():
        return f"{ticker}: drawdown halt active — no new entries"

    # skipping when a position already exists for this symbol
    if any(p["symbol"] == ticker.replace(".", "-") for p in get_positions()):
        return f"{ticker}: position already open"

    # refusing new entries right before a binary earnings event
    try:
        from engine.news_fetcher import fetch_next_earnings
        days = fetch_next_earnings(ticker)
        if days is not None and int(days) <= EARNINGS_BLACKOUT_DAYS:
            note = (f"{ticker}: earnings in {int(days)}d — "
                    f"blackout, no new entry")
            print(f"  [paper] {note}")
            return note
    except Exception:
        pass  # never letting the guard itself block trading on a data error

    levels = compute_levels(ticker)
    if levels is None:
        return f"{ticker}: could not compute risk levels"

    equity = float(get_account()["equity"])
    risk_dollars = equity * RISK_PER_TRADE
    per_share_risk = levels["entry"] - levels["stop"]
    qty = math.floor(risk_dollars / per_share_risk)
    max_qty = math.floor(equity * MAX_POSITION_FRACTION / levels["entry"])
    qty = min(qty, max_qty)
    if qty < 1:
        return f"{ticker}: position size below one share"

    scalp_target = round(levels["entry"]
                         + PARTIAL_R * (levels["entry"] - levels["stop"]), 2)

    # splitting into a one-r scalp half and a two-r runner half when possible
    if qty >= 2:
        runner = qty // 2
        scalp = qty - runner
        _place_bracket(ticker, scalp, scalp_target, levels["stop"])
        _place_bracket(ticker, runner, levels["target"], levels["stop"])
        note = (f"{ticker}: bought {qty} @ ~{levels['entry']} "
                f"stop {levels['stop']} — scalp {scalp} tp {scalp_target}, "
                f"runner {runner} tp {levels['target']} "
                f"(atr {levels['atr']})")
    else:
        _place_bracket(ticker, qty, levels["target"], levels["stop"])
        note = (f"{ticker}: bought {qty} @ ~{levels['entry']} "
                f"stop {levels['stop']} target {levels['target']} "
                f"(atr {levels['atr']})")
    print(f"  [paper] {note}")
    return note


def _place_bracket(ticker, qty, target, stop):
    # submitting one bracket order for a slice of the position
    order = {"symbol": ticker.replace(".", "-"), "qty": str(int(qty)),
             "side": "buy", "type": "market", "time_in_force": "day",
             "order_class": "bracket",
             "take_profit": {"limit_price": str(target)},
             "stop_loss": {"stop_price": str(stop)}}
    r = requests.post(f"{base_url()}/orders", headers=_headers(), json=order,
                      timeout=20)
    r.raise_for_status()
    return r.json()


def _cancel_open_orders(symbol):
    # cancelling every open order on one symbol so the position can close
    orders = _get(f"/orders?status=open&symbols={symbol}&limit=100")
    for o in orders:
        try:
            requests.delete(f"{base_url()}/orders/{o['id']}",
                            headers=_headers(), timeout=20).raise_for_status()
        except Exception as e:
            print(f"  [paper] {symbol}: cancel {o['id'][:8]} failed: {e}")
    return len(orders)


def maybe_exit(ticker):
    # closing any open position when the panels vote sell
    if not enabled():
        return "paper trading disabled"
    ticker = validate_ticker(ticker)
    sym = ticker.replace(".", "-")
    if not any(p["symbol"] == sym for p in get_positions()):
        return f"{ticker}: no open position to exit"
    # clearing bracket legs first so alpaca allows the liquidation
    _cancel_open_orders(sym)
    r = requests.delete(f"{base_url()}/positions/{sym}", headers=_headers(),
                        timeout=20)
    r.raise_for_status()
    print(f"  [paper] {ticker}: position closed on SELL vote")
    return f"{ticker}: closed"


def _open_stop_orders(symbol):
    # listing open protective stop legs, including ones nested under parents
    orders = _get(f"/orders?status=open&symbols={symbol}"
                  f"&limit=100&nested=true")
    flat = []
    for o in orders:
        flat.append(o)
        flat.extend(o.get("legs") or [])
    live = ("new", "accepted", "held", "partially_filled")
    return [{"id": o["id"], "stop_price": o["stop_price"]}
            for o in flat
            if o.get("type") in ("stop", "stop_limit")
            and o.get("side") == "sell" and o.get("stop_price")
            and o.get("status") in live]


def _replace_stop(order_id, stop_price):
    # replacing one stop order at the new trailing level
    r = requests.patch(f"{base_url()}/orders/{order_id}", headers=_headers(),
                       json={"stop_price": str(stop_price)}, timeout=20)
    r.raise_for_status()
    return r.json()


def _daily_bars(symbol):
    # fetching a lowercase ohlc frame for trailing stop computation
    import pandas as pd
    import yfinance as yf
    hist = yf.download(symbol, period="6mo", auto_adjust=True, progress=False)
    if hist is None or hist.empty or len(hist) < 30:
        return None
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    return hist.rename(columns=str.lower)[["open", "high", "low", "close"]]


def ratchet_stops():
    # tightening bracket stops toward the chandelier level, never loosening
    if not enabled():
        return []
    from engine.stop_ratchet import ratchet_open_stops
    return ratchet_open_stops(
        list_positions=lambda: [
            {"symbol": p["symbol"],
             "side": "long" if float(p["qty"]) > 0 else "short",
             "qty": p["qty"]} for p in get_positions()],
        list_stop_orders=_open_stop_orders,
        replace_stop=_replace_stop,
        fetch_bars=_daily_bars)


def sync_positions_table():
    # mirroring live alpaca positions while preserving first-seen entry dates
    if not enabled():
        return
    live = get_positions()
    client = get_client()
    existing = {r["ticker"]: r for r in
                (client.table("positions").select("ticker,entry_date")
                 .eq("status", "OPEN").execute().data or [])}
    client.table("positions").update({"status": "CLOSED"}) \
        .eq("status", "OPEN").execute()
    for p in live:
        ticker = p["symbol"].replace("-", ".")
        first_seen = (existing.get(ticker) or {}).get("entry_date") \
            or datetime.now(timezone.utc).isoformat()
        client.table("positions").upsert({
            "ticker": ticker,
            "qty": float(p["qty"]),
            "entry_price": float(p["avg_entry_price"]),
            "entry_date": first_seen,
            "status": "OPEN"}).execute()


def manage_positions():
    # closing positions past the hold limit unless a thesis backs them
    if not enabled():
        return
    from engine.memory import get_active_thesis
    rows = get_client().table("positions").select("ticker,entry_date") \
        .eq("status", "OPEN").execute().data or []
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
            print(f"  [paper] {r['ticker']}: {age}d old, "
                  f"held on active thesis")
            continue
        print(f"  [paper] {r['ticker']}: {age}d exceeds "
              f"{MAX_HOLD_DAYS}d limit — closing")
        maybe_exit(r["ticker"])


def paper_report():
    # summarising paper account performance for the weekly log
    if not enabled():
        print("paper trading: disabled")
        return
    acct = get_account()
    positions = get_positions()
    print(f"paper account: equity ${float(acct['equity']):,.2f} "
          f"(last ${float(acct['last_equity']):,.2f})")
    for p in positions:
        print(f"  open: {p['symbol']} x{p['qty']} "
              f"entry {p['avg_entry_price']} "
              f"unrealized {float(p['unrealized_pl']):+,.2f}")
    if not positions:
        print("  no open positions")


def is_trading_day():
    # asking alpaca whether the market opens at all today
    if not (os.environ.get("ALPACA_API_KEY")
            and os.environ.get("ALPACA_SECRET_KEY")):
        return True
    try:
        from datetime import date
        today = str(date.today())
        cal = _get(f"/calendar?start={today}&end={today}")
        return bool(cal)
    except Exception:
        return True
