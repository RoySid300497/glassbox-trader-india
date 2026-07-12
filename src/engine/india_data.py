"""india-specific market data: FII/DII flows, India VIX, Nifty regime,
and delivery percentage — free NSE-published signals with no US analog

every fetch is defensive with a cached fallback so a feed outage or an NSE
endpoint change never blocks a run; missing data degrades to neutral values
rather than raising. all values are point-in-time (as-of a date) with no
lookahead
"""

import os
import json
import time
from datetime import datetime, timezone

import pandas as pd

_CACHE_TTL = 3600            # seconds to trust an in-process cache
_cache = {}


def _cached(key, fn):
    # a tiny time-boxed memoiser so repeated calls in one run hit the net once
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < _CACHE_TTL:
        return hit[1]
    val = fn()
    _cache[key] = (now, val)
    return val


# ----- India VIX (NSE fear gauge, via yfinance ^INDIAVIX) -----------------

def india_vix_series(period="2y"):
    # pulling the India VIX history; used as a market-regime feature
    def _f():
        import yfinance as yf
        for sym in ("^INDIAVIX", "INDIAVIX.NS"):
            try:
                h = yf.download(sym, period=period, auto_adjust=True,
                                progress=False)
                if not h.empty:
                    s = h["Close"].squeeze()
                    s.index = pd.to_datetime(s.index).tz_localize(None)
                    return s.rename("india_vix")
            except Exception:
                continue
        return pd.Series(dtype="float64", name="india_vix")
    return _cached("india_vix", _f)


def latest_vix():
    s = india_vix_series(period="3mo")
    return float(s.iloc[-1]) if len(s) else None


# ----- Nifty 50 index regime (^NSEI) --------------------------------------

def nifty_series(period="2y"):
    # the nifty 50 index itself, for a market-trend regime feature
    def _f():
        import yfinance as yf
        try:
            h = yf.download("^NSEI", period=period, auto_adjust=True,
                            progress=False)
            if not h.empty:
                s = h["Close"].squeeze()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                return s.rename("nifty_close")
        except Exception:
            pass
        return pd.Series(dtype="float64", name="nifty_close")
    return _cached("nifty", _f)


def nifty_regime():
    # returning a simple regime read: is nifty above its 50d MA, and its
    # 20d return — the "check market sentiment first" rule from the corpus
    s = nifty_series(period="6mo")
    if len(s) < 55:
        return {"above_ma50": None, "ret_20d": None, "trend": "unknown"}
    ma50 = s.rolling(50).mean().iloc[-1]
    above = bool(s.iloc[-1] > ma50)
    ret20 = float(s.iloc[-1] / s.iloc[-21] - 1) if len(s) >= 21 else None
    trend = "up" if above and (ret20 or 0) > 0 else (
        "down" if not above and (ret20 or 0) < 0 else "mixed")
    return {"above_ma50": above, "ret_20d": ret20, "trend": trend}


# ----- FII / DII institutional flows (NSE, the headline india signal) -----

_FII_DII_CACHE = None


def fii_dii_flows():
    # fetching recent FII/DII net buy/sell; NSE publishes this daily.
    # tries the NSE json endpoint, falls back to a cached supabase copy so a
    # blocked request never blinds the packet
    def _f():
        import requests
        url = ("https://www.nseindia.com/api/fiidiiTradeReact")
        headers = {
            "User-Agent": "Mozilla/5.0 (glassbox-india)",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/",
        }
        try:
            sess = requests.Session()
            # NSE requires a homepage hit first to set cookies
            sess.get("https://www.nseindia.com", headers=headers, timeout=10)
            r = sess.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            out = {}
            for row in data:
                cat = (row.get("category") or "").upper()
                net = float(row.get("netValue") or 0)
                if "FII" in cat or "FPI" in cat:
                    out["fii_net"] = net
                elif "DII" in cat:
                    out["dii_net"] = net
            if out:
                _persist_flows(out)
                return out
        except Exception as e:
            print(f"  [india] FII/DII live fetch failed ({e}); using cache")
        return _load_cached_flows()
    return _cached("fii_dii", _f)


def _persist_flows(flows):
    # storing the latest flows in supabase config so a later blocked fetch
    # can still read yesterday's numbers
    try:
        from engine.memory import get_client
        get_client().table("config").upsert({
            "key": "india_fii_dii",
            "value": json.dumps({**flows,
                                 "as_of": datetime.now(
                                     timezone.utc).isoformat()})}).execute()
    except Exception:
        pass


def _load_cached_flows():
    try:
        from engine.memory import get_client
        rows = get_client().table("config").select("value").eq(
            "key", "india_fii_dii").execute().data
        if rows:
            d = json.loads(rows[0]["value"])
            return {"fii_net": d.get("fii_net"), "dii_net": d.get("dii_net")}
    except Exception:
        pass
    return {"fii_net": None, "dii_net": None}


# ----- delivery percentage (NSE, conviction signal) -----------------------

def delivery_pct(ticker):
    # % of traded volume taken to delivery — a conviction read unique to
    # indian data; defensive, returns None when unavailable
    def _f():
        import requests
        sym = ticker.replace(".NS", "").replace(".BO", "").upper()
        url = (f"https://www.nseindia.com/api/quote-equity?symbol={sym}"
               f"&section=trade_info")
        headers = {"User-Agent": "Mozilla/5.0 (glassbox-india)",
                   "Accept": "application/json",
                   "Referer": "https://www.nseindia.com/"}
        try:
            sess = requests.Session()
            sess.get("https://www.nseindia.com", headers=headers, timeout=10)
            r = sess.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            sec = data.get("securityWiseDP", {})
            return float(sec.get("deliveryToTradedQuantity") or 0) or None
        except Exception:
            return None
    return _cached(f"delivery:{ticker}", _f)


# ----- a single market-context block for the judge packet -----------------

def india_market_context():
    # assembling one block: nifty regime + VIX + FII/DII, all defensive
    regime = nifty_regime()
    vix = latest_vix()
    flows = fii_dii_flows()
    return {
        "nifty_trend": regime["trend"],
        "nifty_above_ma50": regime["above_ma50"],
        "nifty_ret_20d": (round(regime["ret_20d"], 4)
                          if regime["ret_20d"] is not None else None),
        "india_vix": round(vix, 2) if vix is not None else None,
        "fii_net_cr": (round(flows["fii_net"], 1)
                       if flows.get("fii_net") is not None else None),
        "dii_net_cr": (round(flows["dii_net"], 1)
                       if flows.get("dii_net") is not None else None),
    }
