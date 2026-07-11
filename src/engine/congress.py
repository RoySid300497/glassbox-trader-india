"""reading congressional stock disclosures as smart-money evidence

public stock act filings, delayed by law, but with documented anomalous
returns — fetched once per run from the free stockwatcher datasets and
attached per ticker to the packet
"""

import os
import json
from datetime import date, timedelta

SENATE_URL = ("https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com"
              "/aggregate/all_transactions.json")
LOOKBACK_DAYS = 60
_cache = None


def _load():
    # fetching the aggregate file once per run
    global _cache
    if _cache is not None:
        return _cache
    import requests
    try:
        r = requests.get(SENATE_URL, timeout=30)
        r.raise_for_status()
        _cache = r.json() or []
        print(f"  [congress] {len(_cache)} disclosures loaded")
    except Exception as e:
        print(f"  [congress] feed unavailable: {e}")
        _cache = []
    return _cache


def congress_block(ticker):
    # summarising recent congressional activity in one ticker
    rows = _load()
    if not rows:
        return None
    cutoff = date.today() - timedelta(days=LOOKBACK_DAYS)
    buys, sells, names = 0, 0, set()
    for r in rows:
        if str(r.get("ticker", "")).upper() != ticker:
            continue
        try:
            tx_date = date.fromisoformat(
                str(r.get("transaction_date", ""))[:10])
        except ValueError:
            # some rows use mm/dd/yyyy
            try:
                m, d, y = str(r.get("transaction_date", "")).split("/")
                tx_date = date(int(y), int(m), int(d))
            except Exception:
                continue
        if tx_date < cutoff:
            continue
        kind = str(r.get("type", "")).lower()
        if "purchase" in kind:
            buys += 1
            names.add(r.get("senator") or "?")
        elif "sale" in kind:
            sells += 1
            names.add(r.get("senator") or "?")
    if buys == 0 and sells == 0:
        return None
    return {
        "window_days": LOOKBACK_DAYS,
        "purchases": buys,
        "sales": sells,
        "distinct_members": len(names),
        "note": ("cluster of congressional buying" if buys >= 3 and buys > sells
                 else "cluster of congressional selling" if sells >= 3
                 and sells > buys else "mixed or light activity"),
    }
