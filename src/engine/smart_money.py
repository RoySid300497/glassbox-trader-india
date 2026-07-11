"""summarising insider filings into citable smart-money evidence"""

import os
from datetime import date, timedelta
import requests

FINNHUB_URL = "https://finnhub.io/api/v1/stock/insider-transactions"
LOOKBACK_DAYS = 90           # judging insider behaviour over one quarter
CLUSTER_MIN_BUYERS = 3       # flagging conviction when several insiders buy
OPEN_MARKET_BUY = "P"        # sec code for an open-market purchase
OPEN_MARKET_SELL = "S"       # sec code for an open-market sale


def insider_activity(ticker):
    # aggregating recent form-4 filings into buy and sell pressure
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        return None
    since = str(date.today() - timedelta(days=LOOKBACK_DAYS))
    try:
        r = requests.get(FINNHUB_URL,
                         params={"symbol": ticker.replace(".", "-"),
                                 "from": since, "token": key},
                         timeout=15)
        if r.status_code != 200:
            print(f"  [insider] {ticker}: http {r.status_code} — skipping")
            return None
        rows = (r.json() or {}).get("data") or []
    except Exception as e:
        print(f"  [insider] {ticker}: {e}")
        return None
    if not rows:
        return {"lookback_days": LOOKBACK_DAYS, "open_market_buys": 0,
                "open_market_sells": 0, "note": "no insider filings"}

    buys, sells = [], []
    for t in rows:
        code = t.get("transactionCode")
        shares = abs(t.get("change") or 0)
        price = t.get("transactionPrice") or 0
        if not shares:
            continue
        entry = {"name": t.get("name"), "shares": shares,
                 "value": round(shares * price),
                 "date": t.get("transactionDate")}
        if code == OPEN_MARKET_BUY:
            buys.append(entry)
        elif code == OPEN_MARKET_SELL:
            sells.append(entry)

    distinct_buyers = {b["name"] for b in buys}
    out = {
        "lookback_days": LOOKBACK_DAYS,
        "open_market_buys": len(buys),
        "open_market_sells": len(sells),
        "distinct_buyers": len(distinct_buyers),
        "buy_value_usd": sum(b["value"] for b in buys),
        "sell_value_usd": sum(s["value"] for s in sells),
        "cluster_buying": len(distinct_buyers) >= CLUSTER_MIN_BUYERS,
        "latest_buy": max((b["date"] for b in buys), default=None),
        "largest_buy": max(buys, key=lambda b: b["value"], default=None),
    }
    return out
