"""fetching NSE participant-wise F&O positioning and bulk deals.

Lives at src/engine/nse_flow.py. Run manually first to TEST ACCESS:
    export PYTHONPATH=$PWD/src
    python -m engine.nse_flow --days 5

NSE aggressively blocks datacenter/non-browser traffic, so this may work
from your laptop but 403 on a GitHub Actions runner — the --days test run
tells you which world you're in before anything gets built on top of it.

Two sources, both free daily files on the archives host (historically more
lenient than nseindia.com itself):
  1. participant-wise F&O open interest (FII/DII/Pro/Client long-short) —
     forward-looking POSITIONING, not price history
  2. bulk deals — large institutional prints, per ticker

Stores into Supabase tables fii_positioning and bulk_deals (see the SQL in
the deploy notes). Fail-soft everywhere: a blocked or missing day is
skipped with a message, never a crash.
"""

import argparse
import io
import time
from datetime import date, timedelta

import requests

BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0 Safari/537.36"),
    "Accept": "text/csv,application/csv,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

PARTICIPANT_OI_URL = ("https://archives.nseindia.com/content/nsccl/"
                      "fao_participant_oi_{ddmmyyyy}.csv")
BULK_DEALS_URL = "https://archives.nseindia.com/content/equities/bulk.csv"


def _get(url, timeout=25):
    r = requests.get(url, headers=BROWSER_HEADERS, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code} for {url.rsplit('/', 1)[-1]}")
    return r


def fetch_participant_oi(day):
    # one day's participant-wise OI: rows FII/DII/Pro/Client with long/short
    # contract counts across index futures/options and stock futures/options
    import pandas as pd
    url = PARTICIPANT_OI_URL.format(ddmmyyyy=day.strftime("%d%m%Y"))
    r = _get(url)
    # the file has a title line above the header — sniff for the header row
    text = r.text
    lines = text.splitlines()
    header_i = next((i for i, l in enumerate(lines)
                     if "Client Type" in l), 0)
    df = pd.read_csv(io.StringIO("\n".join(lines[header_i:])))
    df.columns = [c.strip() for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        ctype = str(row.get("Client Type", "")).strip()
        if ctype.lower() in ("", "total", "nan"):
            continue
        def _n(col):
            try:
                return int(float(str(row.get(col, 0)).replace(",", "")))
            except Exception:
                return 0
        rows.append({
            "trade_date": str(day),
            "client_type": ctype,
            "fut_idx_long": _n("Future Index Long"),
            "fut_idx_short": _n("Future Index Short"),
            "fut_stk_long": _n("Future Stock Long"),
            "fut_stk_short": _n("Future Stock Short"),
            "opt_idx_call_long": _n("Option Index Call Long"),
            "opt_idx_put_long": _n("Option Index Put Long"),
            "opt_idx_call_short": _n("Option Index Call Short"),
            "opt_idx_put_short": _n("Option Index Put Short"),
        })
    return rows


def fetch_bulk_deals():
    # the rolling bulk-deals CSV (recent window); per-ticker institutional
    # prints with buy/sell side, quantity, and price
    import pandas as pd
    r = _get(BULK_DEALS_URL)
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        try:
            d = pd.to_datetime(str(row.get("Date", "")).strip(),
                               dayfirst=True).date()
        except Exception:
            continue
        rows.append({
            "trade_date": str(d),
            "symbol": str(row.get("Symbol", "")).strip(),
            "client_name": str(row.get("Client Name", "")).strip()[:200],
            "side": str(row.get("Buy/Sell", "")).strip().upper()[:4],
            "quantity": int(float(str(row.get("Quantity Traded", 0))
                                  .replace(",", "") or 0)),
            "avg_price": float(str(row.get("Trade Price / Wght. Avg. Price",
                                           0)).replace(",", "") or 0),
        })
    return rows


def store(table, rows, conflict):
    if not rows:
        return 0
    from engine.memory import get_client
    get_client().table(table).upsert(rows, on_conflict=conflict).execute()
    return len(rows)


def run(days_back=5, store_db=True):
    ok_oi, ok_bulk = 0, 0
    d = date.today()
    tried = 0
    while tried < days_back:
        d -= timedelta(days=1)
        if d.weekday() >= 5:
            continue
        tried += 1
        try:
            rows = fetch_participant_oi(d)
            if store_db:
                ok_oi += store("fii_positioning", rows,
                               "trade_date,client_type")
            print(f"[nse_flow] participant OI {d}: {len(rows)} rows"
                  + ("" if store_db else " (not stored)"))
        except Exception as e:
            print(f"[nse_flow] participant OI {d}: FAILED ({e})")
        time.sleep(1)
    try:
        rows = fetch_bulk_deals()
        if store_db:
            ok_bulk = store("bulk_deals", rows,
                            "trade_date,symbol,client_name,side")
        print(f"[nse_flow] bulk deals: {len(rows)} rows"
              + ("" if store_db else " (not stored)"))
    except Exception as e:
        print(f"[nse_flow] bulk deals: FAILED ({e})")
    print(f"[nse_flow] stored: {ok_oi} OI rows, {ok_bulk} bulk-deal rows")
    if ok_oi == 0 and ok_bulk == 0 and store_db:
        print("[nse_flow] NOTHING fetched — if running on a GitHub runner, "
              "NSE is likely blocking the datacenter IP; run from your own "
              "machine or via an aggregator instead")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--no-store", action="store_true",
                    help="fetch-only access test, no Supabase writes")
    a = ap.parse_args()
    run(days_back=a.days, store_db=not a.no_store)
