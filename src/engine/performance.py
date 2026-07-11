"""syncing paper equity history and closed trades for the performance page"""

from collections import defaultdict, deque
from datetime import datetime, timezone
from engine.execution import enabled, _get
from engine.memory import get_client


def sync_equity_history():
    # storing the daily paper equity curve from alpaca portfolio history
    if not enabled():
        return
    hist = _get("/account/portfolio/history?period=1A&timeframe=1D")
    stamps = hist.get("timestamp") or []
    equity = hist.get("equity") or []
    by_day = {}
    for ts, eq in zip(stamps, equity):
        if eq is None or float(eq) <= 0:
            continue
        day = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        by_day[str(day)] = float(eq)
    rows = [{"date": d, "equity": e} for d, e in sorted(by_day.items())]
    if rows:
        get_client().table("portfolio_history").upsert(rows).execute()
    print(f"[perf] equity history synced: {len(rows)} days")


def sync_closed_trades():
    # reconstructing round-trips from fills with a fifo lot matcher
    if not enabled():
        return
    fills = _get("/account/activities/FILL?page_size=100")
    fills = sorted(fills, key=lambda f: f["transaction_time"])

    lots = defaultdict(deque)
    closed = []
    for f in fills:
        symbol = f["symbol"]
        qty = float(f["qty"])
        price = float(f["price"])
        when = f["transaction_time"]
        if f["side"] == "buy":
            lots[symbol].append({"qty": qty, "price": price, "at": when})
            continue
        # matching this sell against open lots first-in-first-out
        remaining = qty
        while remaining > 0 and lots[symbol]:
            lot = lots[symbol][0]
            take = min(remaining, lot["qty"])
            pnl = (price - lot["price"]) * take
            closed.append({
                "exit_fill_id": f["id"],
                "ticker": symbol.replace("-", "."),
                "qty": take,
                "entry_price": lot["price"],
                "exit_price": price,
                "entry_at": lot["at"],
                "exit_at": when,
                "pnl": round(pnl, 2),
                "pnl_pct": round(price / lot["price"] - 1, 4),
            })
            lot["qty"] -= take
            remaining -= take
            if lot["qty"] <= 0:
                lots[symbol].popleft()

    if closed:
        get_client().table("trades").upsert(closed).execute()
    print(f"[perf] closed trades synced: {len(closed)}")


def sync_performance():
    # running both syncs with independent failure isolation
    for fn in (sync_equity_history, sync_closed_trades):
        try:
            fn()
        except Exception as e:
            print(f"[perf] {fn.__name__} failed: {e}")
