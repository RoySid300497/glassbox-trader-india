"""syncing paper equity history and closed trades for the performance page

simulation-native: the paper book lives in supabase (written by the
execution simulator), so this reads from there rather than calling a broker
REST API. the old alpaca _get() endpoints do not exist in simulation mode.
"""

from datetime import datetime, timezone
from engine.execution import enabled
from engine.memory import get_client


def sync_equity_history():
    # recording today's simulated equity point into the equity curve so the
    # performance page can plot it over time
    if not enabled():
        return
    try:
        client = get_client()
        rows = client.table("config").select("value").eq(
            "key", "sim_equity").execute().data
        if not rows:
            print("[perf] no sim equity yet")
            return
        equity = float(rows[0]["value"])
        today = datetime.now(timezone.utc).date().isoformat()
        client.table("portfolio_history").upsert(
            {"date": today, "equity": equity}).execute()
        print(f"[perf] equity history point saved: {today} = {equity:,.2f}")
    except Exception as e:
        print(f"[perf] equity history sync failed: {e}")


def sync_closed_trades():
    # closed round-trips are written directly by the simulator when it books
    # a fill, so in simulation there is nothing extra to reconstruct here.
    # kept for interface compatibility with the weekly review caller.
    if not enabled():
        return
    print("[perf] closed trades tracked by simulator; nothing to reconstruct")


def sync_performance():
    # running both syncs with independent failure isolation
    for fn in (sync_equity_history, sync_closed_trades):
        try:
            fn()
        except Exception as e:
            print(f"[perf] {fn.__name__} failed: {e}")
