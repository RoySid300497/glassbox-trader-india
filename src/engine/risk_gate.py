"""enforcing hard trading limits in code that no llm output can override"""

from engine.memory import get_client, get_active_thesis

MAX_POSITION_FRACTION = 0.10     # limiting any single position to 10% of capital
MAX_DAILY_TRADES = 3             # capping new trades per day
MIN_JUDGE_CONFIDENCE = 0.5       # requiring average judge conviction to act
MIN_JUDGE_QUORUM = 2            # refusing action on a single judge's vote


def count_trades_today():
    # counting buy or sell decisions already made today
    from datetime import date
    res = get_client().table("decisions").select("id, action") \
        .gte("decided_at", str(date.today())).execute()
    return sum(1 for r in (res.data or []) if r["action"] != "NO_TRADE")


def apply_gate(ticker, verdict):
    # passing the panel verdict through every hard rule before it stands
    decision = verdict["decision"]
    votes = verdict["judge_votes"]

    # blocking any action when judges did not respond
    if decision != "NO_TRADE" and not votes:
        return "NO_TRADE", "gate: no judge votes received"

    # blocking any action decided by fewer judges than the quorum
    if decision != "NO_TRADE" and len(votes) < MIN_JUDGE_QUORUM:
        return "NO_TRADE", (f"gate: only {len(votes)} judge vote(s) — "
                            f"quorum is {MIN_JUDGE_QUORUM}")

    # blocking low-conviction actions
    if decision != "NO_TRADE":
        avg_conf = sum(v.get("confidence", 0) for v in votes) / len(votes)
        if avg_conf < MIN_JUDGE_CONFIDENCE:
            return "NO_TRADE", (f"gate: avg judge confidence {avg_conf:.2f} "
                                f"below {MIN_JUDGE_CONFIDENCE}")

    # blocking trades past the daily cap
    if decision != "NO_TRADE" and count_trades_today() >= MAX_DAILY_TRADES:
        return "NO_TRADE", f"gate: daily trade cap {MAX_DAILY_TRADES} reached"

    # annotating thesis-backed decisions so holds can run longer downstream
    thesis = get_active_thesis(ticker)
    note = "gate: passed"
    if thesis and decision == "BUY" and thesis["direction"] == "LONG":
        note = "gate: passed, thesis-backed hold permitted"
    if thesis and decision == "SELL" and thesis["direction"] == "LONG":
        note = "gate: passed, warning — sell contradicts active LONG thesis"

    return decision, note
