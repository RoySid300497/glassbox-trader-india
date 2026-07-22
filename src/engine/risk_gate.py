"""enforcing hard trading limits in code that no llm output can override"""
import os

from engine.memory import get_client, get_active_thesis

MAX_POSITION_FRACTION = 0.10     # limiting any single position to 10% of capital
# a high-conviction trade may exceed the soft cap; this is the confidence
# (of the winning side) required to do so, and the hard ceiling it still obeys.
CAP_OVERRIDE_CONFIDENCE = float(
    os.environ.get("CAP_OVERRIDE_CONFIDENCE", "0.70"))


def _max_daily_trades():
    # flexible daily trade cap: scales with how many stocks we actually debate
    # (DEBATE_BUDGET) so it never silently blocks good signals on high-budget
    # days, with a floor of 3 and a hard env override. ~40% of debated names
    # may become trades — enough to act on real signals, low enough to prevent
    # runaway over-trading.
    import math
    override = os.environ.get("MAX_DAILY_TRADES")
    if override and override.strip():
        return int(override)
    budget = int(os.environ.get("DEBATE_BUDGET", "10"))
    return max(3, math.ceil(0.4 * budget))


def _hard_daily_ceiling():
    # even high-conviction trades cannot exceed this absolute ceiling, so a day
    # of uniformly confident signals still can't cause runaway trading. set to
    # 1.5x the soft cap (env-overridable).
    override = os.environ.get("MAX_DAILY_TRADES_HARD")
    if override and override.strip():
        return int(override)
    import math
    return math.ceil(1.5 * _max_daily_trades())


def _winning_side_confidence(decision, votes):
    # conviction of the judges who actually voted for the winning direction,
    # not diluted by dissenters — the right signal for "is this exceptional?"
    agree = [v.get("confidence", 0) for v in votes
             if v.get("vote") == decision]
    return (sum(agree) / len(agree)) if agree else 0.0
MIN_JUDGE_CONFIDENCE = float(os.environ.get("MIN_JUDGE_CONFIDENCE", "0.40"))  # avg judge conviction to act (env-tunable)
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
    # daily cap with a high-conviction override: once the soft cap is reached,
    # only exceptional signals (strong agreement among the winning judges) may
    # still trade, and only up to a hard ceiling.
    if decision != "NO_TRADE":
        traded = count_trades_today()
        cap = _max_daily_trades()
        if traded >= cap:
            win_conf = _winning_side_confidence(decision, votes)
            ceiling = _hard_daily_ceiling()
            if win_conf >= CAP_OVERRIDE_CONFIDENCE and traded < ceiling:
                note = (f"gate: soft cap {cap} exceeded on high conviction "
                        f"{win_conf:.2f} (ceiling {ceiling})")
                # fall through to thesis annotation below, preserving this note
                thesis = get_active_thesis(ticker)
                if thesis and decision == "BUY" and thesis["direction"] == "LONG":
                    note += "; thesis-backed hold permitted"
                if thesis and decision == "SELL" and thesis["direction"] == "LONG":
                    note += "; warning — sell contradicts active LONG thesis"
                return decision, note
            reason = (f"gate: daily trade cap {cap} reached"
                      if win_conf < CAP_OVERRIDE_CONFIDENCE
                      else f"gate: hard daily ceiling {ceiling} reached")
            return "NO_TRADE", reason

    # annotating thesis-backed decisions so holds can run longer downstream
    thesis = get_active_thesis(ticker)
    note = "gate: passed"
    if thesis and decision == "BUY" and thesis["direction"] == "LONG":
        note = "gate: passed, thesis-backed hold permitted"
    if thesis and decision == "SELL" and thesis["direction"] == "LONG":
        note = "gate: passed, warning — sell contradicts active LONG thesis"

    return decision, note