"""enforcing hard trading limits in code that no llm output can override"""
import os

from engine.memory import get_client, get_active_thesis

MAX_POSITION_FRACTION = float(os.environ.get("MAX_POSITION_FRACTION", "0.10"))  # single-position cap
# a high-conviction trade may exceed the soft cap; this is the confidence
# (of the winning side) required to do so, and the hard ceiling it still obeys.
CAP_OVERRIDE_CONFIDENCE = float(
    os.environ.get("CAP_OVERRIDE_CONFIDENCE", "0.70"))

# minimum reward-to-risk ratio required to act. the transcript's core rule:
# "if I'm wrong what do I lose, if I'm right what do I gain — is the math in my
# favor?" risk = distance to the chandelier stop; reward = an ATR-multiple
# target. trades whose upside does not justify their downside are rejected even
# when judges are confident. env-tunable; set to 0 to disable.
MIN_REWARD_RISK = float(os.environ.get("MIN_REWARD_RISK", "1.0"))
TARGET_ATR_MULT = float(os.environ.get("TARGET_ATR_MULT", "3.0"))


def _reward_risk(ticker, decision):
    # returns (rr_ratio, note) using the SAME atr + chandelier logic as
    # pipeline.ta_structure, computed from recent daily bars. fails OPEN
    # (returns None) if data is unavailable, so it never blocks on a fetch error.
    try:
        import pandas as pd
        from pipeline.ta_structure import _atr, chandelier_exit
        try:
            from engine.yf_session import yf_download
            from core.config import EXCHANGE_SUFFIX
            sym = (ticker if not EXCHANGE_SUFFIX or ticker.endswith(EXCHANGE_SUFFIX)
                   else ticker + EXCHANGE_SUFFIX)
        except Exception:
            import yfinance as yf
            yf_download = lambda s, **k: yf.download(s, **k)
            sym = ticker.replace(".", "-")
        raw = yf_download(sym, period="60d", auto_adjust=True, progress=False)
        if raw is None or len(raw) < 25:
            return None, "rr: insufficient data (skipped)"
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.rename(columns=str.lower)[["high", "low", "close"]].dropna()
        if len(df) < 25:
            return None, "rr: insufficient data (skipped)"
        atr = float(_atr(df).iloc[-1])
        price = float(df["close"].iloc[-1])
        if atr <= 0 or price <= 0:
            return None, "rr: bad atr/price (skipped)"
        ch = chandelier_exit(df)
        if decision == "BUY":
            stop = float(ch["chandelier_long"].iloc[-1])
            risk = price - stop
            reward = TARGET_ATR_MULT * atr          # target = N*ATR above entry
        elif decision == "SELL":
            stop = float(ch["chandelier_short"].iloc[-1])
            risk = stop - price
            reward = TARGET_ATR_MULT * atr          # N*ATR below entry
        else:
            return None, "rr: n/a"
        if risk <= 0:
            return None, "rr: non-positive risk (skipped)"
        rr = reward / risk
        return rr, f"rr={rr:.2f} (reward {reward:.2f} / risk {risk:.2f})"
    except Exception as e:
        return None, f"rr: unavailable ({e})"


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
MIN_JUDGE_QUORUM = int(os.environ.get("MIN_JUDGE_QUORUM", "2"))  # refusing action on a single judge's vote


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

    # blocking trades whose reward does not justify the risk (asymmetry rule)
    if decision != "NO_TRADE" and MIN_REWARD_RISK > 0:
        rr, rr_note = _reward_risk(ticker, decision)
        if rr is not None and rr < MIN_REWARD_RISK:
            return "NO_TRADE", (f"gate: reward-to-risk {rr:.2f} below "
                                f"{MIN_REWARD_RISK} ({rr_note})")

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