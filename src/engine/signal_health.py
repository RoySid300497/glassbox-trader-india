"""monitoring signal-1 (the model layer) health: regime drift, calibration,
and cross-model agreement — and translating that into a performance-based risk
multiplier the execution layer can apply.

This module is READ-ONLY over model_predictions/decisions (it never trades). It
answers three honest questions:
  1. Is the live signal decaying toward random? (drift)
  2. Does stated confidence mean anything? (calibration)
  3. Do the models agree right now? (cross-model agreement)

It cannot make the signal more accurate — accuracy is capped by how much
predictable signal the market contains. What it does is PROTECT capital when the
signal weakens, and surface whether confidence is real, so risk is only taken
when the signal is behaving.
"""

import os
from collections import defaultdict
from engine.memory import get_client

# --- config (all env-tunable) ---
DRIFT_WINDOW = int(os.environ.get("DRIFT_WINDOW", "150"))          # recent scored preds to judge on
DRIFT_MIN_N = int(os.environ.get("DRIFT_MIN_N", "40"))            # need this many before judging
RANDOM_BASELINE = float(os.environ.get("RANDOM_BASELINE", "0.3333"))
# how far above random the champion must be to be "healthy"; below this we derisk
DRIFT_HEALTHY_EDGE = float(os.environ.get("DRIFT_HEALTHY_EDGE", "0.05"))  # +5pp over random
# risk multipliers applied by execution when signal is degraded (never > 1.0)
DERISK_FLOOR = float(os.environ.get("DERISK_FLOOR", "0.25"))      # never cut below 25% risk


def _champion():
    try:
        from engine.champion import get_champion
        return get_champion()
    except Exception:
        return "cnn1d"


def _recent_scored(model=None, window=DRIFT_WINDOW):
    q = get_client().table("model_predictions") \
        .select("model,was_correct,confidence,outcome_label,direction") \
        .not_.is_("scored_at", "null") \
        .order("pred_date", desc=True).limit(int(window) * 12)
    rows = q.execute().data or []
    if model:
        rows = [r for r in rows if r.get("model") == model]
    return rows[: int(window)] if not model else rows[: int(window)]


def drift_status():
    # is the CHAMPION's recent live hit rate healthily above random?
    champ = _champion()
    rows = _recent_scored(model=champ)
    n = len(rows)
    if n < DRIFT_MIN_N:
        return {"champion": champ, "n": n, "hit_rate": None,
                "edge": None, "state": "insufficient_data",
                "risk_multiplier": 1.0}
    hits = sum(1 for r in rows if r.get("was_correct"))
    hit_rate = hits / n
    edge = hit_rate - RANDOM_BASELINE
    # translate edge into a risk multiplier: full risk at healthy edge, scaling
    # down linearly to the floor as edge -> 0, and floor if at/below random.
    if edge >= DRIFT_HEALTHY_EDGE:
        mult = 1.0
        state = "healthy"
    elif edge <= 0:
        mult = DERISK_FLOOR
        state = "at_or_below_random"
    else:
        # linear between floor (edge 0) and 1.0 (edge = healthy)
        frac = edge / DRIFT_HEALTHY_EDGE
        mult = round(DERISK_FLOOR + (1.0 - DERISK_FLOOR) * frac, 3)
        state = "degrading"
    return {"champion": champ, "n": n, "hit_rate": round(hit_rate, 4),
            "edge": round(edge, 4), "state": state,
            "risk_multiplier": mult}


def calibration_report(buckets=((0.0, 0.5), (0.5, 0.6), (0.6, 0.7),
                                (0.7, 0.85), (0.85, 1.01))):
    # does higher stated confidence actually mean higher accuracy? if not, the
    # confidence signal is noise and thresholds on it are meaningless.
    champ = _champion()
    rows = _recent_scored(model=champ, window=DRIFT_WINDOW * 2)
    out = []
    for lo, hi in buckets:
        b = [r for r in rows
             if r.get("confidence") is not None and lo <= r["confidence"] < hi]
        if not b:
            out.append({"bucket": f"{lo:.2f}-{hi:.2f}", "n": 0,
                        "hit_rate": None})
            continue
        hr = sum(1 for r in b if r.get("was_correct")) / len(b)
        out.append({"bucket": f"{lo:.2f}-{hi:.2f}", "n": len(b),
                    "hit_rate": round(hr, 3)})
    # monotonic check: is accuracy rising with confidence?
    hrs = [x["hit_rate"] for x in out if x["hit_rate"] is not None]
    monotone = all(a <= b for a, b in zip(hrs, hrs[1:])) if len(hrs) >= 2 else None
    return {"champion": champ, "buckets": out, "monotonic": monotone}


def agreement_multiplier(ticker):
    # how much do TODAY's models agree on this ticker? high agreement = more
    # robust signal; disagreement = noisier, so trade smaller. returns 1.0 for
    # strong consensus down to a floor for a split.
    from datetime import date
    rows = get_client().table("model_predictions") \
        .select("model,direction") \
        .eq("ticker", ticker).eq("pred_date", str(date.today())) \
        .execute().data or []
    # exclude the constant baselines and ensemble from the agreement vote
    votes = [r["direction"] for r in rows
             if not r["model"].startswith("always_")
             and r["model"] != "ensemble"]
    if len(votes) < 3:
        return 1.0, None            # not enough to judge; don't penalize
    top = max(set(votes), key=votes.count)
    frac = votes.count(top) / len(votes)
    # map agreement fraction (0.33 split .. 1.0 unanimous) to [floor..1.0]
    floor = float(os.environ.get("AGREEMENT_FLOOR", "0.5"))
    mult = round(floor + (1.0 - floor) * max(0.0, (frac - 0.5) / 0.5), 3)
    mult = min(1.0, max(floor, mult))
    return mult, {"agree_frac": round(frac, 2), "n": len(votes), "top": top}


def signal_risk_multiplier(ticker=None):
    # the single number execution multiplies RISK_PER_TRADE by. combines drift
    # (portfolio-wide signal health) with per-ticker model agreement. always in
    # [DERISK_FLOOR, 1.0]; defaults to 1.0 (no change) when data is thin, so it
    # never silently amplifies risk — it can only hold or reduce it.
    d = drift_status()
    mult = d["risk_multiplier"]
    detail = {"drift": d}
    if ticker:
        am, ainfo = agreement_multiplier(ticker)
        mult = round(mult * am, 3)
        detail["agreement"] = ainfo
    mult = min(1.0, max(DERISK_FLOOR, mult))
    detail["risk_multiplier"] = mult
    return mult, detail


def health_report():
    # one-line-per-item health summary for the weekly review / logs
    d = drift_status()
    print("signal health — drift:")
    if d["state"] == "insufficient_data":
        print(f"  champion {d['champion']}: only {d['n']} scored — "
              f"need {DRIFT_MIN_N} to judge drift")
    else:
        print(f"  champion {d['champion']}: hit {d['hit_rate']:.0%} on "
              f"{d['n']} (edge {d['edge']:+.1%} vs random) — {d['state']}; "
              f"risk x{d['risk_multiplier']}")
    cal = calibration_report()
    print("signal health — calibration (does confidence mean accuracy?):")
    for b in cal["buckets"]:
        if b["hit_rate"] is not None:
            print(f"  conf {b['bucket']}: {b['hit_rate']:.0%} on {b['n']}")
    if cal["monotonic"] is False:
        print("  WARNING: accuracy does NOT rise with confidence — "
              "confidence may be uninformative")
    return {"drift": d, "calibration": cal}
