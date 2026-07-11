"""running the fixed three-round debate that terminates by construction"""

from collections import Counter
from engine.panels import (run_cases, run_rebuttal, run_judges,
                           BUY_PANEL, SELL_PANEL)


def majority_vote(votes, default="NO_TRADE"):
    # scoring the panel on a 300-point scale: each judge seat carries 100
    # points, scaled by that provider's earned importance (0.75-1.35, from
    # its own scored record) and dampened conviction (0.5 + 0.5*confidence,
    # so miscalibrated confidence can sway but never dominate). an action
    # needs two thirds of the points present AND at least two judges
    # voting that direction — one judge alone can never trade
    if not votes:
        return default
    try:
        from engine.judge_weights import provider_weights
        weights = provider_weights()
    except Exception:
        weights = {}
    total, tallies, heads = 0.0, {}, Counter()
    for v in votes:
        w = weights.get(v.get("provider"), 1.0)
        try:
            conf = min(1.0, max(0.0, float(v.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        pts = 100.0 * w * (0.5 + 0.5 * conf)
        total += 100.0 * w
        tallies[v["vote"]] = tallies.get(v["vote"], 0.0) + pts
        heads[v["vote"]] += 1
    conf_sum = {}
    for v in votes:
        try:
            c = min(1.0, max(0.0, float(v.get("confidence", 0.5))))
        except (TypeError, ValueError):
            c = 0.5
        conf_sum[v["vote"]] = conf_sum.get(v["vote"], 0.0) + c
    top = max(tallies, key=tallies.get)
    # a weighted majority of points carries the vote, but a directional
    # verdict also needs two heads and genuine conviction — the agreeing
    # judges must average at least 0.6 confidence, so confident pairs
    # trade while coin-flip votes still abstain
    if tallies[top] <= total / 2:
        return default
    if top in ("BUY", "SELL"):
        if heads[top] < 2:
            return default
        if conf_sum[top] / heads[top] < 0.6:
            return default
    return top


def decide(packet):
    # executing round 1 cases, round 2 rebuttals, round 3 votes — then stop
    bull = run_cases(packet, "bull", BUY_PANEL)
    bear = run_cases(packet, "bear", SELL_PANEL)

    bull_reb = run_rebuttal(packet, "bull", BUY_PANEL, bear)
    bear_reb = run_rebuttal(packet, "bear", SELL_PANEL, bull)

    votes = run_judges(packet, bull, bear, bull_reb, bear_reb)
    decision = majority_vote(votes)

    return {"decision": decision,
            "bull_case": {"opening": bull, "rebuttal": bull_reb},
            "bear_case": {"opening": bear, "rebuttal": bear_reb},
            "judge_votes": votes}
