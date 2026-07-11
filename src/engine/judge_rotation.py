"""weighted-exploration rotation for judge seats

the three judge seats have always gone to the same primaries when healthy,
so only those few accumulate an accuracy record. this rotates seats with
epsilon-greedy exploration — usually seating the proven-best providers, but
sometimes an under-tested one — so every provider builds a record and a
genuinely better dark horse can earn its way in. exploration is quota-aware
(never seats a benched provider) and floored (no provider stays untested)
"""

import os
import random
from datetime import date

EXPLORE_RATE = float(os.environ.get("JUDGE_EXPLORE_RATE") or "0.25")
MIN_VOTES_TRUSTED = 10          # below this a provider is "under-tested"
SEED_DAILY = True              # deterministic within a day, varies across days
_table_cache = None            # accuracy table fetched once per process


def _accuracy_table():
    # per-provider scored directional accuracy, fetched once and reused
    global _table_cache
    if _table_cache is not None:
        return _table_cache
    try:
        from engine.judge_weights import provider_weights
        # provider_weights returns importance multipliers; we want raw counts
        import json
        from engine.memory import get_client
        rows = get_client().table("decisions") \
            .select("judge_votes,outcome_label") \
            .not_.is_("outcome_label", "null") \
            .order("id", desc=True).limit(500).execute().data or []
        tally = {}
        for r in rows:
            votes = r["judge_votes"]
            if isinstance(votes, str):
                try:
                    votes = json.loads(votes)
                except Exception:
                    continue
            out = r["outcome_label"]
            for v in votes or []:
                d = str(v.get("vote", "")).upper()
                if d not in ("BUY", "SELL"):
                    continue
                p = v.get("provider")
                if not p:
                    continue
                hit = (d == "BUY" and out == "Up") or \
                      (d == "SELL" and out == "Down")
                s = tally.setdefault(p, [0, 0])
                s[1] += 1
                s[0] += 1 if hit else 0
        _table_cache = tally
        return tally
    except Exception as e:
        print(f"  [rotation] accuracy table unavailable ({e})")
        _table_cache = {}
        return {}


def choose_panel(role, default_panel, pool, n_seats):
    # returning an ordered provider preference list for a role's seats,
    # blending exploitation (proven accuracy) with exploration (under-tested)
    if SEED_DAILY:
        random.seed(f"{date.today()}-{role}")
    tally = _accuracy_table()

    def acc(p):
        c, n = tally.get(p, [0, 0])
        return (c / n) if n >= MIN_VOTES_TRUSTED else None

    tested = [p for p in pool if acc(p) is not None]
    untested = [p for p in pool if acc(p) is None]

    # exploit order: proven providers by descending accuracy, then defaults
    exploit = sorted(tested, key=lambda p: -acc(p))
    # keep the configured defaults as a stable backbone when nothing is proven
    backbone = [p for p in default_panel if p in pool]
    exploit = list(dict.fromkeys(exploit + backbone +
                                 [p for p in pool if p not in exploit]))

    chosen = []
    for _ in range(min(n_seats, len(pool))):
        remaining_untested = [p for p in untested if p not in chosen]
        # explore an under-tested provider with EXPLORE_RATE probability,
        # but only if any remain — otherwise exploit the best available
        if remaining_untested and random.random() < EXPLORE_RATE:
            pick = random.choice(remaining_untested)
        else:
            pick = next((p for p in exploit if p not in chosen), None)
            if pick is None:
                pick = next((p for p in pool if p not in chosen), None)
        if pick:
            chosen.append(pick)

    # the seat-filler still falls through the full healthy pool if a chosen
    # provider is quota-dead, so exploration never costs a missed seat
    if chosen != list(default_panel):
        print(f"  [rotation] {role} seats -> {chosen} "
              f"(explore={EXPLORE_RATE})")
    return chosen
