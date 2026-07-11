"""weighting judge votes by each provider's own scored directional record

all judges start equal; weights drift from 1.0 only as a provider banks
scored directional votes, are shrunk hard toward equal, and are capped so
no single judge can ever outvote the other two combined
"""

from engine.memory import get_client

MIN_VOTES = 10               # equal weight until a provider banks this many
SHRINK = 20                  # prior observations pulling weight toward equal
WEIGHT_FLOOR = 0.75
WEIGHT_CAP = 1.35            # cap < 2 * floor: two judges always beat one
WINDOW = 500

_cache = None


def provider_weights():
    # computing shrunk accuracy-relative weights, once per run
    global _cache
    if _cache is not None:
        return _cache
    try:
        import json
        rows = get_client().table("decisions") \
            .select("judge_votes,outcome_label") \
            .not_.is_("outcome_label", "null") \
            .order("id", desc=True).limit(WINDOW).execute().data or []
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

        graded = {p: s for p, s in tally.items() if s[1] >= MIN_VOTES}
        if not graded:
            _cache = {}
            return _cache
        pool_h = sum(s[0] for s in graded.values())
        pool_n = sum(s[1] for s in graded.values())
        base = pool_h / pool_n if pool_n else 0.0
        weights = {}
        if base > 0:
            for p, (h, n) in graded.items():
                shrunk = (h + base * SHRINK) / (n + SHRINK)
                w = max(WEIGHT_FLOOR, min(WEIGHT_CAP, shrunk / base))
                weights[p] = round(w, 3)
        if weights:
            print(f"  [judges] accuracy weights in effect: {weights}")
        _cache = weights
    except Exception as e:
        print(f"  [judges] weight computation failed ({e}) — equal weights")
        _cache = {}
    return _cache
