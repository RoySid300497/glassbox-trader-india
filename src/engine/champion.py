"""electing the live signal model from scored tournament results"""

from engine.memory import get_client

DEFAULT_CHAMPION = "cnn1d"
MIN_SCORED = 20          # requiring enough evidence before any election
SWITCH_MARGIN = 0.05     # demanding a clear win before dethroning the champion


def get_champion():
    # reading the currently elected signal model
    res = get_client().table("config").select("value") \
        .eq("key", "signal_champion").limit(1).execute().data
    return res[0]["value"] if res else DEFAULT_CHAMPION


def set_champion(name):
    # recording a new elected champion
    get_client().table("config").upsert(
        {"key": "signal_champion", "value": str(name)[:40]}).execute()


def _electable():
    # limiting the election to models the screener and packet can run
    names = {"random_forest"}
    try:
        from inference.predictors import load_seq_predictor
        seq = load_seq_predictor()
        names.add(seq["meta"].get("kind", "cnn1d") if seq else "cnn1d")
    except Exception:
        names.add("cnn1d")
    return names


def elect_champion():
    # promoting whichever routable model clearly won recent predictions
    rows = get_client().table("model_predictions") \
        .select("model,was_correct").not_.is_("scored_at", "null") \
        .order("pred_date", desc=True).limit(400).execute().data or []
    stats = {}
    for r in rows:
        s = stats.setdefault(r["model"], [0, 0])
        s[1] += 1
        s[0] += 1 if r["was_correct"] else 0
    routable = _electable()
    rates = {m: (c / n, n) for m, (c, n) in stats.items()
             if n >= MIN_SCORED and m in routable}
    shadow_only = {m: f"{c / n:.0%}/{n}" for m, (c, n) in stats.items()
                   if n >= MIN_SCORED and m not in routable}
    if shadow_only:
        print(f"election: shadow-only standings (promoted via retrain "
              f"gate, not election): {shadow_only}")
    if not rates:
        print("election: not enough scored predictions yet — "
              f"champion stays {get_champion()}")
        return get_champion()

    current = get_champion()
    best = max(rates, key=lambda m: rates[m][0])
    current_rate = rates.get(current, (0.0, 0))[0]

    # switching only on a clear margin so the champion cannot flip-flop
    if best != current and rates[best][0] >= current_rate + SWITCH_MARGIN:
        set_champion(best)
        print(f"election: champion switched {current} -> {best} "
              f"({rates[best][0]:.0%} vs {current_rate:.0%})")
        return best

    standings = {m: f"{r:.0%} on {n}" for m, (r, n) in rates.items()}
    print(f"election: champion stays {current} — standings {standings}")
    return current
