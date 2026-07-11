"""deciding from live performance when a retrain should dispatch itself"""

import os
import requests
from datetime import datetime, timezone
from engine.memory import get_client

MIN_SCORED = 40            # requiring enough champion evidence before judging
DECAY_THRESHOLD = 0.36     # flagging hit rates decaying toward the 1/3 baseline
DOMINANCE_MARGIN = 0.07    # flagging any shadow clearly beating the champion
DOMINANCE_MIN_N = 30       # requiring enough shadow evidence for dominance
COOLDOWN_DAYS = 14         # spacing retrains so each model gets judged fairly
WINDOW = 200               # judging over the most recent scored predictions
PEER_MIN_N = 20            # requiring peer evidence for the anomaly guard


def _last_trigger_days():
    # measuring days since the last automated retrain dispatch
    rows = get_client().table("config").select("value") \
        .eq("key", "last_retrain_at").limit(1).execute().data
    if not rows:
        return 10 ** 6
    then = datetime.fromisoformat(rows[0]["value"])
    return (datetime.now(timezone.utc) - then).days


def _stats():
    # aggregating recent scored hit rates per model alongside the champion
    from engine.champion import get_champion
    rows = get_client().table("model_predictions") \
        .select("model,was_correct").not_.is_("scored_at", "null") \
        .order("pred_date", desc=True).limit(WINDOW).execute().data or []
    agg = {}
    for r in rows:
        s = agg.setdefault(r["model"], [0, 0])
        s[1] += 1
        s[0] += 1 if r["was_correct"] else 0
    return get_champion(), {m: (c / n, n) for m, (c, n) in agg.items()}


def retrain_signal():
    # returning a human-readable reason when performance demands a retrain
    champion, rates = _stats()
    champ_rate, champ_n = rates.get(champion, (None, 0))
    if champ_n < MIN_SCORED:
        return None
    if champ_rate < DECAY_THRESHOLD:
        # holding when the whole roster decayed together — that is the
        # market breaking, not the model, and retraining would chase it
        peers = [r for m, (r, n) in rates.items()
                 if m not in (champion, "ensemble") and n >= PEER_MIN_N]
        if peers and max(peers) < DECAY_THRESHOLD:
            print(f"retrain trigger: whole roster decayed (best peer "
                  f"{max(peers):.0%}) — market anomaly, holding")
            return None
        return (f"champion {champion} hit rate {champ_rate:.0%} over "
                f"{champ_n} scored — decaying toward the random baseline")
    for m, (r, n) in rates.items():
        if m in (champion, "ensemble"):
            continue
        if n >= DOMINANCE_MIN_N and r >= champ_rate + DOMINANCE_MARGIN:
            return (f"shadow {m} at {r:.0%} over {n} beats champion "
                    f"{champion} at {champ_rate:.0%} by the margin")
    return None


def _dispatch_retrain():
    # firing the retrain workflow through the github api from inside actions
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        return False
    r = requests.post(
        f"https://api.github.com/repos/{repo}/actions/workflows/"
        f"retrain.yml/dispatches",
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json"},
        json={"ref": "main"}, timeout=20)
    return r.status_code == 204


def maybe_trigger_retrain():
    # dispatching a retrain when the signals fire outside the cooldown
    try:
        if _last_trigger_days() < COOLDOWN_DAYS:
            print("retrain trigger: inside cooldown — skipping")
            return
        reason = retrain_signal()
        if not reason:
            print("retrain trigger: performance healthy — no retrain")
            return
        if _dispatch_retrain():
            get_client().table("config").upsert(
                {"key": "last_retrain_at",
                 "value": datetime.now(timezone.utc).isoformat()}).execute()
            print(f"retrain trigger: DISPATCHED — {reason}")
        else:
            print(f"retrain trigger: wanted a retrain ({reason}) but the "
                  f"dispatch failed — run the retrain workflow manually")
    except Exception as e:
        print(f"retrain trigger failed: {e}")
