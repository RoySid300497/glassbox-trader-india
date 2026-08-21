"""closing the evidence loop CAREFULLY: measured concept edges become
bounded numeric weights and hysteresis-armed gates — but nothing governs a
live trade until the system's own shadow record proves the scheme helps,
and everything auto-reverts the moment it stops helping.

three layers, in trust order:
  1. WEIGHTS  — per-concept multipliers from multi-week scored evidence,
                shrunk hard toward 1.0 and clamped to [FLOOR, CAP] so a few
                lucky weeks can never dominate a debate. always shown to
                judges as guidance.
  2. GATES    — concepts persistently BELOW baseline (every one of the last
                ARM_WEEKS weeks) become "armed". a directional call whose
                winning side leans on armed concepts with no reliable
                support is a candidate for blocking. arming has hysteresis:
                a concept disarms only after recovering, so the gate cannot
                flap week to week.
  3. GOVERNANCE — the gate starts in SHADOW mode: every would-block is
                logged, nothing is blocked. weekly, hypothetical outcomes
                (from outcome_label, which is recorded whether or not a
                trade happened) are compared: only when would-blocked calls
                measurably underperform allowed ones over a real sample is
                the gate promoted to LIVE. if that margin ever disappears,
                it demotes itself back to shadow — auto-correction includes
                auto-reversion.

all thresholds are env-tunable; all transitions are logged to config with
their evidence, so every adaptation is visible and reversible.
"""

import os
import json
from datetime import date, datetime, timedelta, timezone

from engine.memory import get_client

# --- weights (layer 1) ---
WEIGHT_FLOOR = float(os.environ.get("EVW_FLOOR", "0.70"))
WEIGHT_CAP = float(os.environ.get("EVW_CAP", "1.30"))
WEIGHT_GAIN = float(os.environ.get("EVW_GAIN", "2.0"))     # weight = 1 + gain*edge
WEIGHT_SHRINK = int(os.environ.get("EVW_SHRINK", "40"))    # prior obs toward baseline
WEIGHT_MIN_CITED = int(os.environ.get("EVW_MIN_CITED", "20"))
WEIGHT_WEEKS = int(os.environ.get("EVW_WEEKS", "8"))       # aggregate this many weeks

# --- gates (layer 2) ---
ARM_WEEKS = int(os.environ.get("EVG_ARM_WEEKS", "3"))      # consecutive bad weeks to arm
ARM_EDGE = float(os.environ.get("EVG_ARM_EDGE", "-0.05"))  # weekly shrunk edge below this
DISARM_WEEKS = int(os.environ.get("EVG_DISARM_WEEKS", "2"))
DISARM_EDGE = float(os.environ.get("EVG_DISARM_EDGE", "-0.02"))
ARM_MIN_CITED = int(os.environ.get("EVG_MIN_CITED", "5"))  # per week, else week ignored
POS_SUPPORT = float(os.environ.get("EVG_POS_SUPPORT", "1.08"))
GATE_OVERRIDE_CONF = float(os.environ.get("EVG_OVERRIDE_CONF", "0.75"))

# --- governance (layer 3) ---
GOV_MIN_N = int(os.environ.get("EVGOV_MIN_N", "10"))       # per branch (blocked/allowed)
GOV_PROMOTE_MARGIN = float(os.environ.get("EVGOV_PROMOTE", "0.10"))
GOV_DEMOTE_MARGIN = float(os.environ.get("EVGOV_DEMOTE", "0.0"))
GOV_WINDOW_DAYS = int(os.environ.get("EVGOV_WINDOW_DAYS", "90"))

MODE_KEY = "adaptive_evidence_mode"          # "shadow" (default) | "live"
ARMED_KEY = "adaptive_armed_concepts"        # json list
WEIGHTS_KEY = "adaptive_evidence_weights"    # json snapshot for audit


# ---------------------------------------------------------------- helpers

def _cfg_get(key, default=None):
    try:
        rows = get_client().table("config").select("value") \
            .eq("key", key).limit(1).execute().data
        return rows[0]["value"] if rows else default
    except Exception:
        return default


def _cfg_set(key, value):
    try:
        get_client().table("config").upsert(
            {"key": key, "value": value}).execute()
    except Exception as e:
        print(f"  [adaptive] config write failed for {key}: {e}")


def get_mode():
    # shadow is the safe default: measure everything, block nothing
    mode = _cfg_get(MODE_KEY, "shadow")
    return mode if mode in ("shadow", "live") else "shadow"


def _weekly_stats(weeks=None):
    # evidence_stats rows grouped by week, newest first
    rows = get_client().table("evidence_stats") \
        .select("week_ending,concept,cited,hits,hit_rate,baseline") \
        .order("week_ending", desc=True).limit(1000).execute().data or []
    by_week = {}
    for r in rows:
        by_week.setdefault(r["week_ending"], []).append(r)
    ordered = sorted(by_week, reverse=True)
    if weeks:
        ordered = ordered[:weeks]
    return [(w, by_week[w]) for w in ordered]


def _shrunk_edge(hits, n, baseline, shrink):
    # edge pulled toward zero: thin samples cannot move it far
    return (hits + baseline * shrink) / (n + shrink) - baseline


# ---------------------------------------------------------------- layer 1

def concept_weights():
    # bounded multipliers from the multi-week aggregated scored record.
    # returns {concept: weight}; concepts without enough evidence stay 1.0
    # implicitly (absent from the dict).
    try:
        weekly = _weekly_stats(weeks=WEIGHT_WEEKS)
    except Exception as e:
        print(f"  [adaptive] weights unavailable: {e}")
        return {}
    agg = {}
    for _, rows in weekly:
        for r in rows:
            a = agg.setdefault(r["concept"], [0, 0, 0.0, 0])
            hits = r.get("hits")
            if hits is None:                       # older snapshots lack hits
                hits = round(float(r["hit_rate"]) * int(r["cited"]))
            a[0] += int(hits)
            a[1] += int(r["cited"])
            a[2] += float(r["baseline"]) * int(r["cited"])
            a[3] += int(r["cited"])
    out = {}
    for concept, (hits, n, base_w, base_n) in agg.items():
        if n < WEIGHT_MIN_CITED or base_n == 0:
            continue
        baseline = base_w / base_n
        edge = _shrunk_edge(hits, n, baseline, WEIGHT_SHRINK)
        w = 1.0 + WEIGHT_GAIN * edge
        out[concept] = round(min(WEIGHT_CAP, max(WEIGHT_FLOOR, w)), 3)
    return out


def judge_guidance_block():
    # the numeric guidance judges receive inside every packet. shown in both
    # modes — weights are bounded advice, not a trade decision, so they are
    # safe to surface immediately; only the GATE waits for governance.
    weights = concept_weights()
    armed = get_armed()
    if not weights and not armed:
        return None
    block = {
        "how_to_use": ("each evidence concept below carries a numeric weight "
                       "from this system's own multi-week scored record "
                       "(1.0 = neutral, capped at "
                       f"{WEIGHT_FLOOR}-{WEIGHT_CAP}). scale your trust in a "
                       "claim by its concept's weight. concepts listed in "
                       "'persistently_failing' have been below baseline for "
                       "weeks — do not let them carry a directional vote "
                       "on their own."),
        "weights": dict(sorted(weights.items(), key=lambda kv: -kv[1])),
    }
    if armed:
        block["persistently_failing"] = sorted(armed)
    return block


# ---------------------------------------------------------------- layer 2

def get_armed():
    raw = _cfg_get(ARMED_KEY, "[]")
    try:
        return set(json.loads(raw))
    except Exception:
        return set()


def _set_armed(armed):
    _cfg_set(ARMED_KEY, json.dumps(sorted(armed)))


def update_armed():
    # hysteresis: arm after ARM_WEEKS consecutive qualifying bad weeks;
    # disarm after DISARM_WEEKS consecutive recovered weeks. weeks with too
    # few citations are ignored (no evidence either way).
    try:
        weekly = _weekly_stats(weeks=max(ARM_WEEKS, DISARM_WEEKS) + 2)
    except Exception as e:
        print(f"  [adaptive] arming skipped: {e}")
        return get_armed()
    # per concept, newest-first list of weekly shrunk edges (qualified weeks)
    series = {}
    for _, rows in weekly:                      # weekly is newest-first
        for r in rows:
            if int(r["cited"]) < ARM_MIN_CITED:
                continue
            hits = r.get("hits")
            if hits is None:
                hits = round(float(r["hit_rate"]) * int(r["cited"]))
            edge = _shrunk_edge(int(hits), int(r["cited"]),
                                float(r["baseline"]), SHRINK_WEEKLY)
            series.setdefault(r["concept"], []).append(edge)

    armed = get_armed()
    changed = []
    for concept, edges in series.items():
        recent_bad = edges[:ARM_WEEKS]
        recent_ok = edges[:DISARM_WEEKS]
        if concept not in armed:
            if len(recent_bad) >= ARM_WEEKS and \
                    all(e < ARM_EDGE for e in recent_bad):
                armed.add(concept)
                changed.append(f"+{concept}")
        else:
            if len(recent_ok) >= DISARM_WEEKS and \
                    all(e > DISARM_EDGE for e in recent_ok):
                armed.discard(concept)
                changed.append(f"-{concept}")
    if changed:
        _set_armed(armed)
        print(f"  [adaptive] armed set changed: {' '.join(changed)} "
              f"-> {sorted(armed) or 'none'}")
    return armed


SHRINK_WEEKLY = int(os.environ.get("EVG_SHRINK_WEEKLY", "10"))


def _winning_concepts(verdict, decision):
    # the concepts the winning side actually cited, via the report's bucketer
    from engine.evidence_report import _cited_buckets
    side = "bull_case" if decision == "BUY" else "bear_case"
    return _cited_buckets(verdict.get(side) or {}, side)


def evaluate_decision(ticker, verdict, decision, win_conf):
    # the gate judgment for one directional call. ALWAYS logs the shadow
    # record; returns (block, note) where block is True only in live mode.
    if decision not in ("BUY", "SELL"):
        return False, None
    armed = get_armed()
    weights = concept_weights()
    cited = _winning_concepts(verdict, decision)
    hit_armed = sorted(armed & cited)
    has_support = any(weights.get(c, 1.0) >= POS_SUPPORT for c in cited)
    would_block = bool(hit_armed) and not has_support \
        and win_conf < GATE_OVERRIDE_CONF
    mode = get_mode()
    # shadow record for governance, in both modes
    try:
        get_client().table("adaptive_evidence_log").insert({
            "log_date": str(date.today()), "ticker": ticker,
            "action_proposed": decision, "would_block": would_block,
            "armed_cited": hit_armed, "cited": sorted(cited),
            "win_conf": round(float(win_conf), 4), "mode": mode}).execute()
    except Exception as e:
        print(f"  [adaptive] shadow log failed: {e}")
    if would_block and mode == "live":
        return True, (f"gate: adaptive evidence — winning case leans on "
                      f"persistently failing {hit_armed} with no reliable "
                      f"support (conf {win_conf:.2f} < {GATE_OVERRIDE_CONF})")
    if would_block:
        print(f"  [adaptive] SHADOW would block {decision} {ticker}: "
              f"armed {hit_armed}, conf {win_conf:.2f}")
    return False, None


# ---------------------------------------------------------------- layer 3

def _hypothetical_correct(action, label):
    return (action == "BUY" and label == "Up") or \
           (action == "SELL" and label == "Down")


def evaluate_governance():
    # weekly: does the would-block flag actually separate bad calls from
    # good ones? uses outcome_label (recorded whether or not a trade
    # happened), so the comparison works identically in shadow and live.
    client = get_client()
    since = str(date.today() - timedelta(days=GOV_WINDOW_DAYS))
    logs = client.table("adaptive_evidence_log").select("*") \
        .gte("log_date", since).execute().data or []
    if not logs:
        print("  [adaptive] governance: no shadow records yet")
        return get_mode()
    # join to decisions on (ticker, day) for the outcome label
    decs = client.table("decisions") \
        .select("ticker,decided_at,outcome_label") \
        .gte("decided_at", since) \
        .not_.is_("outcome_label", "null").execute().data or []
    label_by = {}
    for d in decs:
        label_by[(d["ticker"], str(d["decided_at"])[:10])] = d["outcome_label"]

    branch = {True: [0, 0], False: [0, 0]}      # would_block -> [correct, n]
    for r in logs:
        label = label_by.get((r["ticker"], str(r["log_date"])[:10]))
        if label is None:
            continue
        b = branch[bool(r["would_block"])]
        b[1] += 1
        b[0] += 1 if _hypothetical_correct(r["action_proposed"], label) else 0

    (bc, bn), (ac, an) = branch[True], branch[False]
    mode = get_mode()
    if bn < GOV_MIN_N or an < GOV_MIN_N:
        print(f"  [adaptive] governance: insufficient evidence "
              f"(blocked {bn}, allowed {an}, need {GOV_MIN_N} each) — "
              f"staying {mode}")
        return mode
    blocked_hit, allowed_hit = bc / bn, ac / an
    sep = allowed_hit - blocked_hit
    print(f"  [adaptive] governance: allowed {allowed_hit:.0%} ({an}) vs "
          f"would-block {blocked_hit:.0%} ({bn}) — separation {sep:+.0%}")
    if mode == "shadow" and sep >= GOV_PROMOTE_MARGIN:
        _cfg_set(MODE_KEY, "live")
        _log_transition("shadow->live", sep, bn, an)
        print("  [adaptive] PROMOTED to live: the gate has earned governance")
        return "live"
    if mode == "live" and sep <= GOV_DEMOTE_MARGIN:
        _cfg_set(MODE_KEY, "shadow")
        _log_transition("live->shadow (auto-revert)", sep, bn, an)
        print("  [adaptive] AUTO-REVERTED to shadow: separation vanished")
        return "shadow"
    return mode


def _log_transition(what, sep, bn, an):
    _cfg_set(f"adaptive_transition_{datetime.now(timezone.utc).isoformat()}",
             json.dumps({"what": what, "separation": round(sep, 4),
                         "blocked_n": bn, "allowed_n": an}))


def weekly_update():
    # the one weekly entry point: refresh weights snapshot, update arming
    # with hysteresis, then let governance promote or revert the gate.
    weights = concept_weights()
    _cfg_set(WEIGHTS_KEY, json.dumps(weights))
    print(f"  [adaptive] weights ({len(weights)} concepts with evidence): "
          f"{dict(sorted(weights.items(), key=lambda kv: -kv[1]))}")
    armed = update_armed()
    print(f"  [adaptive] armed concepts: {sorted(armed) or 'none'}")
    mode = evaluate_governance()
    print(f"  [adaptive] gate mode: {mode}")
    return {"weights": weights, "armed": sorted(armed), "mode": mode}
