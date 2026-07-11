"""distilling systematic patterns from scored mistakes into reusable lessons"""

import json
from engine.llm_clients import ask, parse_json_reply
from engine.memory import get_client, get_active_lessons

MIN_MISTAKES = 5          # requiring enough errors before seeking patterns
MAX_ACTIVE_LESSONS = 10   # keeping the prompt injection bounded
LESSON_SCHEMA = ["lessons"]


def _wrong_cases(limit=40):
    # collecting recent scored mistakes with their reasoning and context
    rows = get_client().table("decisions") \
        .select("id,ticker,decided_at,action,cnn_direction,cnn_confidence,"
                "judge_votes,outcome_label,outcome_return_1d") \
        .eq("was_correct", False).not_.is_("scored_at", "null") \
        .order("decided_at", desc=True).limit(int(limit)).execute().data or []
    cases = []
    for r in rows:
        votes = r.get("judge_votes") or []
        reason = votes[0].get("reason", "")[:200] if votes else ""
        news = get_client().table("news_archive") \
            .select("headline,sentiment").eq("ticker", r["ticker"]) \
            .lte("published_at", r["decided_at"]) \
            .order("published_at", desc=True).limit(2).execute().data or []
        cases.append({"id": r["id"], "ticker": r["ticker"],
                      "date": str(r["decided_at"])[:10],
                      "called": r["action"],
                      "cnn": f"{r['cnn_direction']} "
                             f"{r['cnn_confidence']:.2f}",
                      "judge_reason": reason,
                      "actual": f"{r['outcome_label']} "
                                f"{(r['outcome_return_1d'] or 0) * 100:+.1f}%",
                      "news_before": [n["headline"][:100] for n in news]})
    return cases


def _too_similar(text, existing):
    # skipping lessons that repeat what an active lesson already says
    words = set(text.lower().split())
    for old in existing:
        old_words = set(old.lower().split())
        overlap = len(words & old_words) / max(len(words | old_words), 1)
        if overlap > 0.5:
            return True
    return False


def distill_lessons():
    # asking gemini for systematic error patterns, then storing survivors
    cases = _wrong_cases()
    if len(cases) < MIN_MISTAKES:
        print(f"lessons: only {len(cases)} scored mistakes — "
              f"waiting for {MIN_MISTAKES} before distilling")
        return

    existing = get_active_lessons(limit=MAX_ACTIVE_LESSONS)
    prompt = ("You are auditing a trading research system's mistakes. Below "
              "are recent wrong calls with the system's stated reasoning, "
              "the CNN signal it saw, the news before the decision, and "
              "what actually happened. Identify AT MOST 2 SYSTEMATIC "
              "patterns — recurring reasons for error, not one-off bad "
              "luck. Each lesson must be one sentence of actionable "
              "guidance for future debates and must cite which case ids "
              "show the pattern. If no systematic pattern exists, return "
              'an empty list. Respond with ONLY json: {"lessons": '
              '[{"lesson_text": str, "case_ids": [int]}]}\n\n'
              "ALREADY-KNOWN LESSONS (do not repeat):\n"
              + json.dumps(existing)[:800] + "\n\nMISTAKES:\n"
              + json.dumps(cases, default=str)[:6000])

    reply = parse_json_reply(ask("gemini", prompt), LESSON_SCHEMA)
    if not reply or not isinstance(reply.get("lessons"), list):
        print("lessons: no valid reply from distiller")
        return

    stored = 0
    for lesson in reply["lessons"][:2]:
        text = str(lesson.get("lesson_text", "")).strip()[:500]
        if not text or _too_similar(text, existing):
            continue
        get_client().table("lessons").insert(
            {"lesson_text": text,
             "evidence": {"case_ids": lesson.get("case_ids", [])[:10]}}) \
            .execute()
        existing.append(text)
        stored += 1
        print(f"lessons: learned — {text}")

    # retiring the oldest lessons beyond the active cap
    active = get_client().table("lessons").select("id") \
        .eq("active", True).order("created_at", desc=True).execute().data or []
    for row in active[MAX_ACTIVE_LESSONS:]:
        get_client().table("lessons").update({"active": False}) \
            .eq("id", row["id"]).execute()
    if stored == 0:
        print("lessons: no new systematic patterns found")
