"""grading evidence concepts so judges can weigh claims by reliability

grades start from research-informed priors and are automatically replaced
by the system's own measured hit-rate edge once a concept has been cited
in enough scored decisions — priors decay into data
"""

from engine.memory import get_client

MIN_MEASURED = 10            # citations needed before data replaces prior

# prior grades: A = well-documented effects, B = reasonable evidence,
# C = computable trader lore awaiting proof, D = weak or self-referential
PRIOR_GRADES = {
    "insider_activity": ("A", "clustered open-market insider buying is one "
                              "of the best-documented bullish signals"),
    "congress_trading": ("B", "congressional disclosures are delayed but "
                              "show documented anomalous returns"),
    "connors_rsi2": ("A", "published mean-reversion system with decades of "
                          "replicated backtests"),
    "market_stage": ("B", "trend and regime persistence are established "
                          "market effects"),
    "ema_regime": ("B", "trading with the long-term trend has robust "
                        "empirical support"),
    "earnings": ("B", "earnings events are genuine binary risks — proximity "
                      "deserves caution"),
    "news_sentiment": ("B", "event-driven moves are real; sentiment is a "
                            "noisy but informative proxy"),
    "macro_news": ("B", "market-wide events move single names through "
                        "supply, cost, and risk-appetite channels"),
    "prior_day_break": ("C", "gaps and closes beyond the prior day's range "
                            "signal breakout or failed-breakout structure; "
                            "unproven on daily S&P names until measured"),
    "market_context": ("B", "index trend and volatility regime condition "
                            "every single-name bet"),
    "overnight_gap": ("B", "gaps carry information about overnight flows"),
    "divergence": ("C", "momentum divergence has mixed formal evidence"),
    "vix_fix_capitulation": ("C", "volatility climaxes often mark bottoms "
                                  "but timing is imprecise"),
    "volume": ("C", "volume confirms trends but rarely predicts alone"),
    "liquidity_sweep": ("C", "trader-taught concept, mechanically detected, "
                             "unproven on daily bars"),
    "fvg_order_block": ("C", "trader-taught zone concept awaiting scored "
                             "evidence"),
    "market_structure": ("C", "structure breaks lag price by construction"),
    "first_pullback": ("C", "continuation heuristic awaiting evidence"),
    "range_or_squeeze": ("C", "compression precedes expansion but not "
                              "direction"),
    "candles_momentum": ("C", "single-bar patterns are weak alone"),
    "adx_trend": ("C", "trend-strength gauge, not directional"),
    "model_signal": ("C", "price-only model with a thin measured edge — "
                          "treat as a calibrated prior, not an oracle"),
    "track_record": ("D", "small-sample history; suggestive at best"),
    "thesis_lessons": ("D", "self-referential guidance; weigh lightly"),
}


def _measured_grades():
    # pulling the latest weekly snapshot of scored evidence effectiveness
    rows = get_client().table("evidence_stats") \
        .select("concept,cited,hit_rate,baseline,week_ending") \
        .order("week_ending", desc=True).limit(60).execute().data or []
    if not rows:
        return {}
    latest = rows[0]["week_ending"]
    out = {}
    for r in rows:
        if r["week_ending"] != latest or r["cited"] < MIN_MEASURED:
            continue
        edge = float(r["hit_rate"]) - float(r["baseline"])
        grade = ("A" if edge > 0.10 else "B" if edge > 0.03
                 else "C" if edge > -0.05 else "D")
        out[r["concept"]] = (grade,
                             f"measured: {float(r['hit_rate']):.0%} hit rate "
                             f"over {r['cited']} scored citations "
                             f"({edge:+.0%} vs baseline)")
    return out


def evidence_reliability_block():
    # composing the grade table judges see: measured beats prior
    grades = dict(PRIOR_GRADES)
    try:
        grades.update(_measured_grades())
    except Exception:
        pass  # priors alone are fine when no snapshots exist yet
    return {
        "how_to_use": ("weigh conflicting claims by these grades — A is "
                       "strongest evidence, D weakest; 'measured' grades "
                       "come from this system's own scored track record "
                       "and supersede prior expectations"),
        "grades": {k: {"grade": g, "why": w}
                   for k, (g, w) in sorted(grades.items(),
                                           key=lambda kv: kv[1][0])},
    }
