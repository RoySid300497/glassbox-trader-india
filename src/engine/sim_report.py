"""paper-trading simulation observations and plots

reads the simulated book (positions, decisions, equity) from supabase and
renders a full set of performance plots and stat summaries into OBS_PATH,
so every simulation run leaves an auditable visual trail
"""

import os
import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from core.config import OBS_PATH
from engine.memory import get_client


def _save(fig_name):
    path = os.path.join(OBS_PATH, fig_name)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  [sim-plot] {fig_name}")


def _load_decisions():
    rows = get_client().table("decisions").select("*").order(
        "decided_at").execute().data or []
    return pd.DataFrame(rows)


def _load_positions():
    rows = get_client().table("positions").select("*").execute().data or []
    return pd.DataFrame(rows)


def _sim_equity():
    rows = get_client().table("config").select("value").eq(
        "key", "sim_equity").execute().data
    return float(rows[0]["value"]) if rows else None


def simulation_report():
    # generating the full observation set; each plot guarded so one empty
    # table never blocks the rest
    os.makedirs(OBS_PATH, exist_ok=True)
    dec = _load_decisions()
    print("\n=== SIMULATION OBSERVATIONS ===")

    if not dec.empty:
        _plot_action_mix(dec)
        _plot_decisions_over_time(dec)
        _plot_confidence_dist(dec)
        _plot_outcome_by_action(dec)
        _plot_accuracy_over_time(dec)
        _plot_ticker_frequency(dec)
        _text_summary(dec)
    else:
        print("  no decisions yet — plots will populate after the first runs")

    pos = _load_positions()
    if not pos.empty:
        _plot_position_pnl(pos)
        _plot_open_vs_closed(pos)

    eq = _sim_equity()
    if eq is not None:
        print(f"  simulated equity: Rs {eq:,.2f}")


def _plot_action_mix(dec):
    if "action" not in dec:
        return
    counts = dec["action"].value_counts()
    plt.figure(figsize=(7, 4))
    counts.plot(kind="bar", color=["#34d399", "#fb7185", "#71717a"])
    plt.title("decision action mix")
    plt.ylabel("count")
    _save("sim_action_mix.png")


def _plot_decisions_over_time(dec):
    if "decided_at" not in dec:
        return
    d = dec.copy()
    d["day"] = pd.to_datetime(d["decided_at"]).dt.date
    daily = d.groupby(["day", "action"]).size().unstack(fill_value=0)
    daily.plot(kind="area", stacked=True, figsize=(11, 4), alpha=0.8)
    plt.title("decisions per day by action")
    plt.ylabel("count")
    _save("sim_decisions_over_time.png")


def _plot_confidence_dist(dec):
    col = next((c for c in ("cnn_confidence",) if c in dec), None)
    if not col:
        return
    vals = pd.to_numeric(dec[col], errors="coerce").dropna()
    if vals.empty:
        return
    plt.figure(figsize=(7, 4))
    plt.hist(vals, bins=25, color="#6366f1", alpha=0.85)
    plt.title("model confidence distribution")
    plt.xlabel("confidence")
    _save("sim_confidence_dist.png")


def _plot_outcome_by_action(dec):
    if "action" not in dec or "was_correct" not in dec:
        return
    d = dec.dropna(subset=["was_correct"])
    if d.empty:
        return
    rate = d.groupby("action")["was_correct"].mean()
    plt.figure(figsize=(7, 4))
    rate.plot(kind="bar", color="#38bdf8")
    plt.axhline(0.33, color="#fb7185", ls="--", label="random (0.33)")
    plt.title("hit rate by action (scored decisions)")
    plt.ylabel("accuracy")
    plt.legend()
    _save("sim_hit_rate_by_action.png")


def _plot_accuracy_over_time(dec):
    if "was_correct" not in dec or "decided_at" not in dec:
        return
    d = dec.dropna(subset=["was_correct"]).copy()
    if d.empty:
        return
    d["day"] = pd.to_datetime(d["decided_at"]).dt.date
    roll = d.set_index("day")["was_correct"].astype(float) \
        .rolling(10, min_periods=3).mean()
    plt.figure(figsize=(11, 4))
    plt.plot(roll.index, roll.values, color="#34d399")
    plt.axhline(0.33, color="#fb7185", ls="--", label="random")
    plt.title("rolling 10-decision hit rate")
    plt.ylabel("accuracy")
    plt.legend()
    _save("sim_accuracy_over_time.png")


def _plot_ticker_frequency(dec):
    if "ticker" not in dec:
        return
    top = dec["ticker"].value_counts().head(20)
    plt.figure(figsize=(9, 5))
    top.plot(kind="barh", color="#a78bfa")
    plt.title("most-debated tickers")
    plt.xlabel("decisions")
    _save("sim_ticker_frequency.png")


def _plot_position_pnl(pos):
    if "entry_price" not in pos:
        return
    plt.figure(figsize=(9, 4))
    plt.bar(range(len(pos)), pd.to_numeric(pos.get("qty", 0),
            errors="coerce").fillna(0), color="#38bdf8")
    plt.title("position sizes")
    plt.ylabel("qty")
    _save("sim_position_sizes.png")


def _plot_open_vs_closed(pos):
    if "status" not in pos:
        return
    counts = pos["status"].value_counts()
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color=["#34d399", "#71717a"])
    plt.title("open vs closed positions")
    _save("sim_open_vs_closed.png")


def _text_summary(dec):
    # writing a plain-text stat digest alongside the plots
    lines = ["SIMULATION SUMMARY",
             f"generated: {datetime.now(timezone.utc).isoformat()}",
             f"total decisions: {len(dec)}"]
    if "action" in dec:
        for a, n in dec["action"].value_counts().items():
            lines.append(f"  {a}: {n}")
    if "was_correct" in dec:
        scored = dec.dropna(subset=["was_correct"])
        if not scored.empty:
            lines.append(f"scored decisions: {len(scored)}")
            lines.append(f"overall hit rate: "
                         f"{scored['was_correct'].mean():.3f}")
    path = os.path.join(OBS_PATH, "sim_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [sim-summary] sim_summary.txt")
    for ln in lines:
        print("  " + ln)
