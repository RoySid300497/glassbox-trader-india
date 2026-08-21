"""measuring whether a candidate feature set actually adds edge.

Lives at src/pipeline/measure_feature.py. Run from the repo root:
    export PYTHONPATH=$PWD/src STOCK_LENS_BASE=$PWD/stock-lens-data
    python -m pipeline.measure_feature --candidate-csv flow.csv --tickers 25

The candidate CSV must have a `date` column (YYYY-MM-DD), optionally a
`symbol` column (per-ticker features; omit for market-wide features), and
one or more numeric feature columns. It is merged onto the standard OOS
frame; market-wide features broadcast to every ticker on that date.

Then the SAME cnn1d is trained twice on the SAME chronological split —
baseline features vs baseline + candidate — and judged on the SAME holdout
against the same stratified baseline. The only difference between the two
runs is the candidate columns, so the edge delta is attributable to them.

Decision rule (honest): adopt the candidate only if it adds >= +0.02 edge.
Below that is noise on a 60-day holdout — do not adopt on hope.
"""

import argparse
from datetime import date, timedelta
import numpy as np

from core.helpers import log, section
from pipeline.oos_evaluation import prepare_oos_frame
from pipeline.sequence_models import build_sequences, train_eval_seq, \
    score_seq_model
from pipeline.retrain_cnn import tech_feature_cols

TRAIL_YEARS = 3
EVAL_DAYS = 60
WARMUP_DAYS = 120


def baselines(y, n_boot=200, seed=42):
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=3).astype(float)
    p = counts / counts.sum()
    strat = [f1_score(y, rng.choice(3, size=len(y), p=p), average="macro")
             for _ in range(n_boot)]
    return float(np.mean(strat))


def train_once(train, val, ev, feature_cols):
    from sklearn.preprocessing import StandardScaler
    tr, va, e = train.copy(), val.copy(), ev.copy()
    scaler = StandardScaler().fit(tr[feature_cols])
    for part in (tr, va, e):
        part[feature_cols] = scaler.transform(part[feature_cols])
    Xtr, rtr, ytr = build_sequences(tr, feature_cols)
    Xva, _, yva = build_sequences(va, feature_cols)
    Xe, _, ye = build_sequences(e, feature_cols)
    if Xtr is None or Xe is None:
        return None, None
    counts = np.bincount(ytr, minlength=3).astype(float)
    cw = (counts.sum() / (3 * np.maximum(counts, 1))).tolist()
    _, model = train_eval_seq("cnn1d", "classification",
                              Xtr, rtr, ytr, Xva, yva,
                              return_model=True, class_weights=cw)
    f1, _ = score_seq_model(model, "classification", Xe, ye)
    return f1, baselines(ye)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-csv", required=True,
                    help="csv with date[,symbol] + numeric feature columns")
    ap.add_argument("--tickers", type=int, default=None)
    args = ap.parse_args()

    import pandas as pd
    section("FEATURE ABLATION — DOES THE CANDIDATE ADD EDGE?")
    cand = pd.read_csv(args.candidate_csv)
    cand["date"] = pd.to_datetime(cand["date"])
    per_ticker = "symbol" in cand.columns
    feat_new = [c for c in cand.columns
                if c not in ("date", "symbol")
                and pd.api.types.is_numeric_dtype(cand[c])]
    if not feat_new:
        log("candidate csv has no numeric feature columns — aborting")
        return
    log(f"candidate features: {feat_new} "
        f"({'per-ticker' if per_ticker else 'market-wide'})")

    end = date.today()
    start = end - timedelta(days=int(TRAIL_YEARS * 365 + WARMUP_DAYS))
    df = prepare_oos_frame(limit=args.tickers, start=str(start), end=str(end))
    if df is None or df.empty:
        log("download failed — aborting")
        return
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    keys = ["date", "symbol"] if per_ticker else ["date"]
    df = df.merge(cand[keys + feat_new], on=keys, how="left")
    # forward-fill within each ticker so publication gaps don't punch holes,
    # then fill remaining leading NaNs with the column median
    for c in feat_new:
        df[c] = df.groupby("symbol")[c].ffill()
        df[c] = df[c].fillna(df[c].median())
    coverage = df[feat_new[0]].notna().mean()
    log(f"rows: {df.shape[0]:,} | candidate coverage after ffill: "
        f"{coverage:.0%}")
    if coverage < 0.5:
        log("WARNING: candidate covers <50% of the frame — the test will "
            "mostly measure fill values, not the feature. collect more "
            "history before trusting this result.")

    base_cols = tech_feature_cols(df)
    base_cols = [c for c in base_cols if c not in feat_new]
    dates = np.sort(df["date"].unique())
    eval_start = dates[-EVAL_DAYS]
    fit_df = df[df["date"] < eval_start]
    ev = df[df["date"] >= eval_start].copy()
    fit_dates = np.sort(fit_df["date"].unique())
    val_start = fit_dates[int(len(fit_dates) * 0.85)]
    train = fit_df[fit_df["date"] < val_start]
    val = fit_df[fit_df["date"] >= val_start]

    log("--- baseline (existing features only) ---")
    f1_base, strat = train_once(train, val, ev, base_cols)
    log(f"  macro-F1 {f1_base:.4f} | stratified {strat:.4f} | "
        f"EDGE {f1_base - strat:+.4f}")

    log("--- baseline + candidate ---")
    f1_cand, strat2 = train_once(train, val, ev, base_cols + feat_new)
    log(f"  macro-F1 {f1_cand:.4f} | stratified {strat2:.4f} | "
        f"EDGE {f1_cand - strat2:+.4f}")

    section("VERDICT")
    delta = (f1_cand - strat2) - (f1_base - strat)
    log(f"edge delta from candidate: {delta:+.4f}")
    if delta >= 0.02:
        log("VERDICT: candidate adds MEANINGFUL edge — wire it into the "
            "feature pipeline and retrain")
    elif delta > 0:
        log("VERDICT: tiny positive delta — within noise on this holdout; "
            "collect more history and re-test before adopting")
    else:
        log("VERDICT: no added edge — do NOT adopt; the honest answer is "
            "this data is already priced in or redundant")


if __name__ == "__main__":
    main()
