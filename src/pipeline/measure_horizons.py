"""measuring how directional signal evolves across horizons 1..7 days.

Run from the repo root (either repo):
    export PYTHONPATH=$PWD/src STOCK_LENS_BASE=$PWD/stock-lens-data
    python -m pipeline.measure_horizons --tickers 25   # sample (recommended)
    python -m pipeline.measure_horizons                # full universe (slow)

For each horizon N in 1..7 it trains the SAME architecture (cnn1d) on the SAME
features and the SAME chronological split, labeled with the N-day forward
return (an N-day return already accumulates days 1..N). Label thresholds scale
with horizon (~1% x sqrt(N), matching the repo's own 1d:1% / 5d:2% / 10d:3%
convention) so class balance stays comparable.

Because class balances still differ across horizons, raw macro-F1 numbers are
NOT directly comparable — each horizon is judged by its EDGE over its own
stratified-random baseline. The sweep shows where (if anywhere) the market
leaks the most predictable signal.
"""

import argparse
import math
from datetime import date, timedelta
import numpy as np

from core.helpers import log, section
from pipeline.oos_evaluation import prepare_oos_frame
from pipeline.sequence_models import train_eval_seq, score_seq_model
from pipeline.retrain_cnn import tech_feature_cols

TRAIL_YEARS = 3
EVAL_DAYS = 60
WARMUP_DAYS = 120
HORIZONS = list(range(1, 8))          # 1..7 day forward returns


def horizon_threshold(n):
    # ±1% at 1 day, scaling with sqrt(horizon) — matches the repo's own
    # convention (1d:1%, 5d:2%, 10d:3%) within rounding
    return 0.01 * math.sqrt(n)


def build_sequences_h(frame, feature_cols, horizon, window=30):
    # sliding windows labeled with the N-day forward return at the scaled
    # threshold; mirrors build_sequences (which hardcodes the 1-day label)
    thr = horizon_threshold(horizon)
    X, y = [], []
    for _, grp in frame.groupby("symbol", sort=False):
        grp = grp.sort_values("date")
        feats = grp[feature_cols].values.astype("float32")
        closes = grp["close"].values.astype("float32")
        for i in range(window, len(grp) - horizon):
            X.append(feats[i - window:i])
            fwd = (closes[i + horizon] - closes[i]) / closes[i]
            y.append(0 if fwd < -thr else (2 if fwd > thr else 1))
    if not X:
        return None, None
    return np.asarray(X, dtype="float32"), np.asarray(y, dtype="int64")


def baselines(y, n_boot=200, seed=42):
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=3).astype(float)
    p = counts / counts.sum()
    maj_f1 = f1_score(y, np.full_like(y, counts.argmax()), average="macro")
    strat = [f1_score(y, rng.choice(3, size=len(y), p=p), average="macro")
             for _ in range(n_boot)]
    return maj_f1, float(np.mean(strat))


def class_weights_for(y):
    counts = np.bincount(y, minlength=3).astype(float)
    return (counts.sum() / (3 * np.maximum(counts, 1))).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=None)
    args = ap.parse_args()

    section("HORIZON SWEEP — WHERE DOES THE MARKET LEAK PREDICTABLE SIGNAL?")
    from sklearn.preprocessing import StandardScaler

    end = date.today()
    start = end - timedelta(days=int(TRAIL_YEARS * 365 + WARMUP_DAYS))
    df = prepare_oos_frame(limit=args.tickers, start=str(start), end=str(end))
    if df is None or df.empty:
        log("download failed — aborting")
        return
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    feature_cols = tech_feature_cols(df)
    log(f"rows: {df.shape[0]:,} | features: {len(feature_cols)}")

    dates = np.sort(df["date"].unique())
    eval_start = dates[-EVAL_DAYS]
    fit_df = df[df["date"] < eval_start]
    eval_df = df[df["date"] >= eval_start].copy()
    fit_dates = np.sort(fit_df["date"].unique())
    val_start = fit_dates[int(len(fit_dates) * 0.85)]
    train = fit_df[fit_df["date"] < val_start].copy()
    val = fit_df[fit_df["date"] >= val_start].copy()

    scaler = StandardScaler().fit(train[feature_cols])
    for part in (train, val):
        part[feature_cols] = scaler.transform(part[feature_cols])
    eval_scaled = eval_df.copy()
    eval_scaled[feature_cols] = scaler.transform(eval_scaled[feature_cols])

    results = {}
    for n in HORIZONS:
        thr = horizon_threshold(n)
        log(f"--- horizon {n}d (threshold ±{thr:.3%}) ---")
        Xtr, ytr = build_sequences_h(train, feature_cols, n)
        Xva, yva = build_sequences_h(val, feature_cols, n)
        Xe, ye = build_sequences_h(eval_scaled, feature_cols, n)
        if Xtr is None or Xe is None or Xva is None:
            log(f"  {n}d: not enough sequences — skipped")
            continue
        cw = class_weights_for(ytr)
        rtr = np.zeros(len(ytr), dtype="float32")   # unused by classification
        _, model = train_eval_seq("cnn1d", "classification",
                                  Xtr, rtr, ytr, Xva, yva,
                                  return_model=True, class_weights=cw)
        f1, _ = score_seq_model(model, "classification", Xe, ye)
        maj, strat = baselines(ye)
        results[n] = {"f1": f1, "strat": strat, "maj": maj,
                      "edge": f1 - strat, "n": len(ye)}
        log(f"  {n}d: holdout macro-F1 {f1:.4f} | stratified {strat:.4f} | "
            f"majority {maj:.4f} | EDGE {f1 - strat:+.4f} on {len(ye):,}")

    section("SWEEP SUMMARY (judge by EDGE, not raw F1)")
    log(f"{'horizon':>8} {'macro-F1':>9} {'baseline':>9} {'EDGE':>8} {'n':>8}")
    for n in sorted(results):
        r = results[n]
        log(f"{n:>7}d {r['f1']:>9.4f} {r['strat']:>9.4f} "
            f"{r['edge']:>+8.4f} {r['n']:>8,}")

    if results:
        best = max(results, key=lambda k: results[k]["edge"])
        base = results.get(1)
        log("")
        log(f"best horizon by edge: {best}d ({results[best]['edge']:+.4f})")
        if base:
            gain = results[best]["edge"] - base["edge"]
            if best == 1 or gain <= 0.01:
                log("VERDICT: no horizon beats daily meaningfully — the "
                    "efficiency ceiling holds across 1-7 days; a regime "
                    "layer will not manufacture signal")
            elif gain <= 0.03:
                log(f"VERDICT: {best}d edges out daily by {gain:+.4f} — "
                    f"marginal; a regime layer may help slightly")
            else:
                log(f"VERDICT: {best}d beats daily by {gain:+.4f} — "
                    f"meaningful extra signal; building the {best}d regime "
                    f"layer is justified")


if __name__ == "__main__":
    main()
