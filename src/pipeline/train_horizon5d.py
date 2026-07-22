"""training the 5-day regime model behind a challenger gate.

Lives at src/pipeline/train_horizon5d.py. Run:
    export PYTHONPATH=$PWD/src STOCK_LENS_BASE=$PWD/stock-lens-data
    python -m pipeline.train_horizon5d              # full universe
    python -m pipeline.train_horizon5d --tickers 50 # faster

Trains a cnn1d on the 5-DAY forward-return label (±~2.24%, i.e. 1%*sqrt(5),
matching the horizon sweep that justified this model). Saves SEPARATE
artifacts (seq_model_5d.pt / seq_meta_5d.pkl / seq_scaler_5d.pkl) so it never
collides with the daily champion. Challenger-gated: if a 5d model already
exists, the new one only deploys when it wins on the held-out recent window.
"""

import os
import math
import shutil
import pickle
import argparse
from datetime import date, timedelta
import numpy as np

from core.config import MODEL_PATH, SEQ_WINDOW
from core.helpers import log, section
from pipeline.oos_evaluation import prepare_oos_frame
from pipeline.sequence_models import train_eval_seq, score_seq_model, \
    make_torch_model
from pipeline.retrain_cnn import tech_feature_cols

TRAIL_YEARS = 5
EVAL_DAYS = 60
WARMUP_DAYS = 120
H5_DAYS = 5
H5_THRESHOLD = 0.01 * math.sqrt(H5_DAYS)     # ±2.236%
ROSTER = ["cnn1d", "lstm", "gru", "tcn", "transformer"]  # same roster as the daily tournament


def build_sequences_5d(frame, feature_cols, window=SEQ_WINDOW):
    X, y = [], []
    for _, grp in frame.groupby("symbol", sort=False):
        grp = grp.sort_values("date")
        feats = grp[feature_cols].values.astype("float32")
        closes = grp["close"].values.astype("float32")
        for i in range(window, len(grp) - H5_DAYS):
            X.append(feats[i - window:i])
            fwd = (closes[i + H5_DAYS] - closes[i]) / closes[i]
            y.append(0 if fwd < -H5_THRESHOLD
                     else (2 if fwd > H5_THRESHOLD else 1))
    if not X:
        return None, None
    return np.asarray(X, dtype="float32"), np.asarray(y, dtype="int64")


def load_existing_5d():
    meta_path = os.path.join(MODEL_PATH, "seq_meta_5d.pkl")
    if not os.path.exists(meta_path):
        return None
    try:
        import torch
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        with open(os.path.join(MODEL_PATH, "seq_scaler_5d.pkl"), "rb") as f:
            scaler = pickle.load(f)
        model = make_torch_model(meta["kind"], meta["n_features"],
                                 meta["head"], window=meta["window"])
        model.load_state_dict(torch.load(
            os.path.join(MODEL_PATH, "seq_model_5d.pt"), map_location="cpu"))
        model.eval()
        return {"model": model, "scaler": scaler, "meta": meta}
    except Exception as e:
        log(f"existing 5d model unavailable ({e})")
        return None


def score_incumbent(incumbent, eval_df):
    if incumbent is None:
        return None
    meta = incumbent["meta"]
    df = eval_df.copy()
    means = dict(zip(meta["feature_cols"], incumbent["scaler"].mean_))
    for c in meta["feature_cols"]:
        df[c] = df[c].fillna(means[c]) if c in df.columns else means[c]
    df[meta["feature_cols"]] = incumbent["scaler"].transform(
        df[meta["feature_cols"]])
    X, y = build_sequences_5d(df, meta["feature_cols"],
                              window=meta["window"])
    if X is None:
        return None
    f1, _ = score_seq_model(incumbent["model"], meta["head"], X, y)
    return f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=None)
    args = ap.parse_args()

    section("TRAINING THE 5-DAY REGIME MODEL (challenger-gated)")
    from sklearn.preprocessing import StandardScaler
    import torch

    end = date.today()
    start = end - timedelta(days=int(TRAIL_YEARS * 365 + WARMUP_DAYS))
    df = prepare_oos_frame(limit=args.tickers, start=str(start), end=str(end))
    if df is None or df.empty:
        log("download failed — aborting")
        return
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    feature_cols = tech_feature_cols(df)
    log(f"rows: {df.shape[0]:,} | features: {len(feature_cols)} | "
        f"5d threshold ±{H5_THRESHOLD:.3%}")

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

    Xtr, ytr = build_sequences_5d(train, feature_cols)
    Xva, yva = build_sequences_5d(val, feature_cols)
    Xe, ye = build_sequences_5d(eval_scaled, feature_cols)
    if Xtr is None or Xe is None:
        log("not enough sequences — aborting")
        return
    counts = np.bincount(ytr, minlength=3).astype(float)
    cw = (counts.sum() / (3 * np.maximum(counts, 1))).tolist()
    rtr = np.zeros(len(ytr), dtype="float32")

    # training the FULL roster on 5d labels and electing the best on the
    # holdout — mirroring the daily tournament in retrain_cnn.py rather than
    # assuming one architecture wins for this horizon
    roster_results = {}
    for kind in ROSTER:
        try:
            vf1, mdl = train_eval_seq(kind, "classification",
                                      Xtr, rtr, ytr, Xva, yva,
                                      return_model=True, class_weights=cw)
            ef1, _ = score_seq_model(mdl, "classification", Xe, ye)
            roster_results[kind] = (ef1, vf1, mdl)
            log(f"  {kind:12s}: val {vf1:.4f} | eval {ef1:.4f}")
        except Exception as e:
            log(f"  {kind:12s}: failed ({e})")
    if not roster_results:
        log("no architecture trained — aborting")
        return
    best_kind = max(roster_results, key=lambda k: roster_results[k][0])
    chal_f1, _, model = roster_results[best_kind]
    log(f"best 5d architecture: {best_kind} (eval {chal_f1:.4f})")

    incumbent = load_existing_5d()
    inc_f1 = score_incumbent(incumbent, eval_df)
    if inc_f1 is not None:
        log(f"incumbent : eval {inc_f1:.4f}")
        if inc_f1 >= chal_f1:
            log("[GATE] incumbent 5d model retained — no changes made")
            return
    else:
        log("[GATE] no existing 5d model — challenger deploys by default")

    # archive any old 5d artifacts, then refit on all pre-eval data and deploy
    archive = os.path.join(MODEL_PATH, "archive", f"5d_{date.today()}")
    os.makedirs(archive, exist_ok=True)
    for fn in ("seq_model_5d.pt", "seq_meta_5d.pkl", "seq_scaler_5d.pkl"):
        p = os.path.join(MODEL_PATH, fn)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(archive, fn))

    deploy_scaler = StandardScaler().fit(fit_df[feature_cols])
    fit_scaled = fit_df.copy()
    fit_scaled[feature_cols] = deploy_scaler.transform(
        fit_scaled[feature_cols])
    aX, ay = build_sequences_5d(fit_scaled, feature_cols)
    _, deploy_model = train_eval_seq(best_kind, "classification",
                                     aX, np.zeros(len(ay), dtype="float32"),
                                     ay, aX, ay, return_model=True,
                                     class_weights=cw)
    torch.save(deploy_model.state_dict(),
               os.path.join(MODEL_PATH, "seq_model_5d.pt"))
    with open(os.path.join(MODEL_PATH, "seq_meta_5d.pkl"), "wb") as f:
        pickle.dump({"kind": best_kind, "head": "classification",
                     "feature_cols": feature_cols, "window": SEQ_WINDOW,
                     "threshold": H5_THRESHOLD, "horizon_days": H5_DAYS,
                     "n_features": len(feature_cols),
                     "classes": ["Down", "Neutral", "Up"],
                     "trained_through": str(eval_start)[:10]}, f)
    with open(os.path.join(MODEL_PATH, "seq_scaler_5d.pkl"), "wb") as f:
        pickle.dump(deploy_scaler, f)
    log("5d regime model deployed (seq_model_5d.pt)")


if __name__ == "__main__":
    main()
