"""retraining the sequence model on recent data behind a challenger gate"""

import os
import shutil
import pickle
import argparse
from datetime import date, timedelta
import numpy as np
import pandas as pd
from core.config import (MODEL_PATH, BASE_FEATURE_COLS, FUND_FILL_COLS,
                         LAG_COLS, LAG_DAYS, RETURN_FEATURES,
                         RELATIVE_FEATURES, SEQ_WINDOW, SEQ_THRESHOLD)
from core.helpers import log, section
from pipeline.oos_evaluation import prepare_oos_frame
from pipeline.sequence_models import (build_sequences, train_eval_seq,
                                      score_seq_model)
from inference.predictors import load_seq_predictor, load_rf_predictor

TRAIL_YEARS = 5          # training on this many trailing years of data
ROSTER = ["cnn1d", "lstm", "gru", "tcn", "transformer"]
EVAL_DAYS = 60           # holding out this many recent days for the gate
WARMUP_DAYS = 120        # padding the download so indicators have history


def tech_feature_cols(df):
    # selecting live-available features, dropping dataset-only fundamentals
    cols = [c for c in BASE_FEATURE_COLS if c not in FUND_FILL_COLS] \
        + [f"{c}_lag{l}" for c in LAG_COLS for l in LAG_DAYS] \
        + RETURN_FEATURES + RELATIVE_FEATURES
    return [c for c in cols if c in df.columns]


def score_champion(champion, eval_df):
    # scoring the currently deployed model on the held-out recent window
    if champion is None:
        return None
    meta = champion["meta"]
    df = eval_df.copy()
    means = dict(zip(meta["feature_cols"], champion["scaler"].mean_))
    for c in meta["feature_cols"]:
        df[c] = df[c].fillna(means[c]) if c in df.columns else means[c]
    df[meta["feature_cols"]] = champion["scaler"].transform(
        df[meta["feature_cols"]])
    X, _, y = build_sequences(df, meta["feature_cols"],
                              window=meta["window"])
    if X is None:
        return None
    f1, _ = score_seq_model(champion["model"], meta["head"], X, y)
    return f1


def retrain(limit=None):
    section("RETRAINING SEQUENCE MODEL ON RECENT DATA")
    from sklearn.preprocessing import StandardScaler

    # downloading the trailing window plus indicator warmup
    end = date.today()
    start = end - timedelta(days=int(TRAIL_YEARS * 365 + WARMUP_DAYS))
    df = prepare_oos_frame(limit=limit, start=str(start), end=str(end))
    if df is None or df.empty:
        log("download failed — aborting retrain")
        return
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    feature_cols = tech_feature_cols(df)
    log(f"rows: {df.shape[0]:,} | live-available features: {len(feature_cols)}")

    # holding out the most recent trading days for the challenger gate
    dates = np.sort(df["date"].unique())
    eval_start = dates[-EVAL_DAYS]
    fit_df = df[df["date"] < eval_start]
    eval_df = df[df["date"] >= eval_start].copy()

    # splitting fit data chronologically for training and validation
    fit_dates = np.sort(fit_df["date"].unique())
    val_start = fit_dates[int(len(fit_dates) * 0.85)]
    train = fit_df[fit_df["date"] < val_start].copy()
    val = fit_df[fit_df["date"] >= val_start].copy()
    log(f"train days: {len(fit_dates[fit_dates < val_start])} | "
        f"val days: {len(fit_dates[fit_dates >= val_start])} | "
        f"eval days: {EVAL_DAYS}")

    # scaling on training rows only, then training the full roster
    import torch
    scaler = StandardScaler().fit(train[feature_cols])
    for part in (train, val):
        part[feature_cols] = scaler.transform(part[feature_cols])
    Xtr, rtr, ctr = build_sequences(train, feature_cols)
    Xva, _, cva = build_sequences(val, feature_cols)
    eval_scaled = eval_df.copy()
    eval_scaled[feature_cols] = scaler.transform(eval_scaled[feature_cols])
    Xe, _, ye = build_sequences(eval_scaled, feature_cols)

    # weighting classes inversely to frequency so neutral cannot dominate
    counts = np.bincount(ctr, minlength=3).astype(float)
    class_weights = (counts.sum() / (3 * np.maximum(counts, 1))).tolist()
    log(f"class weights (Down/Neutral/Up): "
        f"{[round(w, 2) for w in class_weights]}")

    roster_results = {}
    shadow_dir = os.path.join(MODEL_PATH, "shadow")
    for kind in ROSTER:
        log(f"training {kind} on {len(Xtr):,} sequences...")
        try:
            val_f1, model = train_eval_seq(kind, "classification",
                                           Xtr, rtr, ctr, Xva, cva,
                                           return_model=True,
                                           class_weights=class_weights)
            f1, _ = score_seq_model(model, "classification", Xe, ye)
            roster_results[kind] = (f1, model)
            log(f"  {kind}: val {val_f1:.4f} | eval {f1:.4f}")
            # saving every architecture as a shadow competitor
            kdir = os.path.join(shadow_dir, kind)
            os.makedirs(kdir, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(kdir, "seq_model.pt"))
            with open(os.path.join(kdir, "seq_meta.pkl"), "wb") as f:
                pickle.dump({"kind": kind, "head": "classification",
                             "feature_cols": feature_cols,
                             "window": SEQ_WINDOW,
                             "threshold": SEQ_THRESHOLD,
                             "n_features": len(feature_cols),
                             "classes": ["Down", "Neutral", "Up"]}, f)
            with open(os.path.join(kdir, "seq_scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)
        except Exception as e:
            log(f"  {kind}: failed ({e}) — skipping")
    if not roster_results:
        log("no roster model trained — aborting")
        return

    # training the tabular xgboost challenger on the same scaled split
    try:
        from xgboost import XGBClassifier
        from sklearn.metrics import f1_score
        label_map = {"Down": 0, "Neutral": 1, "Up": 2}
        xgb = XGBClassifier(n_estimators=300, max_depth=6,
                            learning_rate=0.08, subsample=0.8,
                            colsample_bytree=0.8, eval_metric="mlogloss",
                            n_jobs=4)
        from sklearn.utils.class_weight import compute_sample_weight
        xgb.fit(train[feature_cols], train["true_label"].map(label_map),
                sample_weight=compute_sample_weight(
                    "balanced", train["true_label"]),
                eval_set=[(val[feature_cols],
                           val["true_label"].map(label_map))],
                verbose=False)
        xgb_f1 = f1_score(eval_scaled["true_label"].map(label_map),
                          xgb.predict(eval_scaled[feature_cols]),
                          average="macro")
        log(f"  xgboost: eval {xgb_f1:.4f} (shadow only)")
        xdir = os.path.join(shadow_dir, "xgboost")
        os.makedirs(xdir, exist_ok=True)
        with open(os.path.join(xdir, "xgb.pkl"), "wb") as f:
            pickle.dump({"model": xgb, "feature_cols": feature_cols,
                         "scaler": scaler,
                         "classes": ["Down", "Neutral", "Up"]}, f)
    except Exception as e:
        log(f"  xgboost: failed ({e}) — skipping")

    # retraining the random forest so the electable tabular model stays fresh
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import f1_score
        le = LabelEncoder().fit(["Down", "Neutral", "Up"])
        imputer = SimpleImputer(strategy="median").fit(train[feature_cols])
        rf_new = RandomForestClassifier(n_estimators=180, max_depth=16,
                                        min_samples_leaf=5, n_jobs=-1,
                                        class_weight="balanced_subsample",
                                        random_state=42)
        rf_new.fit(imputer.transform(train[feature_cols]),
                   le.transform(train["true_label"]))
        rf_f1 = f1_score(le.transform(eval_scaled["true_label"]),
                         rf_new.predict(imputer.transform(
                             eval_scaled[feature_cols])), average="macro")
        log(f"  random_forest challenger: eval {rf_f1:.4f}")

        # scoring the deployed rf on the same window for its mini-gate
        old_rf = load_rf_predictor()
        old_f1 = None
        if old_rf:
            frame = eval_df.copy()
            for c in old_rf["feature_cols"]:
                if c not in frame.columns:
                    frame[c] = np.nan
            Xo = old_rf["scaler"].transform(
                old_rf["imputer"].transform(frame[old_rf["feature_cols"]]))
            old_f1 = f1_score(
                old_rf["label_encoder"].transform(frame["true_label"]),
                old_rf["model"].predict(Xo), average="macro")
            log(f"  random_forest incumbent : eval {old_f1:.4f}")

        if old_f1 is None or rf_f1 > old_f1:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                pickle.dump(rf_new, tmp)
                tmp_path = tmp.name
            size_mb = os.path.getsize(tmp_path) / 1e6
            os.unlink(tmp_path)
            if size_mb > 90:
                log(f"  random_forest: {size_mb:.0f}MB exceeds repo "
                    f"limit — incumbent kept")
            else:
                rf_archive = os.path.join(MODEL_PATH, "archive",
                                          str(date.today()))
                os.makedirs(rf_archive, exist_ok=True)
                rf_files = ["rf_model.pkl", "scaler.pkl", "imputer.pkl",
                            "label_encoder.pkl", "feature_cols.pkl"]
                for fn in rf_files:
                    p = os.path.join(MODEL_PATH, fn)
                    if os.path.exists(p):
                        shutil.copy2(p, os.path.join(rf_archive, fn))
                for fn, obj in [("rf_model.pkl", rf_new),
                                ("scaler.pkl", scaler),
                                ("imputer.pkl", imputer),
                                ("label_encoder.pkl", le),
                                ("feature_cols.pkl", feature_cols)]:
                    with open(os.path.join(MODEL_PATH, fn), "wb") as f:
                        pickle.dump(obj, f)
                log(f"  random_forest deployed ({size_mb:.0f}MB) — "
                    f"incumbent archived")
        else:
            log("  random_forest incumbent retained")
    except Exception as e:
        log(f"  random_forest retrain failed ({e}) — incumbent kept")

    best_kind = max(roster_results, key=lambda k: roster_results[k][0])
    chal_f1, challenger = roster_results[best_kind]
    champion = load_seq_predictor()
    champ_f1 = score_champion(champion, eval_df)
    log(f"[GATE] best challenger = {best_kind} "
        f"eval macro_f1 = {chal_f1:.4f}")
    log(f"[GATE] champion   eval macro_f1 = "
        f"{champ_f1:.4f}" if champ_f1 is not None else
        "[GATE] no champion found — challenger deploys by default")

    # keeping the champion when it still wins on recent data
    if champ_f1 is not None and champ_f1 >= chal_f1:
        log("champion retained — no artifact changes made")
        return

    # archiving old artifacts before deploying the challenger
    archive = os.path.join(MODEL_PATH, "archive", str(date.today()))
    os.makedirs(archive, exist_ok=True)
    for fn in ("seq_model.pt", "seq_meta.pkl", "seq_scaler.pkl"):
        src_path = os.path.join(MODEL_PATH, fn)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(archive, fn))
    log(f"old artifacts archived to {archive}")

    # keeping the dethroned incumbent competing under its own label
    if champion is not None:
        prev_dir = os.path.join(
            shadow_dir, f"{champion['meta'].get('kind', 'cnn1d')}_prev")
        os.makedirs(prev_dir, exist_ok=True)
        for fn in ("seq_model.pt", "seq_meta.pkl", "seq_scaler.pkl"):
            src_path = os.path.join(MODEL_PATH, fn)
            if os.path.exists(src_path):
                shutil.copy2(src_path, os.path.join(prev_dir, fn))
        log(f"dethroned incumbent stays in the tournament as "
            f"{os.path.basename(prev_dir)}")

    # refitting the challenger on all pre-eval data for deployment
    deploy_scaler = StandardScaler().fit(fit_df[feature_cols])
    fit_scaled = fit_df.copy()
    fit_scaled[feature_cols] = deploy_scaler.transform(fit_scaled[feature_cols])
    aX, ar, ac = build_sequences(fit_scaled, feature_cols)
    _, deploy_model = train_eval_seq(best_kind, "classification",
                                     aX, ar, ac, aX, ac, return_model=True,
                                     class_weights=class_weights)
    torch.save(deploy_model.state_dict(),
               os.path.join(MODEL_PATH, "seq_model.pt"))
    with open(os.path.join(MODEL_PATH, "seq_meta.pkl"), "wb") as f:
        pickle.dump({"kind": best_kind, "head": "classification",
                     "feature_cols": feature_cols, "window": SEQ_WINDOW,
                     "threshold": SEQ_THRESHOLD,
                     "n_features": len(feature_cols),
                     "classes": ["Down", "Neutral", "Up"],
                     "trained_through": str(eval_start)[:10]}, f)
    with open(os.path.join(MODEL_PATH, "seq_scaler.pkl"), "wb") as f:
        pickle.dump(deploy_scaler, f)
    log(f"challenger deployed ({best_kind}) — standard artifacts replaced")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=int, default=None,
                        help="limit tickers for a faster run")
    args = parser.parse_args()
    retrain(limit=args.tickers)


if __name__ == "__main__":
    main()