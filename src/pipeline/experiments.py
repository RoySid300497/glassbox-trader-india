"""running the strategy x horizon x granularity experiment with test eval"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from core.config import (DATA_PATH, OBS_PATH, BASE_FEATURE_COLS, LAG_COLS,
                    LAG_DAYS, RETURN_FEATURES, RELATIVE_FEATURES,
                    HORIZONS, TRAIN_END, VAL_END, RANDOM_STATE)
from core.helpers import log, save_plot, section
from pipeline.prep import prep_xy, final_test_eval


def make_models():
    # building the five modelling strategies as fresh estimators
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from xgboost import XGBClassifier
    rf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                        eval_metric="mlogloss", random_state=RANDOM_STATE,
                        n_jobs=-1)
    logreg = LogisticRegression(max_iter=500, class_weight="balanced",
                                n_jobs=-1)
    stacked = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        final_estimator=LogisticRegression(max_iter=500,
                                           class_weight="balanced"),
        n_jobs=-1)
    ovr = OneVsRestClassifier(
        XGBClassifier(n_estimators=150, max_depth=5, eval_metric="logloss",
                      random_state=RANDOM_STATE), n_jobs=-1)
    return {"random_forest": rf, "xgboost": xgb,
            "logistic_regression": logreg, "stacked_ensemble": stacked,
            "one_vs_rest": ovr}


def fit_score(model, Xtr, ytr, Xva, yva, use_weight=False):
    # fitting one model and returning validation macro-f1 and accuracy
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.utils.class_weight import compute_sample_weight
    try:
        if use_weight and model.__class__.__name__ == "XGBClassifier":
            model.fit(Xtr, ytr,
                      sample_weight=compute_sample_weight("balanced", ytr))
        else:
            model.fit(Xtr, ytr)
        pred = model.predict(Xva)
        return (f1_score(yva, pred, average="macro"),
                accuracy_score(yva, pred))
    except Exception as e:
        log(f"    [skip] {model.__class__.__name__}: {e}")
        return (np.nan, np.nan)


def enhanced_feature_cols(df):
    # assembling the full enhanced feature list present in the dataframe
    cols = BASE_FEATURE_COLS + \
        [f"{c}_lag{l}" for c in LAG_COLS for l in LAG_DAYS] + \
        RETURN_FEATURES + RELATIVE_FEATURES
    return [c for c in cols if c in df.columns]


def stage_3b_experiments():
    section("STAGE 3B — FULL EXPERIMENT (5 STRATEGIES x 3 HORIZONS)")
    df = pd.read_csv(os.path.join(DATA_PATH, "master_enhanced.csv"),
                     parse_dates=["date"])
    feature_cols = enhanced_feature_cols(df)
    log(f"using {len(feature_cols)} enhanced features")
    results = []

    # sweeping pooled and per-sector configurations for every horizon
    for horizon in HORIZONS:
        label_col = f"label_{horizon}"
        section(f"HORIZON {horizon} — POOLED")
        train = df[df["date"] < TRAIN_END]
        val = df[(df["date"] >= TRAIN_END) & (df["date"] < VAL_END)]
        Xtr, Xva, _, ytr, yva, _, _, _, _ = prep_xy(train, val, val,
                                                    feature_cols, label_col)
        for strat, model in make_models().items():
            f1, acc = fit_score(model, Xtr, ytr, Xva, yva, use_weight=True)
            log(f"  pooled | {strat:20s} | macro_f1={f1:.4f} acc={acc:.4f}")
            results.append({"strategy": strat, "horizon": horizon,
                            "granularity": "pooled", "macro_f1": f1})

        section(f"HORIZON {horizon} — PER-SECTOR")
        for strat in make_models():
            f1s = []
            for sector in df["sector"].dropna().unique():
                sub = df[df["sector"] == sector]
                s_tr = sub[sub["date"] < TRAIN_END]
                s_va = sub[(sub["date"] >= TRAIN_END) &
                           (sub["date"] < VAL_END)]
                if len(s_tr) < 500 or len(s_va) < 100:
                    continue
                Xtr, Xva, _, ytr, yva, _, _, _, _ = prep_xy(
                    s_tr, s_va, s_va, feature_cols, label_col)
                f1, _ = fit_score(make_models()[strat], Xtr, ytr, Xva, yva,
                                  use_weight=True)
                if not np.isnan(f1):
                    f1s.append(f1)
            avg = float(np.mean(f1s)) if f1s else np.nan
            log(f"  per-sector | {strat:20s} | avg macro_f1={avg:.4f}")
            results.append({"strategy": strat, "horizon": horizon,
                            "granularity": "per_sector", "macro_f1": avg})

    # sweeping per-stock with xgboost only across every horizon
    section("PER-STOCK (xgboost only)")
    from xgboost import XGBClassifier
    from sklearn.metrics import f1_score
    for horizon in HORIZONS:
        label_col = f"label_{horizon}"
        f1s = []
        for tk in df["symbol"].unique():
            sub = df[df["symbol"] == tk]
            s_tr = sub[sub["date"] < TRAIN_END]
            s_va = sub[(sub["date"] >= TRAIN_END) & (sub["date"] < VAL_END)]
            if len(s_tr) < 300 or len(s_va) < 50:
                continue
            try:
                Xtr, Xva, _, ytr, yva, _, _, _, _ = prep_xy(
                    s_tr, s_va, s_va, feature_cols, label_col)
                m = XGBClassifier(n_estimators=100, max_depth=4,
                                  learning_rate=0.1, eval_metric="mlogloss",
                                  random_state=RANDOM_STATE)
                m.fit(Xtr, ytr)
                f1s.append(f1_score(yva, m.predict(Xva), average="macro"))
            except Exception:
                continue
        avg = float(np.mean(f1s)) if f1s else np.nan
        log(f"  per-stock | xgboost | {horizon} | avg macro_f1={avg:.4f}")
        results.append({"strategy": "xgboost", "horizon": horizon,
                        "granularity": "per_stock", "macro_f1": avg})

    # saving the results table and comparison plots
    res = pd.DataFrame(results)
    res.to_csv(os.path.join(OBS_PATH, "s3b_experiment_results.csv"),
               index=False)
    log("\nFULL EXPERIMENT RESULTS:\n" + res.to_string(index=False))
    pooled = res[res["granularity"] == "pooled"]
    plt.figure(figsize=(12, 5))
    sns.barplot(data=pooled, x="strategy", y="macro_f1", hue="horizon")
    plt.axhline(0.333, color="red", linestyle="--")
    plt.title("pooled — macro F1 by strategy and horizon")
    plt.xticks(rotation=30)
    save_plot("s3b_pooled_macro_f1.png")
    plt.figure(figsize=(9, 5))
    sns.barplot(data=res, x="horizon", y="macro_f1", hue="granularity")
    plt.axhline(0.333, color="red", linestyle="--")
    plt.title("macro F1 by horizon and granularity")
    save_plot("s3b_horizon_effect.png")

    # evaluating the validation winner once on the held-out test set
    section("HELD-OUT TEST EVALUATION (winner selected on validation only)")
    pooled_valid = pooled.dropna(subset=["macro_f1"])
    winner = pooled_valid.loc[pooled_valid["macro_f1"].idxmax()]
    strat, horizon = winner["strategy"], winner["horizon"]
    log(f"validation winner: {strat} @ {horizon} "
        f"(val macro_f1={winner['macro_f1']:.4f})")
    final_test_eval(df, feature_cols, f"label_{horizon}",
                    lambda: make_models()[strat], use_weight=True,
                    tag=f"{strat}_{horizon}")
    log("stage 3b complete")
    return res
