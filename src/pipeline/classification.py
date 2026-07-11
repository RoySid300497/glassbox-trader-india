"""training the baseline random forest and xgboost classifiers"""

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from core.config import DATA_PATH, MODEL_PATH, BASE_FEATURE_COLS, RANDOM_STATE
from core.helpers import log, save_plot, section
from pipeline.prep import prep_xy


def stage_3_classify():
    section("STAGE 3 — BASELINE CLASSIFICATION (POOLED, NEXT-DAY)")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 ConfusionMatrixDisplay, f1_score)
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier

    # loading the pre-split data saved by stage 2
    train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"),
                        parse_dates=["date"])
    val = pd.read_csv(os.path.join(DATA_PATH, "val.csv"), parse_dates=["date"])
    test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"),
                       parse_dates=["date"])
    Xtr, Xva, Xte, ytr, yva, yte, imp, sc, le = prep_xy(
        train, val, test, BASE_FEATURE_COLS, "label")

    # training a class-weighted random forest baseline
    log("training random forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(Xtr, ytr)
    rf_pred = rf.predict(Xva)
    log("random forest — validation:\n" +
        classification_report(yva, rf_pred, target_names=le.classes_))

    # training a sample-weighted xgboost
    log("training xgboost...")
    sw = compute_sample_weight("balanced", ytr)
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                        eval_metric="mlogloss", random_state=RANDOM_STATE,
                        n_jobs=-1)
    xgb.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=False)
    xgb_pred = xgb.predict(Xva)
    log("xgboost — validation:\n" +
        classification_report(yva, xgb_pred, target_names=le.classes_))

    # plotting confusion matrices and feature importance
    for name, pred, cmap in [("rf", rf_pred, "Blues"),
                             ("xgb", xgb_pred, "Oranges")]:
        cm = confusion_matrix(yva, pred)
        ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
            cmap=cmap, colorbar=False)
        plt.title(f"{name} — confusion matrix (val)")
        save_plot(f"s3_{name}_confusion_val.png")
    fi = pd.DataFrame({"feature": BASE_FEATURE_COLS,
                       "importance": rf.feature_importances_}) \
        .sort_values("importance", ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=fi, hue="feature",
                palette="Blues_r", legend=False)
    plt.title("random forest — top feature importances")
    save_plot("s3_rf_feature_importance.png")

    # evaluating the better model once on the held-out test set
    best = xgb if f1_score(yva, xgb_pred, average="macro") >= \
        f1_score(yva, rf_pred, average="macro") else rf
    best_name = "xgb" if best is xgb else "rf"
    test_pred = best.predict(Xte)
    log(f"best baseline model: {best_name}")
    log(f"{best_name} — TEST:\n" +
        classification_report(yte, test_pred, target_names=le.classes_))

    # saving every artifact the chatbot and later stages need
    for obj, fn in [(rf, "rf_model.pkl"), (xgb, "xgb_model.pkl"),
                    (sc, "scaler.pkl"), (imp, "imputer.pkl"),
                    (le, "label_encoder.pkl"),
                    (BASE_FEATURE_COLS, "feature_cols.pkl")]:
        with open(os.path.join(MODEL_PATH, fn), "wb") as f:
            pickle.dump(obj, f)
    log("saved baseline models")
    log("stage 3 complete")
