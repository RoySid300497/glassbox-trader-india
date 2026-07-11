"""preparing model inputs and running leak-safe held-out test evaluations"""

import os
import numpy as np
import matplotlib.pyplot as plt
from core.config import VAL_END, OBS_PATH
from core.helpers import log, save_plot


def prep_xy(train, val, test, feature_cols, label_col):
    # imputing, scaling, and encoding using training statistics only
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(train[feature_cols])
    Xva = imp.transform(val[feature_cols])
    Xte = imp.transform(test[feature_cols])
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xva = sc.transform(Xva)
    Xte = sc.transform(Xte)
    le = LabelEncoder()
    ytr = le.fit_transform(train[label_col])
    yva = le.transform(val[label_col])
    yte = le.transform(test[label_col])
    return Xtr, Xva, Xte, ytr, yva, yte, imp, sc, le


def final_test_eval(df, feature_cols, label_col, build_model, use_weight, tag):
    # refitting the chosen model on train+val and scoring the test set once
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 ConfusionMatrixDisplay, f1_score)
    from sklearn.utils.class_weight import compute_sample_weight

    trainval = df[df["date"] < VAL_END]
    test = df[df["date"] >= VAL_END]

    # fitting preprocessing on train+val only to prevent leakage
    imp = SimpleImputer(strategy="median")
    Xtv = imp.fit_transform(trainval[feature_cols])
    Xte = imp.transform(test[feature_cols])
    sc = StandardScaler()
    Xtv = sc.fit_transform(Xtv)
    Xte = sc.transform(Xte)
    le = LabelEncoder()
    ytv = le.fit_transform(trainval[label_col])
    yte = le.transform(test[label_col])

    # fitting with balanced sample weights when the model supports them
    model = build_model()
    if use_weight and model.__class__.__name__ == "XGBClassifier":
        model.fit(Xtv, ytv, sample_weight=compute_sample_weight("balanced", ytv))
    else:
        model.fit(Xtv, ytv)

    # reporting per-class and macro results on the held-out test set
    pred = model.predict(Xte)
    macro = f1_score(yte, pred, average="macro")
    log(f"\n[TEST] {tag} — refit on train+val, held-out test:")
    log(classification_report(yte, pred, target_names=le.classes_))
    log(f"[TEST] {tag} — macro_f1 = {macro:.4f}")
    cm = confusion_matrix(yte, pred)
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
        cmap="Purples", colorbar=False)
    plt.title(f"{tag} — confusion matrix (held-out test)")
    save_plot(f"TEST_{tag}.png")
    per_class = f1_score(yte, pred, average=None)
    return macro, dict(zip(le.classes_, per_class))
