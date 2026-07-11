"""training LSTM, GRU, TCN, CNN, and transformer sequence models"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from core.config import (DATA_PATH, MODEL_PATH, OBS_PATH, SEQ_WINDOW, SEQ_EPOCHS,
                    SEQ_BATCH, SEQ_THRESHOLD, PER_STOCK_SAMPLE,
                    TRAIN_END, VAL_END)
from core.helpers import log, save_plot, section
from pipeline.experiments import enhanced_feature_cols


def build_sequences(frame, feature_cols, window=SEQ_WINDOW):
    # building sliding windows of shape (samples, window, features) per ticker
    X, y_reg, y_cls = [], [], []
    for _, grp in frame.groupby("symbol", sort=False):
        grp = grp.sort_values("date")
        feats = grp[feature_cols].values.astype("float32")
        closes = grp["close"].values.astype("float32")
        for i in range(window, len(grp) - 1):
            X.append(feats[i - window:i])
            fwd = (closes[i + 1] - closes[i]) / closes[i]
            y_reg.append(fwd)
            y_cls.append(0 if fwd < -SEQ_THRESHOLD
                         else (2 if fwd > SEQ_THRESHOLD else 1))
    if not X:
        return None, None, None
    return (np.asarray(X, dtype="float32"),
            np.asarray(y_reg, dtype="float32"),
            np.asarray(y_cls, dtype="int64"))


def make_torch_model(kind, n_features, head, window=SEQ_WINDOW):
    # constructing one architecture with the requested output head
    import torch.nn as nn
    out_dim = 1 if head == "regression" else 3

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(n_features, 64, num_layers=2,
                               batch_first=True, dropout=0.2)
            self.fc = nn.Linear(64, out_dim)

        def forward(self, x):
            o, _ = self.rnn(x)
            return self.fc(o[:, -1, :])

    class GRUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(n_features, 64, num_layers=2,
                              batch_first=True, dropout=0.2)
            self.fc = nn.Linear(64, out_dim)

        def forward(self, x):
            o, _ = self.rnn(x)
            return self.fc(o[:, -1, :])

    class CNN1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(n_features, 64, 3, padding=1), nn.ReLU(),
                nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1))
            self.fc = nn.Linear(64, out_dim)

        def forward(self, x):
            return self.fc(self.net(x.transpose(1, 2)).squeeze(-1))

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            layers, ch = [], n_features
            for d in (1, 2, 4, 8):
                layers += [nn.Conv1d(ch, 64, 3, padding=d, dilation=d),
                           nn.ReLU()]
                ch = 64
            self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1))
            self.fc = nn.Linear(64, out_dim)

        def forward(self, x):
            return self.fc(self.net(x.transpose(1, 2)).squeeze(-1))

    class TransformerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(n_features, 64)
            enc = nn.TransformerEncoderLayer(d_model=64, nhead=4,
                                             dim_feedforward=128,
                                             batch_first=True, dropout=0.2)
            self.enc = nn.TransformerEncoder(enc, num_layers=2)
            self.fc = nn.Linear(64, out_dim)

        def forward(self, x):
            return self.fc(self.enc(self.proj(x))[:, -1, :])

    return {"lstm": LSTMNet, "gru": GRUNet, "cnn1d": CNN1D,
            "tcn": TCN, "transformer": TransformerNet}[kind]()


def train_eval_seq(kind, head, Xtr, ytr_reg, ytr_cls, Xva, yva_cls,
                   return_model=False, class_weights=None):
    # training one architecture and returning validation macro-f1
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_torch_model(kind, Xtr.shape[2], head).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    weights = None
    if class_weights is not None and head == "classification":
        weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    loss_fn = nn.MSELoss() if head == "regression" \
        else nn.CrossEntropyLoss(weight=weights)
    Xtr_t = torch.tensor(Xtr)
    ytr_t = torch.tensor(ytr_reg if head == "regression" else ytr_cls)
    if head == "regression":
        ytr_t = ytr_t.unsqueeze(1)

    # looping over shuffled mini-batches for the configured epochs
    n = len(Xtr_t)
    for _ in range(SEQ_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, SEQ_BATCH):
            idx = perm[i:i + SEQ_BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx].to(device)),
                           ytr_t[idx].to(device))
            loss.backward()
            opt.step()

    f1, _ = score_seq_model(model, head, Xva, yva_cls)
    if return_model:
        return f1, model
    return f1


def score_seq_model(model, head, X, y_cls):
    # scoring a trained sequence model on any evaluation set
    import torch
    from sklearn.metrics import f1_score
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            out = model(torch.tensor(X[i:i + 4096]).to(device)).cpu().numpy()
            if head == "regression":
                preds.append(np.where(out.squeeze() < -SEQ_THRESHOLD, 0,
                             np.where(out.squeeze() > SEQ_THRESHOLD, 2, 1)))
            else:
                preds.append(out.argmax(axis=1))
    preds = np.concatenate(preds)
    return f1_score(y_cls, preds, average="macro"), preds


def stage_4_sequence():
    section("STAGE 4 — SEQUENCE MODELS")
    import torch
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    log(f"cuda available: {torch.cuda.is_available()}")

    df = pd.read_csv(os.path.join(DATA_PATH, "master_enhanced.csv"),
                     parse_dates=["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    feature_cols = enhanced_feature_cols(df)
    architectures = ["lstm", "gru", "tcn", "cnn1d", "transformer"]
    heads = ["regression", "classification"]
    results = []

    # splitting three ways and scaling on training rows only
    section("POOLED SEQUENCES (train/val/test)")
    train = df[df["date"] < TRAIN_END].copy()
    val = df[(df["date"] >= TRAIN_END) & (df["date"] < VAL_END)].copy()
    test = df[df["date"] >= VAL_END].copy()
    scaler = StandardScaler().fit(train[feature_cols])
    for part in (train, val, test):
        part[feature_cols] = scaler.transform(part[feature_cols])

    Xtr, ytr_reg, ytr_cls = build_sequences(train, feature_cols)
    Xva, _, yva_cls = build_sequences(val, feature_cols)
    Xte, _, yte_cls = build_sequences(test, feature_cols)
    log(f"train seq: {Xtr.shape} | val: {Xva.shape} | test: {Xte.shape}")

    # training every architecture and head, selecting on validation
    trained = {}
    for kind in architectures:
        for head in heads:
            try:
                f1, model = train_eval_seq(kind, head, Xtr, ytr_reg, ytr_cls,
                                           Xva, yva_cls, return_model=True)
                log(f"  pooled | {kind:11s} | {head:14s} | VAL f1={f1:.4f}")
                results.append({"architecture": kind, "head": head,
                                "granularity": "pooled", "split": "val",
                                "macro_f1": f1})
                trained[(kind, head)] = model
            except Exception as e:
                log(f"  [skip] {kind}/{head}: {e}")

    # scoring the validation winner once on the held-out test set
    val_rows = [r for r in results if r["split"] == "val"
                and r["granularity"] == "pooled"]
    best = max(val_rows, key=lambda r: r["macro_f1"])
    bk, bh = best["architecture"], best["head"]
    log(f"\nvalidation winner: {bk}/{bh} (val f1={best['macro_f1']:.4f})")
    test_f1, test_preds = score_seq_model(trained[(bk, bh)], bh, Xte, yte_cls)
    log(f"[TEST] {bk}/{bh} — held-out test macro_f1 = {test_f1:.4f}")
    log(classification_report(yte_cls, test_preds,
                              target_names=["Down", "Neutral", "Up"]))
    results.append({"architecture": bk, "head": bh, "granularity": "pooled",
                    "split": "test", "macro_f1": test_f1})

    # refitting the winner on train+val and saving it for the chatbot
    log("refitting winner on train+val for deployment...")
    deploy_scaler = StandardScaler().fit(
        df[df["date"] < VAL_END][feature_cols])
    tv = df[df["date"] < VAL_END].copy()
    tv[feature_cols] = deploy_scaler.transform(tv[feature_cols])
    aX, ar, ac = build_sequences(tv, feature_cols)
    _, deploy_model = train_eval_seq(bk, bh, aX, ar, ac, aX, ac,
                                     return_model=True)
    torch.save(deploy_model.state_dict(),
               os.path.join(MODEL_PATH, "seq_model.pt"))
    with open(os.path.join(MODEL_PATH, "seq_meta.pkl"), "wb") as f:
        pickle.dump({"kind": bk, "head": bh, "feature_cols": feature_cols,
                     "window": SEQ_WINDOW, "threshold": SEQ_THRESHOLD,
                     "n_features": len(feature_cols),
                     "classes": ["Down", "Neutral", "Up"]}, f)
    with open(os.path.join(MODEL_PATH, "seq_scaler.pkl"), "wb") as f:
        pickle.dump(deploy_scaler, f)
    log("saved seq_model.pt, seq_meta.pkl, seq_scaler.pkl")

    # sweeping a per-stock sample as a granularity comparison
    section(f"PER-STOCK SEQUENCES (sample of {PER_STOCK_SAMPLE})")
    sample = df["symbol"].value_counts().head(PER_STOCK_SAMPLE).index
    for kind in architectures:
        for head in heads:
            f1s = []
            for tk in sample:
                sub = df[df["symbol"] == tk].copy()
                s_tr = sub[sub["date"] < TRAIN_END]
                s_va = sub[(sub["date"] >= TRAIN_END) &
                           (sub["date"] < VAL_END)]
                if len(s_tr) < SEQ_WINDOW + 60 or len(s_va) < SEQ_WINDOW + 10:
                    continue
                sc = StandardScaler().fit(s_tr[feature_cols])
                s_tr = s_tr.copy(); s_va = s_va.copy()
                s_tr[feature_cols] = sc.transform(s_tr[feature_cols])
                s_va[feature_cols] = sc.transform(s_va[feature_cols])
                a, b, c = build_sequences(s_tr, feature_cols)
                d_, _, f_ = build_sequences(s_va, feature_cols)
                if a is None or d_ is None:
                    continue
                try:
                    f1s.append(train_eval_seq(kind, head, a, b, c, d_, f_))
                except Exception:
                    continue
            avg = float(np.mean(f1s)) if f1s else np.nan
            log(f"  per-stock | {kind:11s} | {head:14s} | avg f1={avg:.4f}")
            results.append({"architecture": kind, "head": head,
                            "granularity": "per_stock", "split": "val",
                            "macro_f1": avg})

    # saving results and comparison plots
    res = pd.DataFrame(results)
    res.to_csv(os.path.join(OBS_PATH, "s4_sequence_results.csv"), index=False)
    log("\nSEQUENCE MODEL RESULTS:\n" + res.to_string(index=False))
    pooled = res[(res["granularity"] == "pooled") & (res["split"] == "val")]
    plt.figure(figsize=(11, 5))
    sns.barplot(data=pooled, x="architecture", y="macro_f1", hue="head")
    plt.axhline(0.39, color="green", linestyle="--", label="best tabular")
    plt.title("pooled sequence models — VAL macro F1")
    plt.legend()
    save_plot("s4_pooled_sequence_macro_f1.png")
    plt.figure(figsize=(11, 5))
    sns.barplot(data=res[res["split"] == "val"], x="architecture",
                y="macro_f1", hue="granularity")
    plt.title("sequence models — pooled vs per-stock")
    save_plot("s4_granularity_comparison.png")
    log("stage 4 complete")
    return res
