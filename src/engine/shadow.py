"""recording live predictions from every model so they compete over time"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import date, datetime, timezone
from core.config import MODEL_PATH
from inference.predictors import load_seq_predictor, load_rf_predictor
from inference.live_features import build_live_frame, fill_missing_features
from engine.memory import get_client, validate_ticker

_cache = {}


def _load_shadow_seq():
    # loading every trained shadow architecture from the shadow directory
    import torch
    from pipeline.sequence_models import make_torch_model
    out = {}
    shadow_dir = os.path.join(MODEL_PATH, "shadow")
    if not os.path.isdir(shadow_dir):
        return out
    for kind in sorted(os.listdir(shadow_dir)):
        kdir = os.path.join(shadow_dir, kind)
        try:
            with open(os.path.join(kdir, "seq_meta.pkl"), "rb") as f:
                meta = pickle.load(f)
            with open(os.path.join(kdir, "seq_scaler.pkl"), "rb") as f:
                scaler = pickle.load(f)
            model = make_torch_model(meta["kind"], meta["n_features"],
                                     meta["head"])
            model.load_state_dict(torch.load(
                os.path.join(kdir, "seq_model.pt"), map_location="cpu"))
            model.eval()
            out[kind] = {"meta": meta, "scaler": scaler, "model": model}
        except Exception:
            continue
    return out


def _load_shadow_xgb():
    # loading the xgboost shadow when a retrain has produced one
    path = os.path.join(MODEL_PATH, "shadow", "xgboost", "xgb.pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _models():
    # loading every deployed and shadow model once per process
    if not _cache:
        _cache["seq"] = load_seq_predictor()
        _cache["rf"] = load_rf_predictor()
        _cache["shadow_seq"] = _load_shadow_seq()
        _cache["xgb"] = _load_shadow_xgb()
    return _cache


def _seq_predict(bundle, df):
    # producing one direction call from any sequence bundle
    import torch
    meta = bundle["meta"]
    d = fill_missing_features(df.copy(), meta["feature_cols"],
                              bundle["scaler"])
    if len(d) < meta["window"]:
        return None
    win = bundle["scaler"].transform(
        d.iloc[-meta["window"]:][meta["feature_cols"]]
        .values.astype("float32"))
    with torch.no_grad():
        out = bundle["model"](torch.tensor(win).unsqueeze(0))             .numpy().squeeze()
    p = np.exp(out) / np.exp(out).sum()
    i = int(p.argmax())
    return meta["classes"][i], round(float(p[i]), 4)


def record_predictions(ticker):
    # writing today's call from each live model for later scoring
    import torch
    ticker = validate_ticker(ticker)
    m = _models()
    df = build_live_frame(ticker)
    if df is None or df.empty:
        return []
    rows = []

    seq = m.get("seq")
    if seq:
        pred = _seq_predict(seq, df)
        if pred:
            rows.append({"model": seq["meta"].get("kind", "cnn1d"),
                         "direction": pred[0], "confidence": pred[1]})

    # recording every shadow architecture on the same data
    deployed_kind = seq["meta"].get("kind") if seq else None
    for kind, bundle in m.get("shadow_seq", {}).items():
        if kind == deployed_kind:
            continue
        pred = _seq_predict(bundle, df)
        if pred:
            rows.append({"model": kind,
                         "direction": pred[0], "confidence": pred[1]})

    rf = m.get("rf")
    if rf:
        d = df.copy()
        for c in rf["feature_cols"]:
            if c not in d.columns:
                d[c] = np.nan
        latest = d.iloc[[-1]][rf["feature_cols"]]
        x = rf["scaler"].transform(rf["imputer"].transform(latest))
        p = rf["model"].predict_proba(x)[0]
        i = int(p.argmax())
        rows.append({"model": "random_forest",
                     "direction": str(rf["label_encoder"].classes_[i]),
                     "confidence": round(float(p[i]), 4)})

    xgb = m.get("xgb")
    if xgb:
        try:
            frame = df.copy()
            for c in xgb["feature_cols"]:
                if c not in frame.columns:
                    frame[c] = np.nan
            latest = xgb["scaler"].transform(
                frame.iloc[[-1]][xgb["feature_cols"]].fillna(0))
            p = xgb["model"].predict_proba(latest)[0]
            i = int(p.argmax())
            rows.append({"model": "xgboost",
                         "direction": xgb["classes"][i],
                         "confidence": round(float(p[i]), 4)})
        except Exception:
            pass

    # adding the free ensemble competitor voting across all models
    if len(rows) >= 3:
        from collections import Counter
        top, _ = Counter(r["direction"] for r in rows).most_common(1)[0]
        agree = [r["confidence"] for r in rows if r["direction"] == top]
        rows.append({"model": "ensemble", "direction": top,
                     "confidence": round(sum(agree) / len(agree), 4)})

    # constant-strategy baselines: any model worth keeping must beat
    # these; they are shadow-only and can never be elected champion
    for const in ("Down", "Up", "Neutral"):
        rows.append({"model": f"always_{const.lower()}",
                     "direction": const, "confidence": 0.3333})

    payload = [{"pred_date": str(date.today()), "ticker": ticker, **r}
               for r in rows]
    if payload:
        get_client().table("model_predictions").upsert(payload).execute()
    return payload


def score_model_predictions():
    # grading each shadow call against the first close after its date
    import yfinance as yf
    pending = get_client().table("model_predictions").select("*") \
        .is_("scored_at", "null").execute().data or []
    for r in pending:
        try:
            decided = pd.Timestamp(r["pred_date"])
            closes = yf.download(r["ticker"].replace(".", "-"), period="1mo",
                                 auto_adjust=True, progress=False)["Close"] \
                .squeeze()
            closes.index = pd.to_datetime(closes.index).tz_localize(None)
            before = closes[closes.index <= decided]
            after = closes[closes.index > decided]
            if before.empty or after.empty:
                continue
            ret = float(after.iloc[0]) / float(before.iloc[-1]) - 1
            label = ("Up" if ret > 0.01
                     else "Down" if ret < -0.01 else "Neutral")
            get_client().table("model_predictions").update(
                {"outcome_label": label,
                 "was_correct": r["direction"] == label,
                 "scored_at": datetime.now(timezone.utc).isoformat()}) \
                .eq("pred_date", r["pred_date"]) \
                .eq("ticker", r["ticker"]) \
                .eq("model", r["model"]).execute()
        except Exception:
            continue


def model_report(window=300):
    # comparing hit rates across every model on identical tickers and days
    rows = get_client().table("model_predictions") \
        .select("model,was_correct").not_.is_("scored_at", "null") \
        .order("pred_date", desc=True).limit(int(window)).execute().data or []
    if not rows:
        print("model comparison: no scored shadow predictions yet")
        return
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        agg[r["model"]][1] += 1
        if r["was_correct"]:
            agg[r["model"]][0] += 1
    print("model comparison (same tickers, same days):")
    for m, (hits, n) in sorted(agg.items()):
        print(f"  {m:16s}: {hits}/{n} correct ({100 * hits / n:.0f}%)")
