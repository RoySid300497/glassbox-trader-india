"""scanning the whole universe with the cnn and ranking debate candidates"""

import numpy as np
from datetime import date, timedelta
from core.helpers import log, section
from pipeline.oos_evaluation import prepare_oos_frame
from inference.predictors import load_seq_predictor, load_rf_predictor
from engine.champion import get_champion

SCAN_LOOKBACK_DAYS = 240     # downloading enough history for indicator warmup
FALLBACK = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM"]


def _rf_scan(df):
    # scoring the universe with the random forest on the latest row per ticker
    rf = load_rf_predictor()
    if rf is None:
        return None
    latest = df.sort_values("date").groupby("symbol", sort=False).tail(1)
    frame = latest.copy()
    for c in rf["feature_cols"]:
        if c not in frame.columns:
            frame[c] = np.nan
    X = rf["scaler"].transform(rf["imputer"].transform(
        frame[rf["feature_cols"]]))
    probs = rf["model"].predict_proba(X)
    classes = list(rf["label_encoder"].classes_)
    results = []
    for (_, row), p in zip(frame.iterrows(), probs):
        idx = int(p.argmax())
        results.append(({"ticker": row["symbol"],
                         "close": round(float(row["close"]), 2),
                         "return_1d": float(row.get("return_1d") or 0),
                         "vol_ratio": float(row.get("vol_ratio") or 1),
                         "rsi": round(float(row.get("rsi") or 50), 1)},
                        str(classes[idx]), float(p[idx])))
    return results


def scan_universe(limit=None):
    # scanning every ticker with the elected champion and scoring interest
    import torch
    champion = "cnn1d"
    try:
        champion = get_champion()
    except Exception:
        pass
    seq = load_seq_predictor()
    if seq is None:
        log("screener: no sequence model loaded")
        return []
    meta = seq["meta"]
    window, fcols = meta["window"], meta["feature_cols"]

    start = date.today() - timedelta(days=SCAN_LOOKBACK_DAYS)
    df = prepare_oos_frame(limit=limit, start=str(start),
                           end=str(date.today() + timedelta(days=1)))
    if df is None or df.empty:
        log("screener: universe download failed")
        return []

    # routing to the random forest when it holds the elected title
    if champion == "random_forest":
        rf_out = _rf_scan(df)
        if rf_out is not None:
            results = []
            for info, direction, conf in rf_out:
                directional = 0.0 if direction == "Neutral" else (conf - 1 / 3)
                move = min(abs(info["return_1d"]) * 10, 0.5)
                volume = min(max(info["vol_ratio"] - 1, 0) * 0.2, 0.4)
                results.append({**info, "direction": direction,
                                "confidence": round(conf, 4),
                                "score": round(directional + move + volume, 4)})
            results.sort(key=lambda r: r["score"], reverse=True)
            log(f"screener[rf]: scanned {len(results)} tickers")
            return results

    # filling any dataset-only columns with scaler means so they scale to zero
    means = dict(zip(fcols, seq["scaler"].mean_))
    for c in fcols:
        df[c] = df[c].fillna(means[c]) if c in df.columns else means[c]

    # building one latest window per ticker for batch inference
    windows, info = [], []
    for tk, grp in df.groupby("symbol", sort=False):
        grp = grp.sort_values("date")
        if len(grp) < window + 1:
            continue
        feats = seq["scaler"].transform(grp[fcols]).astype("float32")
        windows.append(feats[-window:])
        last = grp.iloc[-1]
        info.append({"ticker": tk,
                     "close": round(float(last["close"]), 2),
                     "return_1d": float(last.get("return_1d") or 0),
                     "vol_ratio": float(last.get("vol_ratio") or 1),
                     "rsi": round(float(last.get("rsi") or 50), 1)})
    if not windows:
        return []

    X = np.stack(windows)
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), 2048):
            outs.append(seq["model"](torch.tensor(X[i:i + 2048])).numpy())
    out = np.concatenate(outs)
    probs = np.exp(out) / np.exp(out).sum(axis=1, keepdims=True)

    # scoring interest from conviction, abnormal move, and abnormal volume
    results = []
    for row, p in zip(info, probs):
        idx = int(p.argmax())
        direction = meta["classes"][idx]
        conf = float(p[idx])
        directional = 0.0 if direction == "Neutral" else (conf - 1 / 3)
        move = min(abs(row["return_1d"]) * 10, 0.5)
        volume = min(max(row["vol_ratio"] - 1, 0) * 0.2, 0.4)
        results.append({**row, "direction": direction,
                        "confidence": round(conf, 4),
                        "score": round(directional + move + volume, 4)})

    results.sort(key=lambda r: r["score"], reverse=True)
    log(f"screener: scanned {len(results)} tickers, "
        f"top: {[r['ticker'] for r in results[:5]]}")
    return results


def select_watchlist(k=5, limit=None, exclude=None):
    # picking top names plus exploration wildcards from the quiet middle
    import os
    import random
    from datetime import date as _date
    exclude = exclude or set()
    results = scan_universe(limit=limit)
    if not results:
        log("screener: falling back to the core watchlist")
        return FALLBACK, []
    eligible = [r["ticker"] for r in results if r["ticker"] not in exclude]

    slots = max(0, min(int(os.environ.get("EXPLORE_SLOTS") or "2"), int(k) - 1))
    top = eligible[:int(k) - slots]

    # sampling wildcards from mid-ranked names to counter momentum bias
    pool = [t for t in eligible[max(20, int(k)):250] if t not in top]
    random.seed(str(_date.today()))
    wild = random.sample(pool, min(slots, len(pool))) if pool else []
    if wild:
        log(f"screener: exploration wildcards {wild}")
    return top + wild, results
