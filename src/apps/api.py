"""serving live predictions from the saved models over HTTP"""

import numpy as np
import pandas as pd
import warnings
from fastapi import FastAPI, HTTPException
from inference.predictors import load_seq_predictor, load_rf_predictor
from inference.live_features import build_live_frame, fill_missing_features

warnings.filterwarnings("ignore",
                        message="X does not have valid feature names")

app = FastAPI(title="stock-lens signal API",
              description="Direction predictions from the trained models "
                          "on live yfinance data. Educational only.")

# loading both models once at startup so requests stay fast
seq_pred = load_seq_predictor()
rf_pred = load_rf_predictor()


@app.get("/health")
def health():
    # reporting which models are loaded and ready
    return {"status": "ok",
            "sequence_model": seq_pred is not None,
            "random_forest": rf_pred is not None}


@app.get("/predict")
def predict(ticker: str, model: str = "sequence"):
    # returning a direction prediction for one ticker on the latest live data
    ticker = ticker.upper().strip()

    # validating the model choice before any network fetch
    if model == "sequence" and seq_pred is None:
        raise HTTPException(422, "sequence model not loaded")
    if model == "random_forest" and rf_pred is None:
        raise HTTPException(422, "random forest not loaded")
    if model not in ("sequence", "random_forest"):
        raise HTTPException(422, f"model '{model}' not available — "
                                 f"use 'sequence' or 'random_forest'")

    df = build_live_frame(ticker)
    if df is None or df.empty:
        raise HTTPException(404, f"no live data found for {ticker}")

    if model == "sequence" and seq_pred is not None:
        import torch
        meta = seq_pred["meta"]
        df = fill_missing_features(df, meta["feature_cols"],
                                   seq_pred["scaler"])
        if len(df) < meta["window"]:
            raise HTTPException(422, f"need {meta['window']} days of history")
        win = df.iloc[-meta["window"]:][meta["feature_cols"]] \
            .values.astype("float32")
        win = seq_pred["scaler"].transform(win)
        with torch.no_grad():
            out = seq_pred["model"](
                torch.tensor(win).unsqueeze(0)).numpy().squeeze()
        probs = np.exp(out) / np.exp(out).sum()
        idx = int(probs.argmax())
        return {"ticker": ticker,
                "as_of": str(df["date"].max().date()),
                "model": f"sequence ({meta['kind']})",
                "direction": meta["classes"][idx],
                "confidence": round(float(probs[idx]), 4),
                "close": round(float(df["close"].iloc[-1]), 2),
                "rsi": round(float(df["rsi"].iloc[-1]), 1),
                "disclaimer": "educational output, not financial advice"}

    if model == "random_forest" and rf_pred is not None:
        fcols = rf_pred["feature_cols"]
        for c in fcols:
            if c not in df.columns:
                df[c] = np.nan
        latest = df.iloc[[-1]][fcols]
        x = rf_pred["scaler"].transform(rf_pred["imputer"].transform(latest))
        probs = rf_pred["model"].predict_proba(x)[0]
        idx = int(probs.argmax())
        return {"ticker": ticker,
                "as_of": str(df["date"].max().date()),
                "model": "random_forest",
                "direction": rf_pred["label_encoder"].classes_[idx],
                "confidence": round(float(probs[idx]), 4),
                "close": round(float(df["close"].iloc[-1]), 2),
                "rsi": round(float(df["rsi"].iloc[-1]), 1),
                "disclaimer": "educational output, not financial advice"}

    raise HTTPException(500, "prediction failed unexpectedly")
