"""loading saved models and producing predictions with explanations"""

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
import os
import pickle
import numpy as np
import pandas as pd
from core.config import MODEL_PATH
from core.helpers import log


def load_seq_predictor():
    # loading the saved sequence model, failing gracefully without torch
    meta_path = os.path.join(MODEL_PATH, "seq_meta.pkl")
    if not os.path.exists(meta_path):
        return None
    try:
        import torch
        from pipeline.sequence_models import make_torch_model
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        with open(os.path.join(MODEL_PATH, "seq_scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        model = make_torch_model(meta["kind"], meta["n_features"],
                                 meta["head"], window=meta["window"])
        model.load_state_dict(
            torch.load(os.path.join(MODEL_PATH, "seq_model.pt"),
                       map_location="cpu"))
        model.eval()
        return {"model": model, "scaler": scaler, "meta": meta}
    except Exception as e:
        log(f"sequence model unavailable ({e})")
        return None


def load_rf_predictor():
    # loading the saved random forest with its preprocessing objects
    if not os.path.exists(os.path.join(MODEL_PATH, "rf_model.pkl")):
        return None
    out = {}
    for key, fn in [("model", "rf_model.pkl"), ("scaler", "scaler.pkl"),
                    ("imputer", "imputer.pkl"),
                    ("label_encoder", "label_encoder.pkl"),
                    ("feature_cols", "feature_cols.pkl")]:
        with open(os.path.join(MODEL_PATH, fn), "rb") as f:
            out[key] = pickle.load(f)
    return out


def predict_one(ticker, date, model_choice, seq_pred, rf_pred, data):
    # producing a prediction dict for one ticker and date
    sub = data[data["symbol"] == ticker].sort_values("date")
    if sub.empty:
        return {"error": f"ticker {ticker} not found"}
    upto = sub[sub["date"] <= pd.Timestamp(date)]
    if upto.empty:
        return {"error": f"no data for {ticker} on or before {date}"}

    if model_choice == "sequence" and seq_pred is not None:
        import torch
        meta = seq_pred["meta"]
        if len(upto) < meta["window"]:
            return {"error": f"need {meta['window']} days of history"}
        win = upto.iloc[-meta["window"]:][meta["feature_cols"]] \
            .values.astype("float32")
        win = seq_pred["scaler"].transform(win)
        with torch.no_grad():
            out = seq_pred["model"](torch.tensor(win).unsqueeze(0)) \
                .numpy().squeeze()
        probs = np.exp(out) / np.exp(out).sum()
        idx = int(probs.argmax())
        return {"ticker": ticker, "date": str(pd.Timestamp(date).date()),
                "model": f"sequence ({meta['kind']})",
                "prediction": meta["classes"][idx],
                "confidence": float(probs[idx]),
                "recent": upto.iloc[-1][["close", "rsi",
                                         "MACD_12_26_9"]].to_dict()}

    if model_choice == "random_forest" and rf_pred is not None:
        latest = upto.iloc[[-1]][rf_pred["feature_cols"]]
        x = rf_pred["scaler"].transform(rf_pred["imputer"].transform(latest))
        probs = rf_pred["model"].predict_proba(x)[0]
        idx = int(probs.argmax())
        return {"ticker": ticker, "date": str(pd.Timestamp(date).date()),
                "model": "random_forest",
                "prediction": rf_pred["label_encoder"].classes_[idx],
                "confidence": float(probs[idx]),
                "recent": upto.iloc[-1][["close", "rsi",
                                         "MACD_12_26_9"]].to_dict()}

    return {"error": f"model '{model_choice}' not available"}


def explain_with_llm(pred):
    # requesting a plain-english explanation when an api key is present
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = (f"You are a cautious stock analysis assistant. Explain this "
                  f"model output in 3-4 plain sentences referencing the "
                  f"indicators. Do NOT give financial advice.\n"
                  f"Ticker: {pred['ticker']}\nDate: {pred['date']}\n"
                  f"Prediction: {pred['prediction']}\n"
                  f"Confidence: {pred['confidence']:.1%}\n"
                  f"Indicators: {pred['recent']}")
        msg = client.messages.create(model="claude-sonnet-4-6",
                                     max_tokens=300,
                                     messages=[{"role": "user",
                                                "content": prompt}])
        return msg.content[0].text
    except Exception as e:
        return f"(explanation unavailable: {e})"


def format_answer(pred):
    # turning a prediction dict into a friendly readable answer
    if "error" in pred:
        return f"Sorry — {pred['error']}. Try something like: " \
               f"'Will AAPL go up tomorrow?'"
    base = (f"{pred['ticker']} on {pred['date']}: the model says "
            f"**{pred['prediction']}** with {pred['confidence']:.0%} "
            f"confidence ({pred['model']}).\n"
            f"Recent close ${pred['recent'].get('close', 0):.2f}, "
            f"RSI {pred['recent'].get('rsi', 0):.0f}.")
    explanation = explain_with_llm(pred)
    if explanation:
        base += f"\n\n{explanation}"
    return base + "\n\nEducational output only — not financial advice."
