"""running the chatbot that understands natural language questions"""

import os
import pandas as pd
from core.config import DATA_PATH
from core.helpers import log, section
from inference.predictors import (load_seq_predictor, load_rf_predictor,
                        predict_one, format_answer)
from inference.nlp_parser import load_ticker_names, parse_query


def stage_5_chatbot(mode="cli"):
    section("STAGE 5 — CHATBOT (natural language)")

    # loading models, data, and the ticker name lookup
    seq_pred = load_seq_predictor()
    rf_pred = load_rf_predictor()
    log(f"sequence model loaded: {seq_pred is not None}")
    log(f"random forest loaded: {rf_pred is not None}")
    if seq_pred is None and rf_pred is None:
        log("no models found — run stage 3 and stage 4 first")
        return
    data = pd.read_csv(os.path.join(DATA_PATH, "master_enhanced.csv"),
                       parse_dates=["date"])
    tickers, name_map = load_ticker_names()
    latest_date = data["date"].max()
    default_model = "sequence" if seq_pred is not None else "random_forest"

    def answer(question):
        # parsing the question and routing it to the right model
        q = parse_query(question, tickers, name_map, latest_date,
                        default_model)
        if q["ticker"] is None:
            return ("Sorry, that stock isn't in my dataset (S&P 500 "
                    "companies, 2010-2016 — no TSLA or AMZN, for example), "
                    "or the question needs a ticker. Try: 'Will Apple go "
                    "up tomorrow?' or 'predict JPM on 2016-06-24'")
        pred = predict_one(q["ticker"], q["date"], q["model"],
                           seq_pred, rf_pred, data)
        return format_answer(pred)

    # launching the gradio web interface when requested
    if mode in ("gradio", "both"):
        try:
            import gradio as gr
            demo = gr.Interface(
                fn=answer,
                inputs=gr.Textbox(
                    label="Ask me anything about a stock",
                    placeholder="Will Apple go up tomorrow? "
                                "What about MSFT on 2016-06-15?"),
                outputs=gr.Textbox(label="Answer", lines=8),
                title="stock-lens",
                description="Ask in plain English. Mention a ticker or "
                            "company name, optionally a date, and optionally "
                            "'random forest' or 'cnn'. Data covers "
                            "2010-2016; 'today' means the latest day in the "
                            "dataset.")
            log("launching gradio...")
            demo.launch(share=True)
            return
        except ImportError:
            log("gradio missing — falling back to CLI")

    # running the command-line chat loop
    log("\nchat ready — ask naturally, e.g. 'Will Apple go up tomorrow?'")
    log("type 'quit' to exit")
    while True:
        try:
            raw = input("\nyou> ").strip()
        except EOFError:
            break
        if raw.lower() in ("quit", "exit", "q"):
            break
        if raw:
            print("\nbot> " + answer(raw))
    log("chat session ended")
