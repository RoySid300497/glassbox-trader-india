"""parsing natural language questions into ticker, date, and model choice"""

import os
import re
import pandas as pd
from core.config import DATA_PATH


def load_ticker_names():
    # building ticker and company-name lookup tables from securities.csv
    sec = pd.read_csv(os.path.join(DATA_PATH, "securities.csv"))
    tickers = set(sec["Ticker symbol"].str.upper())
    name_map = {}
    for _, row in sec.iterrows():
        full = str(row["Security"]).lower()
        name_map[full] = row["Ticker symbol"]
        first = full.split()[0]
        # mapping the distinctive first word of each company name to its ticker
        if len(first) > 3 and first not in ("the", "first", "general",
                                            "american", "united", "national"):
            name_map.setdefault(first, row["Ticker symbol"])
    return tickers, name_map


def parse_date(text, latest_date):
    # extracting a date from the question, defaulting to the latest available
    text = text.lower()
    iso = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if iso:
        return iso.group()
    us = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if us:
        m, d, y = us.groups()
        return f"{y}-{int(m):02d}-{int(d):02d}"
    month_pat = (r"(january|february|march|april|may|june|july|august|"
                 r"september|october|november|december)\s+(\d{1,2}),?\s*(\d{4})")
    named = re.search(month_pat, text)
    if named:
        try:
            return str(pd.to_datetime(named.group()).date())
        except Exception:
            pass
    # treating today, tomorrow, latest, and now as the newest available day
    return str(pd.Timestamp(latest_date).date())


def parse_model(text):
    # detecting which model the person wants from casual phrasing
    text = text.lower()
    if any(w in text for w in ("forest", "tree", " rf", "tabular", "simple")):
        return "random_forest"
    if any(w in text for w in ("cnn", "sequence", "neural", "deep", "lstm")):
        return "sequence"
    return None


def parse_ticker(text, tickers, name_map):
    # finding a ticker symbol or company name anywhere in the question
    stopwords = {"I", "A", "UP", "DOWN", "ON", "IN", "AT", "TO", "IS", "IT",
                 "GO", "DO", "BE", "OR", "SO", "USE", "THE", "CNN", "RF"}
    for token in re.findall(r"\b[A-Z][A-Z0-9.\-]{0,5}\b", text):
        if token in tickers and token not in stopwords:
            return token
    lowered = text.lower()
    # preferring the longest matching company name to avoid generic collisions
    matches = [(name, tk) for name, tk in name_map.items() if name in lowered]
    if matches:
        return max(matches, key=lambda m: len(m[0]))[1]
    return None


def parse_query(text, tickers, name_map, latest_date, default_model):
    # combining all extractors into one structured query dict
    ticker = parse_ticker(text, tickers, name_map)
    date = parse_date(text, latest_date)
    model = parse_model(text) or default_model
    return {"ticker": ticker, "date": date, "model": model}
