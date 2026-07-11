"""assembling the grounded data packet every panel argues over"""

import json
from engine.memory import (get_recent_news, get_recent_decisions,
                           get_active_lessons, get_active_thesis,
                           get_market_context, validate_ticker,
                           get_open_position, get_ticker_stats)


def _rf_signal(df):
    # producing the packet signal from the random forest on the latest row
    import numpy as np
    from inference.predictors import load_rf_predictor
    rf = load_rf_predictor()
    if rf is None:
        return None
    frame = df.copy()
    for c in rf["feature_cols"]:
        if c not in frame.columns:
            frame[c] = np.nan
    latest = frame.iloc[[-1]][rf["feature_cols"]]
    x = rf["scaler"].transform(rf["imputer"].transform(latest))
    p = rf["model"].predict_proba(x)[0]
    idx = int(p.argmax())
    row = df.iloc[-1]
    return {"model": "random_forest",
            "direction": str(rf["label_encoder"].classes_[idx]),
            "confidence": round(float(p[idx]), 4),
            "close": round(float(row["close"]), 2),
            "rsi": round(float(row["rsi"]), 1),
            "return_5d": round(float(row["return_5d"]), 4),
            "return_10d": round(float(row["return_10d"]), 4),
            "pct_vs_ma50": round(float(row["close"] / row["ma50"] - 1), 4),
            "vol_ratio": round(float(row["vol_ratio"]), 2),
            "rel_to_sector": round(float(row["rel_to_sector"]), 4)}


def get_cnn_signal(ticker):
    # stamping which model actually holds the title into the signal block
    from engine.champion import get_champion
    champion = "cnn1d"
    try:
        champion = get_champion()
    except Exception:
        pass
    sig = _champion_signal(ticker)
    if isinstance(sig, dict):
        sig["model"] = champion
    return sig


def _champion_signal(ticker):
    # producing the packet signal from whichever model holds the title
    import numpy as np
    import torch
    from inference.predictors import load_seq_predictor
    from inference.live_features import build_live_frame, fill_missing_features
    from engine.champion import get_champion
    champion = "cnn1d"
    try:
        champion = get_champion()
    except Exception:
        pass
    seq = load_seq_predictor()
    if seq is None:
        return {"direction": "unavailable", "confidence": 0.0}
    df = build_live_frame(ticker)
    if df is None or df.empty:
        return {"direction": "unavailable", "confidence": 0.0}
    if champion == "random_forest":
        sig = _rf_signal(df)
        if sig is not None:
            return sig
    meta = seq["meta"]
    df = fill_missing_features(df, meta["feature_cols"], seq["scaler"])
    if len(df) < meta["window"]:
        return {"direction": "unavailable", "confidence": 0.0}
    win = df.iloc[-meta["window"]:][meta["feature_cols"]] \
        .values.astype("float32")
    win = seq["scaler"].transform(win)
    with torch.no_grad():
        out = seq["model"](torch.tensor(win).unsqueeze(0)).numpy().squeeze()
    probs = np.exp(out) / np.exp(out).sum()
    idx = int(probs.argmax())
    latest = df.iloc[-1]
    return {"model": meta.get("kind", "cnn1d"),
            "direction": meta["classes"][idx],
            "confidence": round(float(probs[idx]), 4),
            "close": round(float(latest["close"]), 2),
            "rsi": round(float(latest["rsi"]), 1),
            "return_5d": round(float(latest["return_5d"]), 4),
            "return_10d": round(float(latest["return_10d"]), 4),
            "pct_vs_ma50": round(float(latest["close"] / latest["ma50"] - 1), 4),
            "vol_ratio": round(float(latest["vol_ratio"]), 2),
            "rel_to_sector": round(float(latest["rel_to_sector"]), 4)}


def _structure_block(ticker):
    # summarizing price structure as citable evidence on a guaranteed ohlc frame
    try:
        import pandas as pd
        import yfinance as yf
        from pipeline.ta_structure import technical_structure_block
        hist = yf.download(ticker.replace(".", "-"), period="1y",
                           auto_adjust=True, progress=False)
        if hist is None or hist.empty or len(hist) < 60:
            return None
        # flattening the multiindex newer yfinance returns for single tickers
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        df = hist.rename(columns=str.lower)[["open", "high", "low", "close"]]
        return technical_structure_block(df)
    except Exception as e:
        print(f"  [structure] {ticker} block unavailable: {e}")
        return None


def _overnight_gap(ticker):
    # measuring the move since the prior close using yahoo prepost prints
    try:
        import pandas as pd
        import yfinance as yf
        from datetime import date
        sym = ticker.replace(".", "-")
        daily = yf.download(sym, period="5d", auto_adjust=True,
                            progress=False)
        if daily is None or daily.empty:
            return None
        if isinstance(daily.columns, pd.MultiIndex):
            daily.columns = daily.columns.get_level_values(0)
        # anchoring on the last fully completed session before today
        daily = daily[pd.to_datetime(daily.index).date < date.today()]
        if daily.empty:
            return None
        prev_close = float(daily["Close"].iloc[-1])
        intra = yf.download(sym, period="1d", interval="1m", prepost=True,
                            progress=False)
        if intra is None or intra.empty:
            return None
        if isinstance(intra.columns, pd.MultiIndex):
            intra.columns = intra.columns.get_level_values(0)
        last = float(intra["Close"].dropna().iloc[-1])
        return round(last / prev_close - 1, 4)
    except Exception as e:
        print(f"  [gap] {ticker} unavailable: {e}")
        return None


_reliability_cache = None


def _reliability_block():
    # attaching evidence grades once per run, tolerating any failure
    global _reliability_cache
    if _reliability_cache is None:
        try:
            from engine.evidence_weights import evidence_reliability_block
            _reliability_cache = evidence_reliability_block()
        except Exception as e:
            print(f"  [packet] reliability block failed: {e}")
            _reliability_cache = {}
    return _reliability_cache or None


def _macro_block():
    # attaching market-wide headlines, tolerating any failure
    try:
        from engine.news_fetcher import fetch_macro_news
        from datetime import datetime, timezone
        items = fetch_macro_news()
        out = []
        for i in items:
            age = None
            try:
                ts = datetime.fromisoformat(i["published_at"])
                age = round((datetime.now(timezone.utc) - ts)
                            .total_seconds() / 3600, 1)
            except Exception:
                pass
            out.append({"headline": i["headline"],
                        "sentiment": i["sentiment"],
                        "age_hours": age})
        return out or None
    except Exception as e:
        print(f"  [packet] macro block failed: {e}")
        return None


def _congress_block_safe(ticker):
    # attaching congressional disclosures, tolerating any failure
    try:
        from engine.congress import congress_block
        return congress_block(ticker)
    except Exception as e:
        print(f"  [packet] congress block failed: {e}")
        return None


def _insider_block(ticker):
    # attaching recent insider filing evidence, tolerating any failure
    try:
        from engine.smart_money import insider_activity
        return insider_activity(ticker)
    except Exception as e:
        print(f"  [insider] block failed for {ticker}: {e}")
        return None


def build_packet(ticker, news_items):
    # combining signal, structure, news, history, lessons, thesis, and context
    from engine.news_fetcher import fetch_next_earnings
    ticker = validate_ticker(ticker)
    sentiments = [n.get("sentiment") for n in news_items[:5]
                  if n.get("sentiment") is not None]
    _gap = _overnight_gap(ticker)
    # a meaningful gap with no news catalyst is "gapping to air" — the
    # corpus flags these as fade-prone versus catalyst-backed gaps which
    # carry a real thesis; naming the combination lets judges weigh it
    # directly instead of inferring it across two separate blocks
    _has_catalyst = any(abs(s) > 0.15 for s in sentiments)
    packet = {
        "ticker": ticker,
        "technical_structure": _structure_block(ticker),
        "overnight_gap_pct": _gap,
        "overnight_gap_no_catalyst": bool(
            _gap is not None and abs(_gap) >= 1.5 and not _has_catalyst),
        "insider_activity": _insider_block(ticker),
        "congress_trading": _congress_block_safe(ticker),
        "evidence_reliability": _reliability_block(),
        "macro_news": _macro_block(),
        "days_to_earnings": fetch_next_earnings(ticker),
        "news": [{"headline": n["headline"][:200],
                  "summary": (n.get("summary") or "")[:300],
                  "source": n.get("source", ""),
                  "sentiment": n.get("sentiment")} for n in news_items[:5]],
        "news_sentiment_avg": round(sum(sentiments) / len(sentiments), 3)
        if sentiments else None,
        # one c-grade opinion among many, deliberately not the headline
        "cnn_signal": get_cnn_signal(ticker),
        "recent_decisions": get_recent_decisions(ticker, limit=5),
        "lessons": get_active_lessons(limit=8),
        "active_thesis": get_active_thesis(ticker),
        "open_position": get_open_position(ticker),
        "ticker_track_record": get_ticker_stats(ticker),
        "market_context": get_market_context()[:1000],
    }
    return packet


def packet_to_text(packet):
    # serializing the packet as clearly delimited json for prompts
    return ("=== DATA PACKET (the only permitted evidence) ===\n"
            + json.dumps(packet, indent=1, default=str)[:6000]
            + "\n=== END DATA PACKET ===")
