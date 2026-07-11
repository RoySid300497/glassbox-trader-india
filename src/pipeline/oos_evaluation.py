"""evaluating saved models on yfinance 2017-2025 data across regimes"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from core.config import (DATA_PATH, OBS_PATH, LAG_COLS, LAG_DAYS,
                    OOS_START, OOS_EVAL_FROM, OOS_END)
from core.helpers import log, save_plot, section
from pipeline.features import add_indicators
from pipeline.enhanced_features import add_lags, add_returns
from inference.predictors import load_seq_predictor, load_rf_predictor


def prepare_oos_frame(limit=None, start=OOS_START, end=OOS_END):
    # downloading yfinance data and rebuilding the training feature pipeline
    import yfinance as yf
    # preferring the refreshed current universe over the 2016 training roster
    uni_path = os.path.join(DATA_PATH, "universe.csv")
    sec_path = uni_path if os.path.exists(uni_path) \
        else os.path.join(DATA_PATH, "securities.csv")
    securities = pd.read_csv(sec_path,
                             usecols=["Ticker symbol", "GICS Sector"])
    securities = securities.rename(columns={"Ticker symbol": "symbol",
                                            "GICS Sector": "sector"})
    tickers = securities["symbol"].tolist()
    if limit:
        tickers = tickers[:limit]
    yf_map = {t: t.replace(".", "-") for t in tickers}

    log(f"downloading {len(tickers)} tickers from yfinance "
        f"({start} to {end})...")
    raw = yf.download(list(yf_map.values()), start=start, end=end,
                      group_by="ticker", auto_adjust=True, progress=False,
                      threads=True)
    frames = []
    for orig, yft in yf_map.items():
        try:
            sub = raw[yft].dropna(subset=["Close"]).reset_index()
        except Exception:
            continue
        if len(sub) < 100:
            continue
        sub = sub.rename(columns={"Date": "date", "Open": "open",
                                  "High": "high", "Low": "low",
                                  "Close": "close", "Volume": "volume"})
        sub["symbol"] = orig
        frames.append(sub[["date", "symbol", "open", "high", "low",
                           "close", "volume"]])
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df.merge(securities, on="symbol", how="left")
    log(f"downloaded {df.shape[0]:,} rows across "
        f"{df['symbol'].nunique()} tickers")

    # rebuilding indicators, lags, returns, and relative features
    log("rebuilding features on out-of-sample data...")
    df = pd.concat([add_indicators(g) for _, g in
                    df.groupby("symbol", sort=False)], ignore_index=True)
    df = pd.concat([add_lags(g) for _, g in
                    df.groupby("symbol", sort=False)], ignore_index=True)
    df = pd.concat([add_returns(g) for _, g in
                    df.groupby("symbol", sort=False)], ignore_index=True)
    df["market_return"] = df.groupby("date")["return_1d"].transform("mean")
    df["rel_to_market"] = df["return_1d"] - df["market_return"]
    df["sector_return"] = df.groupby(["date", "sector"])["return_1d"] \
        .transform("mean")
    df["rel_to_sector"] = df["return_1d"] - df["sector_return"]

    # creating the same next-day label used during training
    df = df.sort_values(["symbol", "date"])
    df["next_day_return"] = df.groupby("symbol")["close"] \
        .pct_change().shift(-1)
    df["true_label"] = df["next_day_return"].apply(
        lambda p: "Up" if p > 0.01 else ("Down" if p < -0.01 else "Neutral"))
    return df.dropna(subset=["next_day_return", "ma50", "rsi"]) \
        .reset_index(drop=True)


def stage_6_oos():
    section("STAGE 6 — OUT-OF-SAMPLE EVALUATION (2017-2025)")
    os.system("pip install yfinance -q")
    from sklearn.metrics import f1_score, classification_report

    seq_pred = load_seq_predictor()
    rf_pred = load_rf_predictor()
    if seq_pred is None and rf_pred is None:
        log("no saved models — run stages 3 and 4 first")
        return
    limit = os.environ.get("STOCK_LENS_OOS_TICKERS")
    df = prepare_oos_frame(limit=int(limit) if limit else None)
    if df is None:
        log("yfinance download failed")
        return
    df = df[df["date"] >= OOS_EVAL_FROM]
    df["year"] = df["date"].dt.year
    rows = []

    # scoring the random forest year by year
    if rf_pred is not None:
        fcols = rf_pred["feature_cols"]
        for c in fcols:
            if c not in df.columns:
                df[c] = np.nan
        X = rf_pred["scaler"].transform(rf_pred["imputer"]
                                        .transform(df[fcols]))
        df["rf_pred"] = rf_pred["label_encoder"] \
            .inverse_transform(rf_pred["model"].predict(X))
        for yr, grp in df.groupby("year"):
            rows.append({"model": "random_forest", "year": yr,
                         "macro_f1": f1_score(grp["true_label"],
                                              grp["rf_pred"],
                                              average="macro")})
        log(f"\nrf OOS overall macro_f1 = "
            f"{f1_score(df['true_label'], df['rf_pred'], average='macro'):.4f}")
        log(classification_report(df["true_label"], df["rf_pred"]))

    # scoring the sequence model year by year with windowed inputs
    if seq_pred is not None:
        import torch
        meta = seq_pred["meta"]
        fcols, window = meta["feature_cols"], meta["window"]
        label_to_idx = {"Down": 0, "Neutral": 1, "Up": 2}
        for c in fcols:
            if c not in df.columns:
                df[c] = np.nan
        means = dict(zip(fcols, seq_pred["scaler"].mean_))
        for c in fcols:
            df[c] = df[c].fillna(means[c])
        recs = []
        for _, grp in df.groupby("symbol", sort=False):
            grp = grp.sort_values("date")
            feats = seq_pred["scaler"].transform(grp[fcols]) \
                .astype("float32")
            labels = grp["true_label"].map(label_to_idx).values
            years = grp["year"].values
            for i in range(window, len(grp) - 1):
                recs.append((feats[i - window:i], labels[i], years[i]))
        Xs = np.stack([r[0] for r in recs])
        ys = np.array([r[1] for r in recs])
        yrs = np.array([r[2] for r in recs])
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xs), 4096):
                out = seq_pred["model"](torch.tensor(Xs[i:i + 4096])).numpy()
                preds.append(out.argmax(axis=1))
        preds = np.concatenate(preds)
        for yr in sorted(set(yrs)):
            m = yrs == yr
            rows.append({"model": f"sequence_{meta['kind']}", "year": int(yr),
                         "macro_f1": f1_score(ys[m], preds[m],
                                              average="macro")})
        log(f"\nsequence OOS overall macro_f1 = "
            f"{f1_score(ys, preds, average='macro'):.4f}")
        log(classification_report(ys, preds,
                                  target_names=["Down", "Neutral", "Up"]))

    # saving the per-year regime table and plot
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OBS_PATH, "s6_oos_results.csv"), index=False)
    log("\nOOS RESULTS BY YEAR:\n" + res.to_string(index=False))
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=res, x="year", y="macro_f1", hue="model", marker="o")
    plt.axhline(0.333, color="red", linestyle="--")
    plt.title("out-of-sample macro F1 by year")
    save_plot("s6_oos_macro_f1_by_year.png")
    log("stage 6 complete")
    return res
