"""building technical indicators, merging fundamentals, creating labels"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from core.config import (DATA_PATH, FUND_FILL_COLS, TRAIN_END, VAL_END)
from core.helpers import log, save_plot, section


def add_indicators(group):
    # computing indicators for one ticker so nothing bleeds across stocks
    import pandas_ta as ta
    group = group.copy()
    group["ma10"] = group["close"].rolling(10).mean()
    group["ma30"] = group["close"].rolling(30).mean()
    group["ma50"] = group["close"].rolling(50).mean()
    group["rsi"] = ta.rsi(group["close"], length=14)
    group["vol_ratio"] = group["volume"] / group["volume"].rolling(20).mean()

    # assigning macd columns directly to keep index alignment safe
    macd = ta.macd(group["close"], fast=12, slow=26, signal=9)
    for col in macd.columns:
        group[col] = macd[col].values

    # assigning bollinger columns directly for the same reason
    bb = ta.bbands(group["close"], length=20, std=2)
    for col in bb.columns:
        group[col] = bb[col].values
    return group


def plot_indicator_panel(df, ticker, filename):
    # plotting price, RSI, MACD, and bollinger for one ticker
    d = df[df["symbol"] == ticker].copy()
    if d.empty:
        return
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    axes[0].plot(d["date"], d["close"], label="close", linewidth=1)
    for ma in ["ma10", "ma30", "ma50"]:
        axes[0].plot(d["date"], d[ma], label=ma.upper(), linestyle="--")
    axes[0].set_title(f"{ticker} — close + moving averages")
    axes[0].legend()
    axes[1].plot(d["date"], d["rsi"], color="purple")
    axes[1].axhline(70, color="red", linestyle="--", alpha=0.7)
    axes[1].axhline(30, color="green", linestyle="--", alpha=0.7)
    axes[1].set_title("RSI (14)")
    axes[2].plot(d["date"], d["MACD_12_26_9"], label="MACD")
    axes[2].plot(d["date"], d["MACDs_12_26_9"], label="signal")
    axes[2].bar(d["date"], d["MACDh_12_26_9"], label="histogram", alpha=0.4)
    axes[2].set_title("MACD")
    axes[2].legend()
    axes[3].plot(d["date"], d["close"], label="close")
    axes[3].plot(d["date"], d["BBU_20_2.0_2.0"], label="upper", linestyle="--")
    axes[3].plot(d["date"], d["BBL_20_2.0_2.0"], label="lower", linestyle="--")
    axes[3].set_title("Bollinger Bands (20, 2sd)")
    axes[3].legend()
    save_plot(filename)


def stage_2_features():
    section("STAGE 2 — FEATURE ENGINEERING, LABELS, SPLIT")
    os.system("pip install pandas_ta -q")
    from pipeline.data_loading import stage_1_load
    prices, fundamentals, securities = stage_1_load()

    # merging sector info onto every price row
    sector_info = securities.rename(columns={"Ticker symbol": "symbol",
                                             "GICS Sector": "sector"})
    prices = prices.merge(sector_info, on="symbol", how="left")

    # plotting how many tickers sit in each sector
    sc = prices.groupby("sector")["symbol"].nunique().sort_values()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=sc.values, y=sc.index, hue=sc.index,
                palette="Blues_r", legend=False)
    plt.title("unique tickers per sector")
    save_plot("s2_tickers_per_sector.png")

    # computing indicators with an explicit loop that preserves symbol
    log("computing technical indicators...")
    prices = pd.concat([add_indicators(g) for _, g in
                        prices.groupby("symbol", sort=False)],
                       ignore_index=True)
    log(f"shape after indicators: {prices.shape}")

    # forward-filling annual fundamentals onto daily rows by year
    fundamentals = fundamentals.rename(columns={"Ticker Symbol": "symbol"})
    fundamentals["Period Ending"] = pd.to_datetime(fundamentals["Period Ending"])
    fundamentals["year"] = fundamentals["Period Ending"].dt.year
    prices["year"] = prices["date"].dt.year
    prices = prices.merge(fundamentals.drop(columns=["Period Ending"]),
                          on=["symbol", "year"], how="left").drop(columns=["year"])

    # forward-filling fundamental gaps within each ticker
    prices = prices.sort_values(["symbol", "date"])
    prices[FUND_FILL_COLS] = prices.groupby("symbol")[FUND_FILL_COLS].ffill()

    # dropping warmup rows and remaining indicator nulls
    prices = pd.concat([g.iloc[50:] for _, g in
                        prices.groupby("symbol", sort=False)],
                       ignore_index=True)
    prices = prices.dropna(subset=["ma50", "rsi", "vol_ratio",
                                   "MACD_12_26_9", "BBU_20_2.0_2.0"])
    log(f"shape after cleanup: {prices.shape}")

    # creating the next-day label with the +/-1% threshold
    def make_label(pct):
        return "Up" if pct > 0.01 else ("Down" if pct < -0.01 else "Neutral")

    prices["next_day_return"] = prices.groupby("symbol")["close"] \
        .pct_change().shift(-1)
    prices["label"] = prices["next_day_return"].apply(make_label)
    prices = prices.dropna(subset=["next_day_return"])

    # plotting label balance
    lc = prices["label"].value_counts()
    plt.figure(figsize=(7, 4))
    sns.barplot(x=lc.index, y=lc.values, hue=lc.index,
                palette=["tomato", "steelblue", "seagreen"], legend=False)
    plt.title("label distribution (next-day, +/-1%)")
    save_plot("s2_label_distribution.png")
    log(f"label distribution:\n"
        f"{prices['label'].value_counts(normalize=True).mul(100).round(1)}")

    # plotting an indicator panel and correlation heatmap
    plot_indicator_panel(prices, "AAPL", "s2_AAPL_indicator_panel.png")
    corr_cols = ["close", "ma10", "ma30", "ma50", "rsi", "vol_ratio",
                 "MACD_12_26_9", "Earnings Per Share", "Profit Margin"]
    plt.figure(figsize=(11, 8))
    sns.heatmap(prices[corr_cols].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0)
    plt.title("feature correlation heatmap")
    save_plot("s2_feature_correlation.png")

    # splitting chronologically and saving all artifacts
    train = prices[prices["date"] < TRAIN_END]
    val = prices[(prices["date"] >= TRAIN_END) & (prices["date"] < VAL_END)]
    test = prices[prices["date"] >= VAL_END]
    log(f"train: {train.shape} | val: {val.shape} | test: {test.shape}")
    prices.to_csv(os.path.join(DATA_PATH, "master.csv"), index=False)
    train.to_csv(os.path.join(DATA_PATH, "train.csv"), index=False)
    val.to_csv(os.path.join(DATA_PATH, "val.csv"), index=False)
    test.to_csv(os.path.join(DATA_PATH, "test.csv"), index=False)
    log("stage 2 complete")
    return prices
