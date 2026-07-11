"""loading and inspecting the raw NYSE dataset"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from core.config import DATA_PATH, KAGGLE_DATASET, KAGGLE_FILES, FUNDAMENTAL_COLS
from core.helpers import log, save_plot, section


def stage_1_load():
    section("STAGE 1 — LOADING AND INSPECTING RAW DATA")

    # downloading from kaggle only when files are missing
    if not all(os.path.exists(os.path.join(DATA_PATH, f)) for f in KAGGLE_FILES):
        log("downloading dataset from kaggle...")
        os.system("pip install kaggle -q")
        os.system(f"kaggle datasets download -d {KAGGLE_DATASET} "
                  f"-p {DATA_PATH} --unzip")
    else:
        log("data files already present")

    # loading the three source tables with memory-friendly dtypes
    prices = pd.read_csv(os.path.join(DATA_PATH, "prices-split-adjusted.csv"),
                         parse_dates=["date"],
                         dtype={"open": "float32", "high": "float32",
                                "low": "float32", "close": "float32",
                                "volume": "float32"})
    fundamentals = pd.read_csv(os.path.join(DATA_PATH, "fundamentals.csv"),
                               usecols=FUNDAMENTAL_COLS)
    securities = pd.read_csv(os.path.join(DATA_PATH, "securities.csv"),
                             usecols=["Ticker symbol", "GICS Sector"])
    prices = prices.sort_values(["symbol", "date"]).reset_index(drop=True)

    # printing core shape and coverage observations
    log(f"prices shape      : {prices.shape}")
    log(f"fundamentals shape: {fundamentals.shape}")
    log(f"securities shape  : {securities.shape}")
    log(f"date range        : {prices['date'].min().date()} "
        f"to {prices['date'].max().date()}")
    log(f"unique tickers    : {prices['symbol'].nunique()}")

    # plotting trading days per ticker to spot incomplete histories
    days = prices.groupby("symbol")["date"].count().sort_values()
    plt.figure(figsize=(12, 4))
    plt.plot(days.values)
    plt.title("trading days per ticker")
    save_plot("s1_trading_days_per_ticker.png")

    # plotting the distribution of closing prices
    plt.figure(figsize=(10, 4))
    sns.histplot(prices["close"], bins=100, kde=True)
    plt.title("distribution of closing prices")
    save_plot("s1_close_price_distribution.png")

    # plotting sample tickers to sanity check the raw series
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, tk in zip(axes.flatten(), ["AAPL", "GOOGL", "JPM", "XOM"]):
        d = prices[prices["symbol"] == tk]
        ax.plot(d["date"], d["close"])
        ax.set_title(tk)
    save_plot("s1_sample_ticker_prices.png")

    # plotting total market volume over time
    vol = prices.groupby("date")["volume"].sum()
    plt.figure(figsize=(12, 4))
    plt.plot(vol.index, vol.values)
    plt.title("total market volume over time")
    save_plot("s1_total_market_volume.png")

    log("stage 1 complete")
    return prices, fundamentals, securities
