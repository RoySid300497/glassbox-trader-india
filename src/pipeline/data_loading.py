"""loading NSE price history from yfinance — a drop-in replacement for the
kaggle NYSE loader, returning the identical (prices, fundamentals,
securities) shape so every downstream stage runs unchanged"""

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from core.config import (DATA_PATH, FUNDAMENTAL_COLS,
                         EXCHANGE_SUFFIX, YF_HISTORY_YEARS)
from pipeline.universe import (NIFTY50, NIFTY50_SECTORS,
                               yf_symbols)
from core.helpers import log, save_plot, section

# some NSE names carry different yfinance symbols; try alternates
# before giving up on a ticker
SYMBOL_ALIASES = {
    "TATAMOTORS": ["TATAMOTORS.NS", "TATAMTRDVR.NS", "TML.NS"],
    "LTIM": ["LTIM.NS", "LTIMINDTREE.NS", "MINDTREE.NS"],
}

PRICES_CACHE = os.path.join(DATA_PATH, "prices-split-adjusted.csv")
FUND_CACHE = os.path.join(DATA_PATH, "fundamentals.csv")
SEC_CACHE = os.path.join(DATA_PATH, "securities.csv")


def _download_prices():
    # batch-downloading daily OHLCV for the whole universe via yfinance,
    # falling back to per-ticker pulls for any the batch missed
    import yfinance as yf
    syms = yf_symbols(EXCHANGE_SUFFIX)
    log(f"downloading {len(syms)} NSE tickers from yfinance "
        f"({YF_HISTORY_YEARS}y)...")
    raw = yf.download(syms, period=f"{YF_HISTORY_YEARS}y", auto_adjust=True,
                      group_by="ticker", progress=False, threads=True)

    frames = []
    missing = []
    for t in NIFTY50:
        sym = t + EXCHANGE_SUFFIX
        try:
            df = raw[sym].reset_index()
            if df.dropna(subset=["Close"]).empty:
                missing.append(t)
                continue
        except (KeyError, TypeError):
            missing.append(t)
            continue
        frames.append(_normalize(df, t))

    # retrying the stragglers one at a time (batch API sometimes drops names)
    for t in missing:
        # trying alias symbols for names yfinance lists differently
        candidates = SYMBOL_ALIASES.get(t, [t + EXCHANGE_SUFFIX])
        got = False
        for cand in candidates:
            try:
                df = yf.Ticker(cand).history(
                    period=f"{YF_HISTORY_YEARS}y", auto_adjust=True)
                df = df.reset_index()
                if not df.dropna(subset=["Close"]).empty:
                    frames.append(_normalize(df, t))
                    log(f"  recovered {t} via {cand}")
                    got = True
                    break
                time.sleep(0.4)
            except Exception:
                continue
        if not got:
            log(f"  could not fetch {t} (tried {candidates}); "
                f"skipping — training proceeds without it")

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["symbol", "date"]).reset_index(drop=True)
    return prices


def _normalize(df, ticker):
    # coercing a yfinance frame into the exact columns/dtypes downstream wants
    df = df.rename(columns={"Date": "date", "Open": "open", "High": "high",
                            "Low": "low", "Close": "close",
                            "Volume": "volume"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["symbol"] = ticker
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype("float32")
    return df[["date", "symbol", "open", "high", "low", "close", "volume"]]


def _synth_fundamentals(prices):
    # yfinance fundamentals are sparse and inconsistent; the pipeline
    # forward-fills and imputes these columns anyway, so we emit one neutral
    # annual row per ticker/year with NaNs — the imputer fills them from
    # training means, exactly as it would for missing kaggle values
    years = sorted(prices["date"].dt.year.unique())
    rows = []
    for t in prices["symbol"].unique():
        for y in years:
            rows.append({"Ticker Symbol": t,
                         "Period Ending": f"{y}-03-31"})
    fund = pd.DataFrame(rows)
    for c in FUNDAMENTAL_COLS[2:]:
        fund[c] = pd.NA
    return fund[FUNDAMENTAL_COLS]


def _securities():
    # the sector table, mirroring the kaggle securities.csv columns
    return pd.DataFrame(
        [{"Ticker symbol": t, "GICS Sector": NIFTY50_SECTORS[t]}
         for t in NIFTY50])


def stage_1_load():
    section("STAGE 1 — LOADING NSE DATA (yfinance)")

    # using cached CSVs when present so reruns are fast and offline-friendly
    if os.path.exists(PRICES_CACHE):
        log("cached NSE prices present, loading from disk")
        prices = pd.read_csv(PRICES_CACHE, parse_dates=["date"],
                             dtype={"open": "float32", "high": "float32",
                                    "low": "float32", "close": "float32",
                                    "volume": "float32"})
    else:
        prices = _download_prices()
        prices.to_csv(PRICES_CACHE, index=False)
        log(f"cached prices to {PRICES_CACHE}")

    fundamentals = _synth_fundamentals(prices)
    securities = _securities()
    fundamentals.to_csv(FUND_CACHE, index=False)
    securities.to_csv(SEC_CACHE, index=False)

    # the same coverage observations the US loader printed
    log(f"prices shape      : {prices.shape}")
    log(f"fundamentals shape: {fundamentals.shape}")
    log(f"securities shape  : {securities.shape}")
    log(f"date range        : {prices['date'].min().date()} "
        f"to {prices['date'].max().date()}")
    log(f"unique tickers    : {prices['symbol'].nunique()}")

    # trading days per ticker — flags any thin NSE history
    days = prices.groupby("symbol")["date"].count().sort_values()
    plt.figure(figsize=(12, 4))
    plt.plot(days.values)
    plt.title("trading days per ticker (NSE)")
    save_plot("s1_trading_days_per_ticker.png")

    plt.figure(figsize=(10, 4))
    sns.histplot(prices["close"], bins=100, kde=True)
    plt.title("distribution of closing prices (INR)")
    save_plot("s1_close_price_distribution.png")

    # sample large-caps to sanity-check the raw series
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, tk in zip(axes.flatten(),
                      ["RELIANCE", "TCS", "HDFCBANK", "INFY"]):
        d = prices[prices["symbol"] == tk]
        ax.plot(d["date"], d["close"])
        ax.set_title(tk)
    save_plot("s1_sample_ticker_prices.png")

    vol = prices.groupby("date")["volume"].sum()
    plt.figure(figsize=(12, 4))
    plt.plot(vol.index, vol.values)
    plt.title("total market volume over time (NSE)")
    save_plot("s1_total_market_volume.png")

    log("stage 1 complete")
    return prices, fundamentals, securities
