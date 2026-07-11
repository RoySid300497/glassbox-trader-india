"""building the training feature set on live yfinance data for one ticker"""

import os
import numpy as np
import pandas as pd
from core.config import DATA_PATH, LAG_COLS, LAG_DAYS
from pipeline.features import add_indicators
from pipeline.enhanced_features import add_lags, add_returns

# mapping GICS sectors to their SPDR sector ETF proxies
SECTOR_ETF = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Telecommunications Services": "XLC",
    "Communication Services": "XLC",
}


def lookup_sector(ticker):
    # finding the ticker's GICS sector from the dataset securities file
    uni_path = os.path.join(DATA_PATH, "universe.csv")
    sec_path = uni_path if os.path.exists(uni_path) \
        else os.path.join(DATA_PATH, "securities.csv")
    sec = pd.read_csv(sec_path,
                      usecols=["Ticker symbol", "GICS Sector"])
    row = sec[sec["Ticker symbol"].str.upper() == ticker.upper()]
    return row["GICS Sector"].iloc[0] if not row.empty else None


def fetch_close_series(symbol, days):
    # downloading a recent close series for one symbol from yfinance
    import yfinance as yf
    hist = yf.download(symbol, period=f"{days}d", auto_adjust=True,
                       progress=False)
    if hist is None or hist.empty:
        return None
    hist = hist.reset_index()
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = [c[0] for c in hist.columns]
    return hist


def build_live_frame(ticker, days=250):
    # producing a feature dataframe for one ticker on current market data
    hist = fetch_close_series(ticker.replace(".", "-"), days)
    if hist is None:
        return None
    df = hist.rename(columns={"Date": "date", "Open": "open", "High": "high",
                              "Low": "low", "Close": "close",
                              "Volume": "volume"})
    df["symbol"] = ticker.upper()
    df = df[["date", "symbol", "open", "high", "low", "close", "volume"]]

    # computing indicators, lags, and returns with the training pipeline
    df = add_indicators(df)
    df = add_lags(df)
    df = add_returns(df)

    # proxying market return with SPY daily returns
    spy = fetch_close_series("SPY", days)
    if spy is not None:
        spy["market_return"] = spy["Close"].pct_change()
        spy = spy.rename(columns={"Date": "date"})[["date", "market_return"]]
        df = df.merge(spy, on="date", how="left")
    else:
        df["market_return"] = 0.0
    df["rel_to_market"] = df["return_1d"] - df["market_return"]

    # proxying sector return with the matching SPDR sector ETF
    sector = lookup_sector(ticker)
    etf = SECTOR_ETF.get(sector)
    if etf:
        sec_hist = fetch_close_series(etf, days)
        if sec_hist is not None:
            sec_hist["sector_return"] = sec_hist["Close"].pct_change()
            sec_hist = sec_hist.rename(columns={"Date": "date"})[
                ["date", "sector_return"]]
            df = df.merge(sec_hist, on="date", how="left")
    if "sector_return" not in df.columns:
        df["sector_return"] = df["market_return"]
    df["rel_to_sector"] = df["return_1d"] - df["sector_return"]

    # dropping warmup rows so every indicator is populated
    df = df.dropna(subset=["ma50", "rsi", "vol_ratio"]).reset_index(drop=True)
    return df


def fill_missing_features(df, feature_cols, scaler):
    # creating absent columns and filling them with scaler means
    means = dict(zip(feature_cols, scaler.mean_))
    for c in feature_cols:
        if c not in df.columns:
            df[c] = means[c]
        else:
            df[c] = df[c].fillna(means[c])
    return df
