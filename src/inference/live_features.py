"""building the training feature set on live yfinance data for one ticker"""

import os
import numpy as np
import pandas as pd
from core.config import DATA_PATH, LAG_COLS, LAG_DAYS
from pipeline.features import add_indicators
from pipeline.enhanced_features import add_lags, add_returns

# mapping our GICS-style sector buckets to NSE SECTORAL INDICES (yfinance
# symbols). the old map pointed at US SPDR ETFs (XLK, XLF, ...) — a
# different market, hours, and currency — so rel_to_sector measured an
# indian stock against the AMERICAN sector. training computes sector
# return from indian same-sector peers; these indices are the closest
# live proxy. any index that fails to download degrades to the nifty
# itself, so rel_to_sector falls back to rel_to_market, never to noise.
SECTOR_INDEX = {
    "Information Technology": "^CNXIT",
    "Financials": "^NSEBANK",
    "Health Care": "^CNXPHARMA",
    "Consumer Staples": "^CNXFMCG",
    "Consumer Discretionary": "^CNXAUTO",
    "Energy": "^CNXENERGY",
    "Utilities": "^CNXENERGY",
    "Materials": "^CNXMETAL",
    "Industrials": "^CNXINFRA",
    "Real Estate": "^CNXREALTY",
    "Communication Services": "^CNXSERVICE",
    "Telecommunications Services": "^CNXSERVICE",
}
MARKET_INDEX = "^NSEI"


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
    # forming the NSE symbol (append .NS); the old replace(".", "-") stripped
    # the suffix and made every india download fail with "no timezone found"
    from core.config import EXCHANGE_SUFFIX
    _sym = (ticker if not EXCHANGE_SUFFIX or ticker.endswith(EXCHANGE_SUFFIX)
            else ticker + EXCHANGE_SUFFIX)
    hist = fetch_close_series(_sym, days)
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

    # proxying market return with the NIFTY 50 index (was SPY — the US
    # market — which made rel_to_market measure the wrong country entirely)
    mkt = fetch_close_series(MARKET_INDEX, days)
    if mkt is not None:
        mkt["market_return"] = mkt["Close"].pct_change()
        mkt = mkt.rename(columns={"Date": "date"})[["date", "market_return"]]
        df = df.merge(mkt, on="date", how="left")
    else:
        df["market_return"] = 0.0
    df["rel_to_market"] = df["return_1d"] - df["market_return"]

    # proxying sector return with the matching NSE sectoral index,
    # degrading to the nifty (i.e. rel_to_market) when unavailable
    sector = lookup_sector(ticker)
    idx = SECTOR_INDEX.get(sector)
    if idx:
        sec_hist = fetch_close_series(idx, days)
        if sec_hist is not None:
            sec_hist["sector_return"] = sec_hist["Close"].pct_change()
            sec_hist = sec_hist.rename(columns={"Date": "date"})[
                ["date", "sector_return"]]
            df = df.merge(sec_hist, on="date", how="left")
    if "sector_return" not in df.columns:
        df["sector_return"] = df["market_return"]
    df["sector_return"] = df["sector_return"].fillna(df["market_return"])
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
