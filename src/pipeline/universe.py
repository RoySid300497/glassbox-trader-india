"""the nifty 50 universe and its GICS-style sector map for india/NSE

kept as plain data so the loader and the trading engine share one source
of truth; refresh when NSE rebalances the index (roughly twice a year)
"""

# nifty 50 constituents (symbol -> sector). sectors use broad GICS-like
# buckets so the existing sector-relative features work unchanged
NIFTY50_SECTORS = {
    "RELIANCE": "Energy",
    "ONGC": "Energy",
    "BPCL": "Energy",
    "COALINDIA": "Energy",
    "NTPC": "Utilities",
    "POWERGRID": "Utilities",
    "TCS": "Information Technology",
    "INFY": "Information Technology",
    "HCLTECH": "Information Technology",
    "TECHM": "Information Technology",
    "WIPRO": "Information Technology",
    "LTIM": "Information Technology",
    "HDFCBANK": "Financials",
    "ICICIBANK": "Financials",
    "SBIN": "Financials",
    "KOTAKBANK": "Financials",
    "AXISBANK": "Financials",
    "INDUSINDBK": "Financials",
    "BAJFINANCE": "Financials",
    "BAJAJFINSV": "Financials",
    "SBILIFE": "Financials",
    "HDFCLIFE": "Financials",
    "SHRIRAMFIN": "Financials",
    "HINDUNILVR": "Consumer Staples",
    "ITC": "Consumer Staples",
    "NESTLEIND": "Consumer Staples",
    "BRITANNIA": "Consumer Staples",
    "TATACONSUM": "Consumer Staples",
    "MARUTI": "Consumer Discretionary",
    "TATAMOTORS": "Consumer Discretionary",
    "M&M": "Consumer Discretionary",
    "EICHERMOT": "Consumer Discretionary",
    "HEROMOTOCO": "Consumer Discretionary",
    "BAJAJ-AUTO": "Consumer Discretionary",
    "TITAN": "Consumer Discretionary",
    "ASIANPAINT": "Materials",
    "ULTRACEMCO": "Materials",
    "GRASIM": "Materials",
    "TATASTEEL": "Materials",
    "JSWSTEEL": "Materials",
    "HINDALCO": "Materials",
    "SUNPHARMA": "Health Care",
    "DRREDDY": "Health Care",
    "CIPLA": "Health Care",
    "DIVISLAB": "Health Care",
    "APOLLOHOSP": "Health Care",
    "BHARTIARTL": "Communication Services",
    "LT": "Industrials",
    "ADANIENT": "Industrials",
    "ADANIPORTS": "Industrials",
}

NIFTY50 = list(NIFTY50_SECTORS.keys())


def yf_symbols(suffix=".NS"):
    # returning the yfinance-form tickers for a batch download
    return [t + suffix for t in NIFTY50]
