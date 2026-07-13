"""a shared, rate-limit-resistant yfinance session

yahoo throttles requests from shared cloud IPs (like GitHub Actions runners),
which surfaces as the misleading "no timezone found; possibly delisted" error
even for valid tickers. impersonating a real browser via curl_cffi sharply
reduces this throttling. every engine download should route through
get_yf_session() so the whole system shares one resilient session.
"""

import time


_session = None


def get_yf_session():
    # building a browser-impersonating session once and reusing it; falling
    # back to None (yfinance's default session) if curl_cffi isn't installed
    global _session
    if _session is not None:
        return _session
    try:
        from curl_cffi import requests as cffi_requests
        _session = cffi_requests.Session(impersonate="chrome")
    except Exception:
        _session = None
    return _session


def yf_download(*args, retries=3, backoff=2.0, **kwargs):
    # a resilient wrapper around yf.download: routes through the impersonating
    # session and retries with backoff on empty results (the signature of a
    # transient rate-limit rather than a truly delisted ticker)
    import yfinance as yf
    sess = get_yf_session()
    if sess is not None and "session" not in kwargs:
        kwargs["session"] = sess
    last = None
    for attempt in range(retries):
        try:
            df = yf.download(*args, **kwargs)
            if df is not None and not df.empty:
                return df
            last = df
        except Exception as e:
            last = e
        # transient empty/blocked — wait and retry with growing backoff
        time.sleep(backoff * (attempt + 1))
    # returning whatever we last got (possibly empty) so callers' guards fire
    import pandas as pd
    return last if isinstance(last, pd.DataFrame) else pd.DataFrame()
