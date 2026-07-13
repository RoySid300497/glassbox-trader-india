"""providing logging, plot saving, and section banners for every stage"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core.config import OBS_PATH


def log(msg):
    # printing to console and appending to the persistent run log
    print(msg)
    with open(os.path.join(OBS_PATH, "run_log.txt"), "a") as f:
        f.write(str(msg) + "\n")


def save_plot(name):
    # saving the current figure to the observations folder
    plt.savefig(os.path.join(OBS_PATH, name), dpi=120, bbox_inches="tight")
    plt.close()
    log(f"  [plot saved] {name}")


def section(title):
    # printing a visible banner so long logs stay readable
    bar = "=" * 70
    log(f"\n{bar}\n{title}\n{bar}")


def yf_symbol(ticker):
    # forming the correct yfinance symbol for a ticker. NSE names need the
    # exchange suffix (RELIANCE -> RELIANCE.NS); the old replace(".", "-")
    # was a US share-class convention that stripped the suffix and made every
    # india download fail with "no timezone found"
    try:
        from core.config import EXCHANGE_SUFFIX
    except Exception:
        EXCHANGE_SUFFIX = ""
    t = str(ticker)
    if EXCHANGE_SUFFIX:
        return t if t.endswith(EXCHANGE_SUFFIX) else t + EXCHANGE_SUFFIX
    # no suffix configured (US): use the share-class dash convention
    return t.replace(".", "-")
