"""refreshing the scan universe with the current s&p 500 constituents"""

import os
import sys
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import requests
from core.config import DATA_PATH

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {"User-Agent": "glassbox-trader universe refresh "
                         "(github.com/SidRoy97/glassbox-trader)"}


def fetch_current_constituents():
    # scraping the constituents table from wikipedia with a polite user agent
    r = requests.get(WIKI_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if "symbol" in cols and any("gics" in c for c in cols):
            t.columns = [str(c) for c in t.columns]
            return t
    raise RuntimeError("constituents table not found on the wikipedia page")


def main():
    table = fetch_current_constituents()
    sector_col = next(c for c in table.columns if "GICS Sector" in c)
    out = pd.DataFrame({
        "Ticker symbol": table["Symbol"].astype(str).str.strip().str.upper(),
        "GICS Sector": table[sector_col].astype(str).str.strip(),
    }).drop_duplicates("Ticker symbol").sort_values("Ticker symbol")

    old_path = os.path.join(DATA_PATH, "securities.csv")
    new_path = os.path.join(DATA_PATH, "universe.csv")
    out.to_csv(new_path, index=False)

    # reporting the drift against the 2016 training roster for the record
    if os.path.exists(old_path):
        old = set(pd.read_csv(old_path)["Ticker symbol"].str.upper())
        new = set(out["Ticker symbol"])
        print(f"universe.csv written: {len(new)} tickers")
        print(f"added since 2016 roster   : {len(new - old)} "
              f"(e.g. {sorted(new - old)[:8]})")
        print(f"dropped since 2016 roster : {len(old - new)} "
              f"(e.g. {sorted(old - new)[:8]})")
    else:
        print(f"universe.csv written: {len(out)} tickers")
    print(f"saved to {new_path}")


if __name__ == "__main__":
    main()
