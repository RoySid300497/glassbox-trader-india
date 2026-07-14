"""refreshing the scan universe with the current nifty 50 constituents

pulls the live nifty 50 list; falls back to the hardcoded universe.py list
if the network fetch fails, so a bad fetch never wipes the tradeable set
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from core.config import DATA_PATH
from pipeline.universe import NIFTY50_SECTORS


def fetch_current_constituents():
    # trying NSE's published nifty 50 CSV first; wikipedia as a backup;
    # the static map as the final fallback so we always have a universe
    import requests
    from io import StringIO

    nse_url = ("https://archives.nseindia.com/content/indices/"
               "ind_nifty50list.csv")
    headers = {"User-Agent": "Mozilla/5.0 glassbox-india universe refresh"}
    try:
        r = requests.get(nse_url, headers=headers, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        # NSE csv has Symbol and Industry columns
        sym_col = next(c for c in df.columns if c.strip().lower() == "symbol")
        out = pd.DataFrame({
            "Ticker symbol": df[sym_col].astype(str).str.strip().str.upper(),
        })
        # map to our GICS-style sectors; unknown names fall back to Industry
        out["GICS Sector"] = out["Ticker symbol"].map(NIFTY50_SECTORS)
        out["GICS Sector"] = out["GICS Sector"].fillna("Unknown")
        out = out.dropna(subset=["Ticker symbol"]).drop_duplicates(
            "Ticker symbol").sort_values("Ticker symbol")
        if len(out) >= 40:
            return out
        raise RuntimeError(f"NSE list too short: {len(out)}")
    except Exception as e:
        print(f"NSE fetch failed ({e}); using static universe map")
        return pd.DataFrame(
            [{"Ticker symbol": t, "GICS Sector": s}
             for t, s in NIFTY50_SECTORS.items()]
        ).sort_values("Ticker symbol")


def main():
    table = fetch_current_constituents()
    old_path = os.path.join(DATA_PATH, "securities.csv")
    new_path = os.path.join(DATA_PATH, "universe.csv")
    table.to_csv(new_path, index=False)
    # keep securities.csv in sync so the sector merge always resolves
    table.to_csv(old_path, index=False)
    print(f"universe refreshed: {len(table)} constituents -> {new_path}")


if __name__ == "__main__":
    main()
