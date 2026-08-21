"""refreshing the scan universe with the current nifty constituents

default index is now the NIFTY 200 (UNIVERSE_INDEX=nifty50 to revert):
a 50-name universe starves the screener, the exploration wildcards, and
any cross-sectional statistic. training remains valid — models already
run pure OOS inference on names outside their training roster, exactly
as the US port did after its universe refresh.

fallback chain: NSE published CSV for the chosen index -> the NSE
nifty 50 CSV -> the hardcoded universe.py map, so a bad fetch never
wipes the tradeable set. sectors come from the static nifty-50 map when
known, else from NSE's own Industry column mapped onto the GICS-style
buckets the feature pipeline and sector indices use.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from core.config import DATA_PATH
from pipeline.universe import NIFTY50_SECTORS

INDEX_URLS = {
    "nifty200": ("https://archives.nseindia.com/content/indices/"
                 "ind_nifty200list.csv"),
    "nifty50": ("https://archives.nseindia.com/content/indices/"
                "ind_nifty50list.csv"),
}
MIN_ROWS = {"nifty200": 150, "nifty50": 40}

# NSE "Industry" values -> the GICS-style buckets our pipeline and the
# SECTOR_INDEX map in live_features understand
INDUSTRY_TO_SECTOR = {
    "financial services": "Financials",
    "information technology": "Information Technology",
    "oil gas & consumable fuels": "Energy",
    "oil, gas & consumable fuels": "Energy",
    "fast moving consumer goods": "Consumer Staples",
    "healthcare": "Health Care",
    "automobile and auto components": "Consumer Discretionary",
    "metals & mining": "Materials",
    "construction": "Industrials",
    "construction materials": "Materials",
    "capital goods": "Industrials",
    "power": "Utilities",
    "telecommunication": "Communication Services",
    "consumer services": "Consumer Discretionary",
    "consumer durables": "Consumer Discretionary",
    "chemicals": "Materials",
    "services": "Industrials",
    "realty": "Real Estate",
    "media entertainment & publication": "Communication Services",
    "media, entertainment & publication": "Communication Services",
    "textiles": "Consumer Discretionary",
    "diversified": "Industrials",
    "forest materials": "Materials",
}


def _map_sector(symbol, industry):
    # the curated nifty-50 map wins for names it knows; NSE's industry
    # column covers the rest; anything unmapped lands in Industrials
    # (broadest bucket) rather than an "Unknown" the sector merge drops
    if symbol in NIFTY50_SECTORS:
        return NIFTY50_SECTORS[symbol]
    return INDUSTRY_TO_SECTOR.get(str(industry).strip().lower(),
                                  "Industrials")


def fetch_current_constituents():
    import requests
    from io import StringIO

    index = os.environ.get("UNIVERSE_INDEX", "nifty200").lower()
    if index not in INDEX_URLS:
        index = "nifty200"
    headers = {"User-Agent": "Mozilla/5.0 glassbox-india universe refresh"}
    tried = [index] + [i for i in ("nifty200", "nifty50") if i != index]
    for idx in tried:
        try:
            r = requests.get(INDEX_URLS[idx], headers=headers, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))
            df.columns = [c.strip() for c in df.columns]
            sym_col = next(c for c in df.columns if c.lower() == "symbol")
            ind_col = next((c for c in df.columns
                            if c.lower() == "industry"), None)
            out = pd.DataFrame({
                "Ticker symbol": df[sym_col].astype(str).str.strip()
                .str.upper(),
            })
            # NSE lists occasionally carry DUMMY placeholder rows for
            # pending rebalances — they are not tradeable symbols
            out = out[~out["Ticker symbol"].str.startswith("DUMMY")]
            out = out[out["Ticker symbol"].str.fullmatch(r"[A-Z0-9&\-]+")]
            industries = (df[ind_col] if ind_col is not None
                          else pd.Series([""] * len(df)))
            out["GICS Sector"] = [
                _map_sector(s, i) for s, i in
                zip(out["Ticker symbol"],
                    industries.reindex(out.index, fill_value=""))]
            out = out.dropna(subset=["Ticker symbol"]).drop_duplicates(
                "Ticker symbol").sort_values("Ticker symbol")
            if len(out) >= MIN_ROWS[idx]:
                print(f"fetched {idx}: {len(out)} constituents")
                return out
            raise RuntimeError(f"{idx} list too short: {len(out)}")
        except Exception as e:
            print(f"{idx} fetch failed ({e}); trying next fallback")
    print("all index fetches failed; using static universe map")
    return pd.DataFrame(
        [{"Ticker symbol": t, "GICS Sector": s}
         for t, s in NIFTY50_SECTORS.items()]
    ).sort_values("Ticker symbol")


def main():
    table = fetch_current_constituents()
    old_path = os.path.join(DATA_PATH, "securities.csv")
    new_path = os.path.join(DATA_PATH, "universe.csv")
    # reporting the drift so a rebalance is visible in the CI log
    if os.path.exists(new_path):
        try:
            prev = set(pd.read_csv(new_path)["Ticker symbol"]
                       .astype(str).str.upper())
            now = set(table["Ticker symbol"])
            if prev != now:
                print(f"added  : {sorted(now - prev)[:12]}"
                      f"{' ...' if len(now - prev) > 12 else ''}")
                print(f"dropped: {sorted(prev - now)[:12]}"
                      f"{' ...' if len(prev - now) > 12 else ''}")
        except Exception:
            pass
    table.to_csv(new_path, index=False)
    # keep securities.csv in sync so the sector merge always resolves
    table.to_csv(old_path, index=False)
    print(f"universe refreshed: {len(table)} constituents -> {new_path}")


if __name__ == "__main__":
    main()
