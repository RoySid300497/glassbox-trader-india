"""adding lagged, return, relative features and multi-horizon labels"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from core.config import (DATA_PATH, LAG_COLS, LAG_DAYS, RETURN_FEATURES,
                    HORIZONS, FUND_FILL_COLS)
from core.helpers import log, save_plot, section


def add_lags(group):
    # shifting key indicators backward so trees can see recent history
    group = group.copy()
    for col in LAG_COLS:
        for lag in LAG_DAYS:
            group[f"{col}_lag{lag}"] = group[col].shift(lag)
    return group


def add_returns(group):
    # computing trailing multi-day returns as momentum features
    group = group.copy()
    group["return_1d"] = group["close"].pct_change(1)
    group["return_3d"] = group["close"].pct_change(3)
    group["return_5d"] = group["close"].pct_change(5)
    group["return_10d"] = group["close"].pct_change(10)
    return group


def add_horizon_labels(group):
    # building direction labels for each forward horizon
    group = group.copy()
    for name, cfg in HORIZONS.items():
        fwd = group["close"].pct_change(cfg["days"]).shift(-cfg["days"])
        group[f"fwd_return_{name}"] = fwd
        thr = cfg["threshold"]
        group[f"label_{name}"] = fwd.apply(
            lambda x: "Up" if x > thr else ("Down" if x < -thr else "Neutral"))
    return group


def stage_2b_enhanced():
    section("STAGE 2B — ENHANCED FEATURES + MULTI-HORIZON LABELS")
    master = pd.read_csv(os.path.join(DATA_PATH, "master.csv"),
                         parse_dates=["date"])
    master = master.sort_values(["symbol", "date"]).reset_index(drop=True)

    # applying lag, return, and horizon transforms per ticker
    log("adding lagged features...")
    master = pd.concat([add_lags(g) for _, g in
                        master.groupby("symbol", sort=False)],
                       ignore_index=True)
    log("adding return features...")
    master = pd.concat([add_returns(g) for _, g in
                        master.groupby("symbol", sort=False)],
                       ignore_index=True)

    # computing market-relative and sector-relative performance
    master["market_return"] = master.groupby("date")["return_1d"] \
        .transform("mean")
    master["rel_to_market"] = master["return_1d"] - master["market_return"]
    master["sector_return"] = master.groupby(["date", "sector"])["return_1d"] \
        .transform("mean")
    master["rel_to_sector"] = master["return_1d"] - master["sector_return"]

    log("adding multi-horizon labels...")
    master = pd.concat([add_horizon_labels(g) for _, g in
                        master.groupby("symbol", sort=False)],
                       ignore_index=True)

    # plotting label balance for each horizon
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, name in zip(axes, HORIZONS):
        counts = master[f"label_{name}"].value_counts()
        sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                    palette=["tomato", "steelblue", "seagreen"],
                    legend=False, ax=ax)
        ax.set_title(f"label_{name}")
    save_plot("s2b_horizon_label_distributions.png")

    # imputing fundamentals with ffill, bfill, then column median
    fund_present = [c for c in FUND_FILL_COLS if c in master.columns]
    master = master.sort_values(["symbol", "date"])
    master[fund_present] = master.groupby("symbol")[fund_present].ffill()
    master[fund_present] = master.groupby("symbol")[fund_present].bfill()
    for c in fund_present:
        master[c] = master[c].fillna(master[c].median())

    # dropping rows only where truly unfixable columns are null
    lag_cols = [f"{c}_lag{l}" for c in LAG_COLS for l in LAG_DAYS]
    essential = (lag_cols + RETURN_FEATURES
                 + [f"label_{h}" for h in HORIZONS]
                 + [f"fwd_return_{h}" for h in HORIZONS])
    essential = [c for c in essential if c in master.columns]
    before = master.shape[0]
    master = master.dropna(subset=essential).reset_index(drop=True)
    log(f"dropped {before - master.shape[0]:,} rows with nulls in essentials")
    log(f"enhanced master shape: {master.shape}")

    master.to_csv(os.path.join(DATA_PATH, "master_enhanced.csv"), index=False)
    log("stage 2b complete")
    return master
