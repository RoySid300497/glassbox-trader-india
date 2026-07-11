"""holding every path, constant, and feature list in one place"""

import os

# resolving the base folder from the environment so machines can differ
BASE_PATH = os.environ.get("STOCK_LENS_BASE", os.path.abspath("./stock-lens"))
DATA_PATH = os.path.join(BASE_PATH, "data")
MODEL_PATH = os.path.join(BASE_PATH, "models")
OBS_PATH = os.path.join(BASE_PATH, "observations")
for _p in (DATA_PATH, MODEL_PATH, OBS_PATH):
    os.makedirs(_p, exist_ok=True)

KAGGLE_DATASET = "dgawlik/nyse"
KAGGLE_FILES = ["prices-split-adjusted.csv", "fundamentals.csv",
                "securities.csv", "prices.csv"]

FUNDAMENTAL_COLS = ["Ticker Symbol", "Period Ending", "Earnings Per Share",
                    "Total Revenue", "Net Income", "Total Assets",
                    "Total Liabilities", "Profit Margin", "Total Equity",
                    "Operating Margin", "Current Ratio"]
FUND_FILL_COLS = FUNDAMENTAL_COLS[2:]

BASE_FEATURE_COLS = ["open", "high", "low", "close", "volume",
                     "ma10", "ma30", "ma50", "rsi", "vol_ratio",
                     "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
                     "BBU_20_2.0_2.0", "BBL_20_2.0_2.0", "BBM_20_2.0_2.0",
                     "Earnings Per Share", "Total Revenue", "Net Income",
                     "Total Assets", "Total Liabilities", "Profit Margin",
                     "Total Equity", "Operating Margin", "Current Ratio"]

LAG_COLS = ["close", "rsi", "MACD_12_26_9", "MACDh_12_26_9", "vol_ratio"]
LAG_DAYS = [1, 2, 3, 5]
RETURN_FEATURES = ["return_1d", "return_3d", "return_5d", "return_10d"]
RELATIVE_FEATURES = ["market_return", "rel_to_market",
                     "sector_return", "rel_to_sector"]

HORIZONS = {"1d": {"days": 1, "threshold": 0.01},
            "5d": {"days": 5, "threshold": 0.02},
            "10d": {"days": 10, "threshold": 0.03}}

TRAIN_END = "2015-01-01"
VAL_END = "2016-01-01"
RANDOM_STATE = 42

SEQ_WINDOW = 30
SEQ_EPOCHS = 15
SEQ_BATCH = 256
SEQ_THRESHOLD = 0.01
PER_STOCK_SAMPLE = 40

OOS_START = "2016-06-01"
OOS_EVAL_FROM = "2017-01-01"
OOS_END = "2025-12-31"
