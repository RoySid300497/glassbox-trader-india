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
