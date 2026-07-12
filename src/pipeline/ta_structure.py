"""Structure-based technical features for glassbox-trader.

Lives at src/pipeline/ta_structure.py, imported as
`from pipeline.ta_structure import build_structure_features`.

Implementing the mechanically computable concepts distilled from the
strategy videos: EMA regime and pullback state, fair value gaps and
order-block zones, break-of-structure / change-of-character tracking,
ADX trend strength, candle anatomy, and chandelier trailing levels.
Every function is causal: row i only uses information available at bar i,
matching the leak-safe discipline of the rest of the pipeline.

Two consumers:
  1. build_structure_features(df) -> 22 feature columns for the model
     roster (gate through retrain challengers, never hot-swap).
  2. technical_structure_block(df) -> compact citable dict for the
     evidence packet in engine/data_packet.py.
"""

import numpy as np
import pandas as pd

EMA_SPAN = 50
ATR_SPAN = 22
SWING_WINDOW = 5
CHANDELIER_MULT = 3.0

STRUCTURE_FEATURE_VERSION = "ta_structure_v3.4"


def _atr(df: pd.DataFrame, span: int = ATR_SPAN) -> pd.Series:
    # Computing average true range for volatility normalization across tickers
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=span, adjust=False).mean()


def ema_regime_features(df: pd.DataFrame, span: int = EMA_SPAN) -> pd.DataFrame:
    # Capturing which side of the 50 EMA price sits on and how mature the regime is
    out = pd.DataFrame(index=df.index)
    ema = df["close"].ewm(span=span, adjust=False).mean()
    atr = _atr(df)

    out["ema_dist_atr"] = (df["close"] - ema) / atr
    above = (df["close"] > ema).astype(int)

    # Counting bars elapsed since the most recent EMA cross in either direction
    cross = above.diff().fillna(0).ne(0)
    group = cross.cumsum()
    out["bars_since_ema_cross"] = above.groupby(group).cumcount()
    out["above_ema"] = above

    # Flagging exhaustion breakouts where the candle body dwarfs recent bodies
    body = (df["close"] - df["open"]).abs()
    avg_body = body.rolling(20, min_periods=5).mean()
    out["body_vs_avg"] = body / avg_body.replace(0, np.nan)
    out["exhaustion_breakout"] = (out["body_vs_avg"] > 3.0).astype(int)
    return out


def fair_value_gaps(df: pd.DataFrame, max_zones: int = 20) -> pd.DataFrame:
    # Detecting three-candle gaps and measuring distance to the nearest live zone
    out = pd.DataFrame(index=df.index)
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    atr = _atr(df).to_numpy()
    n = len(df)

    bull_dist = np.full(n, np.nan)
    bear_dist = np.full(n, np.nan)
    bull_count = np.zeros(n)
    bear_count = np.zeros(n)
    bull_zones: list[tuple[float, float]] = []
    bear_zones: list[tuple[float, float]] = []

    for i in range(n):
        # Registering a gap only once its third candle has fully closed
        if i >= 2 and highs[i - 2] < lows[i]:
            bull_zones.append((highs[i - 2], lows[i]))
        if i >= 2 and lows[i - 2] > highs[i]:
            bear_zones.append((highs[i], lows[i - 2]))

        # Retiring any zone the current bar has traded into, keeping zones one-time-use
        bull_zones = [z for z in bull_zones if lows[i] > z[1]][-max_zones:]
        bear_zones = [z for z in bear_zones if highs[i] < z[0]][-max_zones:]

        bull_count[i] = len(bull_zones)
        bear_count[i] = len(bear_zones)
        if bull_zones and atr[i] > 0:
            bull_dist[i] = (closes[i] - max(z[1] for z in bull_zones)) / atr[i]
        if bear_zones and atr[i] > 0:
            bear_dist[i] = (min(z[0] for z in bear_zones) - closes[i]) / atr[i]

    out["fvg_bull_dist_atr"] = bull_dist
    out["fvg_bear_dist_atr"] = bear_dist
    out["fvg_bull_open"] = bull_count
    out["fvg_bear_open"] = bear_count
    return out


def market_structure(df: pd.DataFrame, window: int = SWING_WINDOW) -> pd.DataFrame:
    # Tracking swing breaks so the model knows if trend continued or flipped
    out = pd.DataFrame(index=df.index)
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    n = len(df)

    struct_state = np.zeros(n)
    bars_since_break = np.full(n, np.nan)
    last_swing_high = np.nan
    last_swing_low = np.nan
    trend = 0
    last_break = -1

    for i in range(n):
        # Confirming a swing only after `window` later bars exist, avoiding lookahead
        j = i - window
        if j >= window:
            seg_h = highs[j - window : j + window + 1]
            seg_l = lows[j - window : j + window + 1]
            if highs[j] == seg_h.max():
                last_swing_high = highs[j]
            if lows[j] == seg_l.min():
                last_swing_low = lows[j]

        # Labeling closes through prior swings as continuation or character change
        if not np.isnan(last_swing_high) and closes[i] > last_swing_high:
            trend, last_break = 1, i
            last_swing_high = np.nan
        elif not np.isnan(last_swing_low) and closes[i] < last_swing_low:
            trend, last_break = -1, i
            last_swing_low = np.nan

        struct_state[i] = trend
        if last_break >= 0:
            bars_since_break[i] = i - last_break

    out["structure_trend"] = struct_state
    out["bars_since_structure_break"] = bars_since_break
    return out


def trend_strength(df: pd.DataFrame, span: int = 14) -> pd.DataFrame:
    # Measuring directional conviction with a Wilder-style ADX proxy
    out = pd.DataFrame(index=df.index)
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = _atr(df, span)

    plus_di = 100 * plus_dm.ewm(span=span, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=span, adjust=False).mean() / atr
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    out["adx"] = dx.ewm(span=span, adjust=False).mean()
    out["di_spread"] = plus_di - minus_di
    return out


def candle_anatomy(df: pd.DataFrame) -> pd.DataFrame:
    # Encoding the strength / control-shift / indecision taxonomy as ratios
    out = pd.DataFrame(index=df.index)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body = df["close"] - df["open"]

    out["body_ratio"] = body / rng
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    out["wick_asymmetry"] = (lower_wick - upper_wick) / rng
    out["is_doji"] = (body.abs() / rng < 0.1).astype(int)

    # Flagging engulfing bars where one body swallows the previous body entirely
    prev_body_hi = df[["open", "close"]].max(axis=1).shift(1)
    prev_body_lo = df[["open", "close"]].min(axis=1).shift(1)
    out["bull_engulf"] = (
        (body > 0)
        & (df["close"] >= prev_body_hi)
        & (df["open"] <= prev_body_lo)
    ).astype(int)
    out["bear_engulf"] = (
        (body < 0)
        & (df["open"] >= prev_body_hi)
        & (df["close"] <= prev_body_lo)
    ).astype(int)

    # Comparing advance speed against retrace speed over a rolling window
    ret = df["close"].pct_change()
    up_speed = ret.clip(lower=0).rolling(20, min_periods=10).mean()
    down_speed = (-ret.clip(upper=0)).rolling(20, min_periods=10).mean()
    out["momentum_asymmetry"] = (up_speed - down_speed) / (
        up_speed + down_speed
    ).replace(0, np.nan)
    return out


def chandelier_exit(
    df: pd.DataFrame, span: int = ATR_SPAN, mult: float = CHANDELIER_MULT
) -> pd.DataFrame:
    # Producing ATR-trailed stop levels for the Alpaca execution layer
    out = pd.DataFrame(index=df.index)
    atr = _atr(df, span)
    out["chandelier_long"] = df["high"].rolling(span, min_periods=1).max() - mult * atr
    out["chandelier_short"] = df["low"].rolling(span, min_periods=1).min() + mult * atr
    out["long_stop_dist_atr"] = (df["close"] - out["chandelier_long"]) / atr
    return out


def _rsi(close: pd.Series, span: int = 14) -> pd.Series:
    # computing wilder rsi for the divergence detector
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / span, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / span, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def divergence_features(df: pd.DataFrame, window: int = SWING_WINDOW,
                        recency: int = 40) -> pd.DataFrame:
    # detecting rsi divergences only at pivots already confirmed by later bars
    out = pd.DataFrame(index=df.index)
    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    rsi = _rsi(df["close"]).to_numpy()
    n = len(df)

    bull_since = np.full(n, np.nan)
    bear_since = np.full(n, np.nan)
    lo_pivots: list[tuple[int, float, float]] = []
    hi_pivots: list[tuple[int, float, float]] = []
    last_bull = -1
    last_bear = -1

    for i in range(n):
        # confirming a pivot only once `window` later bars exist, so the
        # divergence signal is attributed at confirmation time, never earlier
        j = i - window
        if j >= window:
            seg_l = lows[j - window : j + window + 1]
            seg_h = highs[j - window : j + window + 1]
            if lows[j] == seg_l.min() and not np.isnan(rsi[j]):
                if lo_pivots:
                    pj, plow, prsi = lo_pivots[-1]
                    # flagging bullish divergence: lower price low, higher rsi low
                    if lows[j] < plow and rsi[j] > prsi and j - pj <= recency:
                        last_bull = i
                lo_pivots.append((j, lows[j], rsi[j]))
                lo_pivots = lo_pivots[-3:]
            if highs[j] == seg_h.max() and not np.isnan(rsi[j]):
                if hi_pivots:
                    pj, phigh, prsi = hi_pivots[-1]
                    # flagging bearish divergence: higher price high, lower rsi high
                    if highs[j] > phigh and rsi[j] < prsi and j - pj <= recency:
                        last_bear = i
                hi_pivots.append((j, highs[j], rsi[j]))
                hi_pivots = hi_pivots[-3:]

        if last_bull >= 0:
            bull_since[i] = i - last_bull
        if last_bear >= 0:
            bear_since[i] = i - last_bear

    out["bars_since_bull_divergence"] = bull_since
    out["bars_since_bear_divergence"] = bear_since
    out["bull_divergence_active"] = (np.nan_to_num(bull_since, nan=1e9)
                                     <= 10).astype(int)
    out["bear_divergence_active"] = (np.nan_to_num(bear_since, nan=1e9)
                                     <= 10).astype(int)
    return out


def volume_profile_features(df: pd.DataFrame, lookback: int = 120,
                            bins: int = 24) -> pd.DataFrame:
    # approximating volume-at-price from daily bars by binning typical price
    out = pd.DataFrame(index=df.index)
    tp = ((df["high"] + df["low"] + df["close"]) / 3).to_numpy()
    vol = df["volume"].to_numpy(dtype=float) if "volume" in df.columns \
        else np.ones(len(df))
    closes = df["close"].to_numpy()
    atr = _atr(df).to_numpy()
    n = len(df)

    poc_dist = np.full(n, np.nan)
    above_poc = np.full(n, np.nan)
    for i in range(lookback, n):
        w_tp = tp[i - lookback : i]
        w_v = vol[i - lookback : i]
        hist, edges = np.histogram(w_tp, bins=bins, weights=w_v)
        poc = (edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2
        if atr[i] > 0:
            poc_dist[i] = (closes[i] - poc) / atr[i]
        above_poc[i] = 1.0 if closes[i] >= poc else 0.0

    out["poc_dist_atr"] = poc_dist
    out["above_poc"] = above_poc
    return out


def liquidity_sweeps(df: pd.DataFrame, window: int = SWING_WINDOW,
                     tol_atr: float = 0.25, memory: int = 60) -> pd.DataFrame:
    # flagging stop-hunt sweeps: equal-level clusters wicked through and reclaimed
    out = pd.DataFrame(index=df.index)
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    atr = _atr(df).to_numpy()
    n = len(df)

    lo_pivots: list[tuple[int, float]] = []
    hi_pivots: list[tuple[int, float]] = []
    lo_levels: list[tuple[int, float]] = []
    hi_levels: list[tuple[int, float]] = []
    bull_since = np.full(n, np.nan)
    bear_since = np.full(n, np.nan)
    last_bull = -1
    last_bear = -1

    for i in range(n):
        # confirming pivots only after `window` later bars exist
        j = i - window
        if j >= window and atr[i] > 0:
            if lows[j] == lows[j - window : j + window + 1].min():
                # pairing near-equal lows into a resting-liquidity level
                for pj, plow in lo_pivots:
                    if i - pj <= memory and abs(lows[j] - plow) <=                             tol_atr * atr[i]:
                        lo_levels.append((i, min(lows[j], plow)))
                lo_pivots.append((j, lows[j]))
                lo_pivots = lo_pivots[-6:]
            if highs[j] == highs[j - window : j + window + 1].max():
                for pj, phigh in hi_pivots:
                    if i - pj <= memory and abs(highs[j] - phigh) <=                             tol_atr * atr[i]:
                        hi_levels.append((i, max(highs[j], phigh)))
                hi_pivots.append((j, highs[j]))
                hi_pivots = hi_pivots[-6:]

        # sweep: trading through the level yet closing back on the right side
        still_lo = []
        for born, level in lo_levels[-6:]:
            if i <= born:
                still_lo.append((born, level))
            elif lows[i] < level and closes[i] > level:
                last_bull = i
            elif closes[i] < level - 0.5 * atr[i]:
                continue  # genuinely broken, not swept — retiring the level
            else:
                still_lo.append((born, level))
        lo_levels = still_lo
        still_hi = []
        for born, level in hi_levels[-6:]:
            if i <= born:
                still_hi.append((born, level))
            elif highs[i] > level and closes[i] < level:
                last_bear = i
            elif closes[i] > level + 0.5 * atr[i]:
                continue
            else:
                still_hi.append((born, level))
        hi_levels = still_hi

        if last_bull >= 0:
            bull_since[i] = i - last_bull
        if last_bear >= 0:
            bear_since[i] = i - last_bear

    out["bars_since_bull_sweep"] = bull_since
    out["bars_since_bear_sweep"] = bear_since
    return out


def vix_fix(df: pd.DataFrame, lookback: int = 22,
            band: int = 20) -> pd.DataFrame:
    # computing the williams vix fix and flagging volatility capitulation
    out = pd.DataFrame(index=df.index)
    hh_close = df["close"].rolling(lookback, min_periods=lookback).max()
    wvf = 100 * (hh_close - df["low"]) / hh_close
    upper = wvf.rolling(band, min_periods=band).mean()         + 2 * wvf.rolling(band, min_periods=band).std()
    cap = (wvf >= upper).astype(int)
    out["wvf"] = wvf
    grp = cap.eq(1).cumsum()
    since = cap.groupby(grp).cumcount().where(grp > 0)
    out["bars_since_capitulation"] = since.astype(float)
    return out


def first_pullback(df: pd.DataFrame) -> pd.DataFrame:
    # flagging the first retrace after a fresh structure break in either side
    out = pd.DataFrame(index=df.index)
    ms = market_structure(df)
    atr = _atr(df)
    closes = df["close"]

    post_hi = closes.copy()
    post_lo = closes.copy()
    brk = ms["bars_since_structure_break"]
    # tracking the running extreme since the most recent structure break
    seg = (brk == 0).cumsum()
    post_hi = df["high"].groupby(seg).cummax()
    post_lo = df["low"].groupby(seg).cummin()

    dd = (post_hi - closes) / atr
    du = (closes - post_lo) / atr
    fresh = brk.between(3, 20)
    out["first_pullback_long"] = ((ms["structure_trend"] == 1) & fresh
                                  & dd.between(0.8, 3.0)).astype(int)
    out["first_pullback_short"] = ((ms["structure_trend"] == -1) & fresh
                                   & du.between(0.8, 3.0)).astype(int)
    return out


def range_regime(df: pd.DataFrame, lookback: int = 40,
                 squeeze_hist: int = 120) -> pd.DataFrame:
    # describing ranging markets and volatility compression before breakouts
    out = pd.DataFrame(index=df.index)
    ts = trend_strength(df)
    ms = market_structure(df)

    hi = df["high"].rolling(lookback, min_periods=10).max()
    lo = df["low"].rolling(lookback, min_periods=10).min()
    width = (hi - lo).replace(0, np.nan)
    out["position_in_range"] = (df["close"] - lo) / width
    out["ranging"] = ((ts["adx"] < 20)
                      & (ms["bars_since_structure_break"].isna()
                         | (ms["bars_since_structure_break"] > 20)))         .astype(int)

    sma20 = df["close"].rolling(20, min_periods=20).mean()
    std20 = df["close"].rolling(20, min_periods=20).std()
    bbw = (4 * std20) / sma20
    pct = bbw.rolling(squeeze_hist, min_periods=40)         .apply(lambda w: (w[-1] >= w).mean() if len(w) else np.nan, raw=True)
    out["bb_width_pctile"] = pct
    out["bb_squeeze"] = (pct <= 0.20).astype(int)
    return out


def inside_bars(df: pd.DataFrame) -> pd.DataFrame:
    # flagging bars contained by the prior bar's range, and coiling streaks
    out = pd.DataFrame(index=df.index)
    ib = ((df["high"] < df["high"].shift(1))
          & (df["low"] > df["low"].shift(1))).astype(int)
    out["inside_bar"] = ib
    grp = (ib == 0).cumsum()
    out["inside_bar_streak"] = ib.groupby(grp).cumsum()
    return out


def prior_day_levels(df: pd.DataFrame) -> pd.DataFrame:
    # positioning today's open and close against the prior day's range, and
    # flagging overnight gaps that break or reclaim the prior day's extremes
    # — a pre-open structure read the daily cadence can act on
    out = pd.DataFrame(index=df.index)
    pdh = df["high"].shift(1)
    pdl = df["low"].shift(1)
    pdc = df["close"].shift(1)
    rng = (pdh - pdl).replace(0, pd.NA)
    gap = (df["open"] - pdc) / rng
    out["gap_vs_prior_range"] = gap.astype(float).fillna(0.0)
    out["gap_above_prior_high"] = (df["open"] > pdh).astype(int)
    out["gap_below_prior_low"] = (df["open"] < pdl).astype(int)
    out["close_above_prior_high"] = (df["close"] > pdh).astype(int)
    out["close_below_prior_low"] = (df["close"] < pdl).astype(int)
    out["failed_gap_up"] = ((df["open"] > pdh) &
                            (df["close"] < pdh)).astype(int)
    out["failed_gap_down"] = ((df["open"] < pdl) &
                              (df["close"] > pdl)).astype(int)
    pos = (df["close"] - pdl) / rng
    out["close_in_prior_range"] = pos.astype(float).clip(-1, 2).fillna(0.5)
    return out


def turn_days(df: pd.DataFrame) -> pd.DataFrame:
    # flagging the first opposite-color close after a strong directional
    # run — the classic short entry on parabolic names, and its mirror
    out = pd.DataFrame(index=df.index)
    green = (df["close"] > df["open"]).astype(int)
    red = 1 - green
    g_run = green.groupby((green == 0).cumsum()).cumsum()
    r_run = red.groupby((red == 0).cumsum()).cumsum()
    ret4 = df["close"].pct_change(4)
    strong_up = (g_run.shift(1) >= 3) | (ret4.shift(1) > 0.06)
    strong_dn = (r_run.shift(1) >= 3) | (ret4.shift(1) < -0.06)
    out["first_red_after_run"] = ((red == 1) & strong_up).astype(int)
    out["first_green_after_run"] = ((green == 1) & strong_dn).astype(int)
    return out


def trend_volume_extras(df: pd.DataFrame) -> pd.DataFrame:
    # adding the long-term 200 ema regime and a fast/slow volume oscillator
    out = pd.DataFrame(index=df.index)
    ema200 = df["close"].ewm(span=200, adjust=False).mean()
    out["above_ema200"] = (df["close"] > ema200).astype(int)
    out["ema200_slope"] = ema200.diff(20)

    # flagging the connors-style setup: deep short-term fear in an uptrend
    rsi2 = _rsi(df["close"], span=2)
    out["rsi2"] = rsi2
    out["connors_pullback_long"] = ((df["close"] > ema200)
                                    & (rsi2 < 10)).astype(int)
    if "volume" in df.columns:
        v5 = df["volume"].rolling(5, min_periods=5).mean()
        v20 = df["volume"].rolling(20, min_periods=20).mean()
        out["volume_osc"] = (v5 - v20) / v20.replace(0, np.nan)
    else:
        out["volume_osc"] = np.nan
    return out


def build_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Concatenating all structure features on an OHLCV frame indexed by date."""
    parts = [
        ema_regime_features(df),
        fair_value_gaps(df),
        market_structure(df),
        trend_strength(df),
        candle_anatomy(df),
        chandelier_exit(df),
        divergence_features(df),
        volume_profile_features(df),
        liquidity_sweeps(df),
        vix_fix(df),
        first_pullback(df),
        range_regime(df),
        inside_bars(df),
        turn_days(df),
        prior_day_levels(df),
        trend_volume_extras(df),
    ]
    return pd.concat(parts, axis=1)


def latest_chandelier_long(df: pd.DataFrame) -> float:
    """Returning the current long trailing stop for the executor's ratchet."""
    return float(chandelier_exit(df)["chandelier_long"].iloc[-1])


def _market_stage(last) -> str:
    # naming the weinstein stage from long-term regime, slope, and ranging
    above = bool(last["above_ema200"])
    rising = (last["ema200_slope"] or 0) > 0
    ranging = bool(last["ranging"])
    if above and rising and not ranging:
        return "stage 2 (advancing)"
    if not above and not rising and not ranging:
        return "stage 4 (declining)"
    if above:
        return "stage 3 (distribution — uptrend stalling)"
    return "stage 1 (accumulation — downtrend basing)"


def technical_structure_block(df: pd.DataFrame) -> dict:
    """Summarizing the latest structure state as a citable evidence-packet field.

    Designed to slot into engine/data_packet.py alongside news_sentiment_avg
    and institutional holders, so panels argue from grounded structure facts.
    """
    feats = build_structure_features(df)
    last = feats.iloc[-1]
    trend_map = {
        1: "uptrend (last structure break was bullish)",
        -1: "downtrend (last structure break was bearish)",
        0: "no confirmed structure",
    }

    def _num(key, digits=2):
        # Guarding against NaN so the packet stays JSON-serializable
        val = last[key]
        return None if pd.isna(val) else round(float(val), digits)

    return {
        "feature_version": STRUCTURE_FEATURE_VERSION,
        "ema_regime": "above 50EMA" if last["above_ema"] else "below 50EMA",
        "bars_in_regime": int(last["bars_since_ema_cross"]),
        "ema_dist_atr": _num("ema_dist_atr"),
        "structure": trend_map[int(last["structure_trend"])],
        "bars_since_structure_break": _num("bars_since_structure_break", 0),
        "adx": _num("adx", 1),
        "nearest_bullish_fvg_atr_below": _num("fvg_bull_dist_atr"),
        "nearest_bearish_fvg_atr_above": _num("fvg_bear_dist_atr"),
        "momentum_asymmetry": _num("momentum_asymmetry"),
        "exhaustion_breakout_today": bool(last["exhaustion_breakout"]),
        "bull_divergence_bars_ago": _num("bars_since_bull_divergence", 0),
        "bear_divergence_bars_ago": _num("bars_since_bear_divergence", 0),
        "close_vs_volume_poc_atr": _num("poc_dist_atr"),
        "bull_liquidity_sweep_bars_ago": _num("bars_since_bull_sweep", 0),
        "bear_liquidity_sweep_bars_ago": _num("bars_since_bear_sweep", 0),
        "volatility_capitulation_bars_ago":
            _num("bars_since_capitulation", 0),
        "first_pullback": ("long setup" if last["first_pullback_long"]
                           else "short setup"
                           if last["first_pullback_short"] else None),
        "ranging_market": bool(last["ranging"]),
        "position_in_range_pct": _num("position_in_range"),
        "bb_squeeze_active": bool(last["bb_squeeze"]),
        "inside_bar_streak": int(last["inside_bar_streak"] or 0),
        "first_red_after_run": bool(last["first_red_after_run"]),
        "gap_vs_prior_range": round(float(last["gap_vs_prior_range"]), 3),
        "gap_above_prior_high": bool(last["gap_above_prior_high"]),
        "gap_below_prior_low": bool(last["gap_below_prior_low"]),
        "close_above_prior_high": bool(last["close_above_prior_high"]),
        "close_below_prior_low": bool(last["close_below_prior_low"]),
        "failed_gap_up": bool(last["failed_gap_up"]),
        "failed_gap_down": bool(last["failed_gap_down"]),
        "close_in_prior_range": round(float(last["close_in_prior_range"]), 3),
        "first_green_after_run": bool(last["first_green_after_run"]),
        "above_200ema": bool(last["above_ema200"]),
        "market_stage": _market_stage(last),
        "rsi2": _num("rsi2"),
        "connors_pullback_setup": bool(last["connors_pullback_long"]),
        "volume_trend": ("rising" if (last["volume_osc"] or 0) > 0.05
                         else "falling"
                         if (last["volume_osc"] or 0) < -0.05 else "flat")
            if not pd.isna(last["volume_osc"]) else None,
        "chandelier_long_stop": _num("chandelier_long"),
    }
