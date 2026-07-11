"""Daily chandelier stop ratchet for open Alpaca positions.

Lives at src/engine/stop_ratchet.py and runs inside run_daily after the
performance sync. Upgrading the static ATR bracket stop into a trailing
stop that only ever moves in the position's favor: raising stops on
longs, lowering on shorts, never loosening either.

Dependency-injected so it reuses the executor's existing Alpaca client
and the pipeline's bar fetcher, and stays unit-testable without a broker.
"""

import logging

from pipeline.ta_structure import chandelier_exit

log = logging.getLogger(__name__)

MIN_IMPROVEMENT_PCT = 0.25


def compute_ratchet(df, side: str, current_stop: float) -> float | None:
    """Returning a tighter stop when the chandelier level beats the current one.

    df is the daily OHLCV frame for the ticker, side is "long" or "short",
    current_stop is the live stop leg price. Returns the new stop price or
    None when no favorable move of at least MIN_IMPROVEMENT_PCT exists.
    """
    levels = chandelier_exit(df).iloc[-1]

    if side == "long":
        candidate = float(levels["chandelier_long"])
        # Ratcheting up only, and only when the move is worth an order replace
        if candidate > current_stop * (1 + MIN_IMPROVEMENT_PCT / 100):
            return round(candidate, 2)
    else:
        candidate = float(levels["chandelier_short"])
        if candidate < current_stop * (1 - MIN_IMPROVEMENT_PCT / 100):
            return round(candidate, 2)
    return None


def ratchet_open_stops(list_positions, list_stop_orders, replace_stop, fetch_bars):
    """Walking every open position and tightening its stop leg where earned.

    Callbacks keep this module broker-agnostic:
      list_positions()            -> [{"symbol", "side", "qty"}]
      list_stop_orders(symbol)    -> [{"id", "stop_price"}] for open stop legs
      replace_stop(order_id, px)  -> submits the replacement at px
      fetch_bars(symbol)          -> daily OHLCV DataFrame with open/high/low/close

    Returning a summary list for the daily report and Supabase logging.
    """
    actions = []
    for pos in list_positions():
        symbol, side = pos["symbol"], pos["side"]
        stops = list_stop_orders(symbol)
        if not stops:
            log.warning("no open stop leg found for %s, skipping ratchet", symbol)
            continue

        df = fetch_bars(symbol)
        if df is None or len(df) < 30:
            log.warning("insufficient bars for %s, skipping ratchet", symbol)
            continue

        for order in stops:
            new_stop = compute_ratchet(df, side, float(order["stop_price"]))
            if new_stop is None:
                continue
            try:
                replace_stop(order["id"], new_stop)
                actions.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "old_stop": float(order["stop_price"]),
                        "new_stop": new_stop,
                    }
                )
                log.info(
                    "ratcheted %s stop %.2f -> %.2f",
                    symbol,
                    float(order["stop_price"]),
                    new_stop,
                )
            except Exception:
                # Never letting a broker hiccup on one leg block the daily run
                log.exception("stop replace failed for %s, leaving stop as-is", symbol)
    return actions
