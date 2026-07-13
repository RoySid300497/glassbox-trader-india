"use client";
// embedding the real TradingView advanced chart widget for a ticker.
// maps a bare NSE symbol (e.g. "DRREDDY") to TradingView's "NSE:DRREDDY"
// form. renders their branded chart with indicators and drawing tools.
import { useEffect, useRef, memo } from "react";

function TradingViewChart({ ticker, exchange = "NSE", height = 500 }:
  { ticker: string; exchange?: string; height?: number }) {
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!container.current) return;
    // clearing any prior widget (ticker changes / remounts)
    container.current.innerHTML = "";

    // normalising the symbol: strip any .NS/.BO suffix, prefix the exchange
    const bare = ticker.toUpperCase().replace(/\.(NS|BO)$/, "");
    const symbol = `${exchange}:${bare}`;

    const script = document.createElement("script");
    script.src =
      "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";
    script.type = "text/javascript";
    script.async = true;
    script.innerHTML = JSON.stringify({
      autosize: true,
      symbol,
      interval: "D",
      timezone: "Asia/Kolkata",
      theme: "dark",
      style: "1",
      locale: "en",
      enable_publishing: false,
      allow_symbol_change: false,
      hide_side_toolbar: false,
      studies: ["STD;SMA", "STD;RSI"],
      support_host: "https://www.tradingview.com",
    });
    container.current.appendChild(script);
  }, [ticker, exchange]);

  return (
    <div
      className="tradingview-widget-container"
      ref={container}
      style={{ height, width: "100%" }}
    >
      <div
        className="tradingview-widget-container__widget"
        style={{ height: "calc(100% - 32px)", width: "100%" }}
      />
      <div className="tradingview-widget-copyright">
        <a
          href={`https://www.tradingview.com/symbols/${exchange}-${ticker
            .toUpperCase()
            .replace(/\.(NS|BO)$/, "")}/`}
          rel="noopener nofollow"
          target="_blank"
        >
          <span className="text-xs text-zinc-500">
            {exchange}:{ticker.toUpperCase().replace(/\.(NS|BO)$/, "")} on
            TradingView
          </span>
        </a>
      </div>
    </div>
  );
}

export default memo(TradingViewChart);