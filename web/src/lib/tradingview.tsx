"use client";
// embedding the TradingView advanced chart widget for an NSE ticker.
// fixed sizing (no autosize/fixed-height conflict), contained overflow,
// and a proper container height so it doesn't collapse vertically.
import { useEffect, useRef, memo } from "react";

function TradingViewChart({ ticker, exchange = "NSE", height = 460 }:
  { ticker: string; exchange?: string; height?: number }) {
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!container.current) return;
    container.current.innerHTML = "";

    const bare = ticker.toUpperCase().replace(/\.(NS|BO)$/, "");
    const symbol = `${exchange}:${bare}`;

    const widgetDiv = document.createElement("div");
    widgetDiv.className = "tradingview-widget-container__widget";
    // the widget fills the inner div; leave ~28px for the attribution row
    widgetDiv.style.height = `${height - 28}px`;
    widgetDiv.style.width = "100%";
    container.current.appendChild(widgetDiv);

    const script = document.createElement("script");
    script.src =
      "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";
    script.type = "text/javascript";
    script.async = true;
    script.innerHTML = JSON.stringify({
      // explicit width/height instead of autosize avoids the collapse and
      // overflow bugs; the widget respects these against the sized container
      width: "100%",
      height: height - 28,
      symbol,
      interval: "D",
      timezone: "Asia/Kolkata",
      theme: "dark",
      style: "1",
      locale: "en",
      hide_top_toolbar: false,
      hide_side_toolbar: true,
      hide_legend: false,
      allow_symbol_change: false,
      save_image: false,
      studies: ["STD;SMA"],
      support_host: "https://www.tradingview.com",
    });
    container.current.appendChild(script);
  }, [ticker, exchange, height]);

  const bare = ticker.toUpperCase().replace(/\.(NS|BO)$/, "");
  return (
    <div
      className="tradingview-widget-container"
      ref={container}
      style={{ height, width: "100%", overflow: "hidden" }}
    >
      <div className="tradingview-widget-copyright" style={{ height: 28 }}>
        <a
          href={`https://www.tradingview.com/symbols/${exchange}-${bare}/`}
          rel="noopener nofollow"
          target="_blank"
        >
          <span className="text-xs text-zinc-500">
            {exchange}:{bare} on TradingView
          </span>
        </a>
      </div>
    </div>
  );
}

export default memo(TradingViewChart);