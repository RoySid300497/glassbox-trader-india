"use client";
// drawing a dark candlestick chart with a marker at the decision moment
import { useEffect, useRef } from "react";
import { createChart, ColorType, UTCTimestamp, CandlestickSeries, createSeriesMarkers } from "lightweight-charts";

export default function CandleChart({ ticker, decisionTime }:
  { ticker: string; decisionTime?: string }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const chart = createChart(ref.current, {
      height: 300,
      layout: { background: { type: ColorType.Solid, color: "transparent" },
                textColor: "#a1a1aa" },  // readable on dark bg
      grid: { vertLines: { color: "#27272a" },
              horzLines: { color: "#27272a" } },
      rightPriceScale: { borderColor: "#3f3f46" },
      timeScale: { borderColor: "#3f3f46" },
    });
    const series = chart.addSeries(CandlestickSeries, {
      upColor: "#34d399", downColor: "#fb7185",
      wickUpColor: "#34d399", wickDownColor: "#fb7185",
      borderVisible: false,
    });

    fetch(`/api/candles/${ticker}`)
      .then((r) => r.json())
      .then(({ candles }) => {
        if (!candles?.length) return;
        series.setData(candles);
        if (decisionTime) {
          const ts = Math.floor(new Date(decisionTime).getTime() / 1000);
          const nearest = candles.reduce(
            (best: { time: number }, c: { time: number }) =>
              Math.abs(c.time - ts) < Math.abs(best.time - ts) ? c : best,
            candles[0]);
          createSeriesMarkers(series, [{
            time: nearest.time as UTCTimestamp, position: "aboveBar",
            color: "#38bdf8", shape: "arrowDown", text: "decision",
          }]);
        }
        chart.timeScale().fitContent();
      });

    const resize = () => chart.applyOptions({ width: ref.current?.clientWidth });
    resize();
    window.addEventListener("resize", resize);
    return () => { window.removeEventListener("resize", resize); chart.remove(); };
  }, [ticker, decisionTime]);

  return <div ref={ref} className="w-full" />;
}
