"use client";
// rendering all recharts visualisations as client components

import {
  ResponsiveContainer, LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, Tooltip, Legend, CartesianGrid, ReferenceLine,
} from "recharts";

const GRID = "#27272a";
const TEXT = "#a1a1aa";  // zinc-400, readable on dark bg
const TOOLTIP = {
  contentStyle: { background: "#18181b", border: "1px solid #3f3f46",
                  borderRadius: 8, color: "#e4e4e7", fontSize: 12 },
  itemStyle: { color: "#e4e4e7" },
  labelStyle: { color: "#a1a1aa" },
};

export function CumulativeAccuracy({ data }:
  { data: { date: string; hitRate: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
        <XAxis dataKey="date" stroke={TEXT} fontSize={11} />
        <YAxis domain={[0, 1]} stroke={TEXT} fontSize={11}
               tickFormatter={(v) => `${Math.round(v * 100)}%`} />
        <Tooltip {...TOOLTIP} formatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
        <ReferenceLine y={0.333} stroke="#f43f5e" strokeDasharray="4 4"
                       label={{ value: "random", fill: "#f43f5e", fontSize: 10 }} />
        <Line type="monotone" dataKey="hitRate" stroke="#38bdf8"
              strokeWidth={2} dot={false} name="rolling hit rate" />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function TickerAccuracyBar({ data }:
  { data: { ticker: string; correct: number; wrong: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
        <XAxis dataKey="ticker" stroke={TEXT} fontSize={11} />
        <YAxis stroke={TEXT} fontSize={11} allowDecimals={false} />
        <Tooltip {...TOOLTIP} />
        <Legend wrapperStyle={{ fontSize: 12, color: "#e4e4e7" }} />
        <Bar dataKey="correct" stackId="a" fill="#34d399" name="correct" />
        <Bar dataKey="wrong" stackId="a" fill="#fb7185" name="wrong" />
      </BarChart>
    </ResponsiveContainer>
  );
}

const ACTION_COLORS: Record<string, string> = {
  BUY: "#34d399", SELL: "#fb7185", NO_TRADE: "#71717a",
};

export function ActionPie({ data }:
  { data: { name: string; value: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={240}>
      <PieChart>
        <Pie data={data} dataKey="value" nameKey="name" innerRadius={55}
             outerRadius={85} paddingAngle={3}>
          {data.map((d) => (
            <Cell key={d.name} fill={ACTION_COLORS[d.name] || "#71717a"} />
          ))}
        </Pie>
        <Tooltip {...TOOLTIP} />
        <Legend wrapperStyle={{ fontSize: 12, color: "#e4e4e7" }} />
      </PieChart>
    </ResponsiveContainer>
  );
}

const TICKER_COLORS = ["#38bdf8", "#34d399", "#fbbf24", "#f472b6", "#a78bfa",
                       "#fb7185", "#4ade80", "#f97316"];

export function ConfidenceLines({ data, tickers }:
  { data: Record<string, string | number>[]; tickers: string[] }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
        <XAxis dataKey="date" stroke={TEXT} fontSize={11} />
        <YAxis domain={[0, 1]} stroke={TEXT} fontSize={11}
               tickFormatter={(v) => `${Math.round(v * 100)}%`} />
        <Tooltip {...TOOLTIP} />
        <Legend wrapperStyle={{ fontSize: 12, color: "#e4e4e7" }} />
        {tickers.map((t, i) => (
          <Line key={t} type="monotone" dataKey={t} connectNulls
                stroke={TICKER_COLORS[i % TICKER_COLORS.length]}
                strokeWidth={2} dot={{ r: 2 }} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function SentimentBar({ data }:
  { data: { ticker: string; sentiment: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
        <XAxis dataKey="ticker" stroke={TEXT} fontSize={11} />
        <YAxis domain={[-1, 1]} stroke={TEXT} fontSize={11} />
        <Tooltip {...TOOLTIP} formatter={(v: number) => v.toFixed(2)} />
        <ReferenceLine y={0} stroke="#52525b" />
        <Bar dataKey="sentiment" name="avg news sentiment">
          {data.map((d) => (
            <Cell key={d.ticker}
                  fill={d.sentiment >= 0 ? "#34d399" : "#fb7185"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function EquityCurve({ equity }:
  { equity: { date: string; equity: number }[] }) {
  const [data, setData] = (require("react") as typeof import("react"))
    .useState<Record<string, string | number>[]>([]);

  (require("react") as typeof import("react")).useEffect(() => {
    // normalising both series to 100 at the first nonzero date
    const clean = equity.filter((e) => e.equity > 0);
    const base = clean[0]?.equity;
    if (!base) { setData([]); return; }
    const own = clean.map((e) => ({
      date: e.date.slice(5), engine: +(100 * e.equity / base).toFixed(2),
    }));
    const first = clean[0]?.date;
    fetch("/api/candles/SPY").then((r) => r.json()).then(({ candles }) => {
      const spy = (candles || []).filter((c: { time: number }) =>
        new Date(c.time * 1000).toISOString().slice(0, 10) >= (first || ""));
      const spyBase = spy[0]?.close;
      const byDate: Record<string, number> = {};
      for (const c of spy) {
        byDate[new Date(c.time * 1000).toISOString().slice(5, 10)] =
          +(100 * c.close / spyBase).toFixed(2);
      }
      setData(own.map((o) => ({ ...o, spy: byDate[o.date] })));
    }).catch(() => setData(own));
  }, [equity]);

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
        <XAxis dataKey="date" stroke={TEXT} fontSize={11} />
        <YAxis domain={["auto", "auto"]} stroke={TEXT} fontSize={11} />
        <Tooltip {...TOOLTIP} />
        <Legend wrapperStyle={{ fontSize: 12, color: "#e4e4e7" }} />
        <Line type="monotone" dataKey="engine" stroke="#38bdf8"
              strokeWidth={2} dot={false} name="paper account" />
        <Line type="monotone" dataKey="spy" stroke="#71717a"
              strokeWidth={2} dot={false} strokeDasharray="5 3"
              name="SPY benchmark" />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function ModelStandings({ data }:
  { data: { model: string; rate: number; scored: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data} layout="vertical" margin={{ left: 30 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
        <XAxis type="number" domain={[0, 1]} stroke={TEXT} fontSize={11}
               tickFormatter={(v) => `${Math.round(v * 100)}%`} />
        <YAxis type="category" dataKey="model" stroke={TEXT} fontSize={11}
               width={110} />
        <Tooltip {...TOOLTIP}
          formatter={(v: number, _n, item) =>
            [`${(v * 100).toFixed(1)}% of ${item.payload.scored}`, "hit rate"]} />
        <ReferenceLine x={0.333} stroke="#f43f5e" strokeDasharray="4 4" />
        <Bar dataKey="rate" name="hit rate">
          {data.map((d, i) => (
            <Cell key={d.model}
                  fill={TICKER_COLORS[i % TICKER_COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function ModelRaceLines({ data, models }:
  { data: Record<string, string | number>[]; models: string[] }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
        <XAxis dataKey="date" stroke={TEXT} fontSize={11} />
        <YAxis domain={[0, 1]} stroke={TEXT} fontSize={11}
               tickFormatter={(v) => `${Math.round(v * 100)}%`} />
        <Tooltip {...TOOLTIP} />
        <Legend wrapperStyle={{ fontSize: 12, color: "#e4e4e7" }} />
        <ReferenceLine y={0.333} stroke="#f43f5e" strokeDasharray="4 4" />
        {models.map((m, i) => (
          <Line key={m} type="monotone" dataKey={m} connectNulls
                stroke={TICKER_COLORS[i % TICKER_COLORS.length]}
                strokeWidth={2} dot={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
