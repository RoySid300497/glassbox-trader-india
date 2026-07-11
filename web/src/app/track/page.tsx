// showing every scored call with charts and honest splits
import Link from "next/link";
import { supabase, Decision } from "@/lib/supabase";
import { ActionBadge, Disclaimer, StatCard } from "@/lib/ui";
import { CumulativeAccuracy, TickerAccuracyBar, ActionPie } from "@/lib/charts";
import DownloadCsvButton from "@/lib/download";

export const dynamic = "force-dynamic";

export default async function Track() {
  const { data } = await supabase.from("decisions").select("*")
    .not("scored_at", "is", null)
    .order("decided_at", { ascending: true }).limit(500);
  const rows = (data || []) as Decision[];

  // building the rolling hit-rate series in chronological order
  let hits = 0;
  const series = rows.map((r, i) => {
    if (r.was_correct) hits += 1;
    return { date: r.decided_at.slice(5, 10), hitRate: hits / (i + 1) };
  });

  // aggregating per-ticker accuracy and action mix
  const byTicker: Record<string, { correct: number; wrong: number }> = {};
  const byAction: Record<string, number> = {};
  for (const r of rows) {
    byTicker[r.ticker] ??= { correct: 0, wrong: 0 };
    byTicker[r.ticker][r.was_correct ? "correct" : "wrong"] += 1;
    byAction[r.action] = (byAction[r.action] || 0) + 1;
  }
  const tickerData = Object.entries(byTicker)
    .map(([ticker, v]) => ({ ticker, ...v }));
  const actionData = Object.entries(byAction)
    .map(([name, value]) => ({ name, value }));

  const trades = rows.filter((r) => r.action !== "NO_TRADE");
  const tradesCorrect = trades.filter((r) => r.was_correct).length;
  const missed = rows.filter((r) => r.action === "NO_TRADE" && !r.was_correct).length;
  const recent = [...rows].reverse().slice(0, 60);

  return (
    <div>
      <h1 className="text-2xl font-bold">Track record</h1>
      <p className="text-sm text-zinc-500 mt-1">
        Every scored call, unedited. Losses and missed opportunities counted separately.
      </p>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
        <StatCard label="scored decisions" value={String(rows.length)} />
        <StatCard label="trades correct"
          value={trades.length ? `${tradesCorrect}/${trades.length}` : "—"} />
        <StatCard label="missed opportunities" value={String(missed)}
          sub="no-trade before a move" />
        <StatCard label="overall hit rate"
          value={rows.length ? `${Math.round(100 * rows.filter(r => r.was_correct).length / rows.length)}%` : "—"} />
      </div>

      <div className="grid md:grid-cols-2 gap-4 mt-6">
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
          <h2 className="text-sm font-semibold text-zinc-300 mb-3">Rolling hit rate</h2>
          <CumulativeAccuracy data={series} />
        </div>
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
          <h2 className="text-sm font-semibold text-zinc-300 mb-3">Action mix</h2>
          <ActionPie data={actionData} />
        </div>
      </div>

      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 mt-4">
        <h2 className="text-sm font-semibold text-zinc-300 mb-3">Accuracy by ticker</h2>
        <TickerAccuracyBar data={tickerData} />
      </div>

      <div className="flex justify-end mt-6">
        <DownloadCsvButton
          rows={rows.map((r) => ({
            date: r.decided_at.slice(0, 10), ticker: r.ticker,
            action: r.action, cnn_direction: r.cnn_direction,
            cnn_confidence: r.cnn_confidence,
            outcome: r.outcome_label,
            return_1d: r.outcome_return_1d,
            result: r.action === "NO_TRADE" && !r.was_correct
              ? "missed" : r.was_correct ? "correct" : "wrong",
          }))}
          filename="glassbox_track_record.csv"
          label="download track record CSV" />
      </div>
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden mt-3">
        <table className="w-full text-sm">
          <thead className="bg-zinc-800/50 text-zinc-400">
            <tr>
              <th className="text-left px-4 py-2 font-medium">Date</th>
              <th className="text-left px-4 py-2 font-medium">Ticker</th>
              <th className="text-left px-4 py-2 font-medium">Call</th>
              <th className="text-left px-4 py-2 font-medium">Next day</th>
              <th className="text-left px-4 py-2 font-medium">Return</th>
              <th className="text-left px-4 py-2 font-medium">Result</th>
            </tr>
          </thead>
          <tbody>
            {recent.map((r) => (
              <tr key={r.id} className="border-t border-zinc-800 hover:bg-zinc-800/30">
                <td className="px-4 py-2 text-zinc-500">{r.decided_at.slice(0, 10)}</td>
                <td className="px-4 py-2 font-medium">
                  <Link href={`/debate/${r.id}`} className="hover:text-sky-400">{r.ticker}</Link>
                </td>
                <td className="px-4 py-2"><ActionBadge action={r.action} /></td>
                <td className="px-4 py-2 text-zinc-300">{r.outcome_label}</td>
                <td className="px-4 py-2 font-mono text-zinc-300">
                  {((r.outcome_return_1d || 0) * 100).toFixed(2)}%
                </td>
                <td className="px-4 py-2">
                  {r.action === "NO_TRADE" && !r.was_correct
                    ? <span className="text-amber-400">missed</span>
                    : r.was_correct
                      ? <span className="text-emerald-400">correct</span>
                      : <span className="text-rose-400">wrong</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length === 0 && <p className="text-zinc-500 mt-6">No scored decisions yet.</p>}
      <Disclaimer />
    </div>
  );
}
