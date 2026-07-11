// showing the weekly report history the engine writes every saturday
import { supabase, WeeklyReport } from "@/lib/supabase";
import { Disclaimer, StatCard } from "@/lib/ui";

export const dynamic = "force-dynamic";

export default async function Reports() {
  const { data } = await supabase.from("reports").select("*")
    .order("week_of", { ascending: false }).limit(26);
  const reports = (data || []) as WeeklyReport[];
  const latest = reports[0];

  return (
    <div>
      <h1 className="text-2xl font-bold">Weekly reports</h1>
      <p className="text-sm text-zinc-500 mt-1">
        The engine audits itself every Saturday — decision volume, accuracy,
        the model tournament, lessons learned, and paper equity.
      </p>

      {latest && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
          <StatCard label="decisions this week"
            value={String(latest.stats.decisions)} />
          <StatCard label="calls correct"
            value={latest.stats.scored
              ? `${latest.stats.correct}/${latest.stats.scored}` : "—"} />
          <StatCard label="trades placed"
            value={String(latest.stats.trades)} />
          <StatCard label="paper equity"
            value={latest.stats.equity
              ? `$${latest.stats.equity.toLocaleString()}` : "—"} />
        </div>
      )}

      <h2 className="text-lg font-semibold mt-8">Weekly self-audits</h2>
      {reports.length === 0 ? (
        <p className="text-zinc-500 mt-8 text-sm">
          No reports yet — the first one is written on the next Saturday
          review.
        </p>
      ) : (
        <div className="mt-6 space-y-4">
          {reports.map((r) => (
            <div key={r.week_of}
              className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <span className="font-semibold">Week of {r.week_of}</span>
                <span className="text-xs text-zinc-500">
                  {r.stats.decisions} decisions · {r.stats.trades} trades ·{" "}
                  {r.stats.scored
                    ? `${Math.round(100 * r.stats.correct / r.stats.scored)}% correct`
                    : "unscored"}
                </span>
              </div>

              {Object.keys(r.stats.models || {}).length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {Object.entries(r.stats.models).map(([m, s]) => (
                    <span key={m}
                      className="text-xs bg-zinc-800 text-zinc-300 px-2.5 py-1 rounded-full">
                      {m}: {s.correct}/{s.scored}
                    </span>
                  ))}
                </div>
              )}

              {(r.stats.new_lessons || []).length > 0 && (
                <ul className="mt-3 space-y-1">
                  {r.stats.new_lessons.map((l, i) => (
                    <li key={i} className="text-sm text-amber-300/90">
                      lesson: {l}
                    </li>
                  ))}
                </ul>
              )}

              {r.stats.equity !== undefined && (
                <div className="mt-3 text-xs text-zinc-500">
                  equity ${r.stats.equity.toLocaleString()}
                  {r.stats.last_equity !== undefined &&
                    ` (prev $${r.stats.last_equity.toLocaleString()})`}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
      <Disclaimer />
    </div>
  );
}
