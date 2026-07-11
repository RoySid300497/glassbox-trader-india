// showing the full-universe morning scan and which names got debated
import Link from "next/link";
import { supabase, ScreenRow, Decision } from "@/lib/supabase";
import { ActionBadge, ConfidenceBar, Disclaimer } from "@/lib/ui";

export const dynamic = "force-dynamic";

export default async function Scan() {
  const { data: latest } = await supabase.from("screen_results")
    .select("scan_date").order("scan_date", { ascending: false }).limit(1);
  const scanDate = latest?.[0]?.scan_date;

  const [scanRes, decRes] = await Promise.all([
    scanDate
      ? supabase.from("screen_results").select("*")
          .eq("scan_date", scanDate).order("score", { ascending: false })
      : Promise.resolve({ data: [] }),
    scanDate
      ? supabase.from("decisions").select("id,ticker,decided_at")
          .gte("decided_at", scanDate)
      : Promise.resolve({ data: [] }),
  ]);
  const rows = (scanRes.data || []) as ScreenRow[];
  const debated = new Map(
    ((decRes.data || []) as Pick<Decision, "id" | "ticker" | "decided_at">[])
      .map((d) => [d.ticker, d.id]));

  return (
    <div>
      <h1 className="text-2xl font-bold">Today&apos;s scan</h1>
      <p className="text-sm text-zinc-500 mt-1">
        Every morning the CNN scans the full S&amp;P universe. The top-ranked
        names below earned a seat at the debate — the rest were considered
        and passed over. {scanDate && `Scan date: ${scanDate}.`}
      </p>

      {rows.length === 0 ? (
        <p className="text-zinc-500 mt-8">
          No scan recorded yet — the first universe sweep runs on the next
          weekday morning.
        </p>
      ) : (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden mt-6">
          <table className="w-full text-sm">
            <thead className="bg-zinc-800/50 text-zinc-400">
              <tr>
                <th className="text-left px-4 py-2 font-medium">#</th>
                <th className="text-left px-4 py-2 font-medium">Ticker</th>
                <th className="text-left px-4 py-2 font-medium">CNN call</th>
                <th className="text-left px-4 py-2 font-medium">Confidence</th>
                <th className="text-left px-4 py-2 font-medium">Interest score</th>
                <th className="text-left px-4 py-2 font-medium">Debated</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={r.ticker}
                    className={`border-t border-zinc-800 ${debated.has(r.ticker) ? "bg-sky-500/5" : ""}`}>
                  <td className="px-4 py-2 text-zinc-500">{i + 1}</td>
                  <td className="px-4 py-2 font-medium">{r.ticker}</td>
                  <td className="px-4 py-2"><ActionBadge action={r.direction} /></td>
                  <td className="px-4 py-2"><ConfidenceBar value={r.confidence} /></td>
                  <td className="px-4 py-2 font-mono text-zinc-300">{r.score.toFixed(3)}</td>
                  <td className="px-4 py-2">
                    {debated.has(r.ticker) ? (
                      <Link href={`/debate/${debated.get(r.ticker)}`}
                            className="text-sky-400 hover:underline">debate →</Link>
                    ) : (
                      <span className="text-zinc-600">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <Disclaimer />
    </div>
  );
}
