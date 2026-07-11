// showing the latest verdicts, market context, and quick stats
import Link from "next/link";
import { supabase, Decision } from "@/lib/supabase";
import { ActionBadge, ConfidenceBar, Disclaimer, ModeBanner, StatCard, SentimentChip } from "@/lib/ui";

export const dynamic = "force-dynamic";

async function fetchData() {
  // pulling the latest decisions, market context, and scored counts
  const [dec, ctx, scored] = await Promise.all([
    supabase.from("decisions").select("*")
      .order("decided_at", { ascending: false }).limit(30),
    supabase.from("market_context").select("summary_text")
      .order("date", { ascending: false }).limit(1),
    supabase.from("decisions").select("was_correct")
      .not("scored_at", "is", null),
  ]);
  const seen = new Set<string>();
  const latest = (dec.data || []).filter((d: Decision) => {
    if (seen.has(d.ticker)) return false;
    seen.add(d.ticker); return true;
  });
  // hiding legacy debug-format context rows from the banner
  const raw = ctx.data?.[0]?.summary_text || "";
  return {
    latest,
    context: raw.startsWith("run at ") ? "" : raw,
    scored: scored.data || [],
  };
}

function judgeSummary(d: Decision): string {
  const counts: Record<string, number> = {};
  for (const v of d.judge_votes || []) counts[v.vote] = (counts[v.vote] || 0) + 1;
  return Object.entries(counts).map(([v, n]) => `${n}× ${v.replace("_", " ")}`).join("  ·  ");
}

export default async function Briefing() {
  const { latest, context, scored } = await fetchData();
  const asOf = latest[0]?.decided_at?.slice(0, 10) || "—";
  const correct = scored.filter((s) => s.was_correct).length;

  return (
    <div>
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold">Morning briefing</h1>
          <p className="text-sm text-zinc-500 mt-1">Latest panel verdicts · {asOf}</p>
        </div>
        <ModeBanner />
      </div>

      {context && (
        <div className="mt-5 bg-zinc-900 border border-zinc-800 rounded-xl px-5 py-3 text-sm text-zinc-300">
          <span className="text-zinc-500 text-xs uppercase tracking-wide mr-3">Market</span>
          {context}
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-5">
        <StatCard label="tickers covered" value={String(latest.length)} />
        <StatCard label="decisions scored" value={String(scored.length)} />
        <StatCard label="calls correct" value={scored.length ? `${correct}/${scored.length}` : "—"} />
        <StatCard label="LLM panels" value="3 families" sub="gemini · llama · mistral" />
      </div>

      <div className="grid md:grid-cols-2 gap-4 mt-6">
        {latest.map((d) => (
          <Link key={d.id} href={`/debate/${d.id}`}
            className="block bg-zinc-900 border border-zinc-800 rounded-xl p-5 hover:border-zinc-600 transition-colors">
            <div className="flex items-center justify-between">
              <span className="text-xl font-bold">{d.ticker}</span>
              <ActionBadge action={d.action} />
            </div>
            <div className="mt-3 flex items-center gap-3 text-sm text-zinc-400">
              <span>CNN: <span className="text-zinc-200">{d.cnn_direction}</span></span>
              <ConfidenceBar value={d.cnn_confidence} />
            </div>
            <div className="mt-3 text-xs text-zinc-500">
              Judges: {judgeSummary(d) || "no votes"}
            </div>
            <div className="mt-1.5 text-xs text-zinc-600">{d.risk_gate_note}</div>
            <div className="mt-3 text-sm text-sky-400 font-medium">Read the debate →</div>
          </Link>
        ))}
      </div>

      {latest.length === 0 && (
        <p className="text-zinc-500 mt-8">No decisions yet — the engine populates this each weekday morning.</p>
      )}
      <Disclaimer />
    </div>
  );
}
