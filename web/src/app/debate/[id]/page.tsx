// showing one decision's full debate in the dark theme
import { supabase, Decision, PanelCase } from "@/lib/supabase";
import { ActionBadge, ConfidenceBar, Disclaimer } from "@/lib/ui";
import CandleChart from "@/lib/candle";

export const dynamic = "force-dynamic";

function CaseBlock({ title, cases, tone }:
  { title: string; cases: PanelCase[]; tone: "bull" | "bear" }) {
  const border = tone === "bull" ? "border-emerald-500/20" : "border-rose-500/20";
  const head = tone === "bull" ? "text-emerald-400" : "text-rose-400";
  return (
    <div className={`bg-zinc-900 border ${border} rounded-xl p-5`}>
      <h3 className={`font-semibold ${head}`}>{title}</h3>
      {(cases || []).map((c, i) => (
        <div key={i} className="mt-4">
          <div className="flex items-center gap-3 text-xs text-zinc-500">
            <span className="font-medium uppercase tracking-wide">{c.provider}</span>
            <ConfidenceBar value={c.confidence} />
          </div>
          <ul className="mt-2 space-y-2">
            {(c.key_points || []).map((p, j) => (
              <li key={j} className="text-sm text-zinc-300">
                {p.claim}
                <code className="ml-2 text-xs bg-zinc-800 text-zinc-500 px-1.5 py-0.5 rounded">
                  {p.evidence_field}
                </code>
              </li>
            ))}
          </ul>
        </div>
      ))}
      {(!cases || cases.length === 0) && (
        <p className="text-sm text-zinc-600 mt-2">No response recorded.</p>
      )}
    </div>
  );
}

export default async function Debate({ params }:
  { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const numericId = parseInt(id, 10);
  if (isNaN(numericId)) return <p>Invalid decision id.</p>;

  const { data } = await supabase
    .from("decisions").select("*").eq("id", numericId).single();
  const d = data as Decision | null;
  if (!d) return <p className="text-zinc-500">Decision not found.</p>;

  return (
    <div>
      <div className="flex items-center gap-4 flex-wrap">
        <h1 className="text-2xl font-bold">{d.ticker}</h1>
        <ActionBadge action={d.action} />
        <span className="text-sm text-zinc-500">
          {d.decided_at.slice(0, 16).replace("T", " ")}
        </span>
      </div>
      <p className="text-sm text-zinc-500 mt-1">
        CNN signal: {d.cnn_direction} ({Math.round(d.cnn_confidence * 100)}%)
        · {d.risk_gate_note}
      </p>

      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 mt-6">
        <h2 className="text-sm font-semibold text-zinc-300 mb-3">
          Price action (6 months, decision marked)
        </h2>
        <CandleChart ticker={d.ticker} decisionTime={d.decided_at} />
      </div>

      <div className="grid md:grid-cols-2 gap-4 mt-6">
        <CaseBlock title="Bull case" cases={d.bull_case?.opening} tone="bull" />
        <CaseBlock title="Bear case" cases={d.bear_case?.opening} tone="bear" />
        <CaseBlock title="Bull rebuttal" cases={d.bull_case?.rebuttal} tone="bull" />
        <CaseBlock title="Bear rebuttal" cases={d.bear_case?.rebuttal} tone="bear" />
      </div>

      <h2 className="text-lg font-semibold mt-8">Judge votes</h2>
      <div className="grid md:grid-cols-3 gap-4 mt-3">
        {(d.judge_votes || []).map((v, i) => (
          <div key={i} className="bg-zinc-900 border border-zinc-800 rounded-xl p-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium uppercase tracking-wide text-zinc-500">
                {v.provider}
              </span>
              <ActionBadge action={v.vote} />
            </div>
            <div className="mt-2"><ConfidenceBar value={v.confidence} /></div>
            <p className="text-sm text-zinc-300 mt-3">{v.reason}</p>
          </div>
        ))}
      </div>

      {d.scored_at && (
        <div className="mt-8 bg-zinc-900 border border-zinc-800 rounded-xl p-5">
          <h2 className="text-lg font-semibold">Outcome</h2>
          <p className="text-sm text-zinc-300 mt-2">
            Next day: {d.outcome_label} (
            <span className="font-mono">
              {((d.outcome_return_1d || 0) * 100).toFixed(2)}%
            </span>) — this call was{" "}
            <span className={d.was_correct ? "text-emerald-400 font-semibold" : "text-rose-400 font-semibold"}>
              {d.was_correct ? "correct" : "wrong"}
            </span>
          </p>
        </div>
      )}
      <Disclaimer />
    </div>
  );
}
