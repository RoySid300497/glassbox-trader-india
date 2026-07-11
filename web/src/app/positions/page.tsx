// showing mode, open positions, and risk gate interventions
import { supabase, Decision, SITE_MODE } from "@/lib/supabase";
import { Disclaimer, ModeBanner } from "@/lib/ui";

export const dynamic = "force-dynamic";

interface Position {
  ticker: string; qty: number; entry_price: number | null;
  entry_date: string | null; status: string;
}

export default async function Positions() {
  const [pos, blocked] = await Promise.all([
    supabase.from("positions").select("*").eq("status", "OPEN"),
    supabase.from("decisions").select("*")
      .like("risk_gate_note", "gate:%")
      .not("risk_gate_note", "like", "gate: passed%")
      .order("decided_at", { ascending: false }).limit(20),
  ]);
  const positions = (pos.data || []) as Position[];
  const blocks = (blocked.data || []) as Decision[];

  return (
    <div>
      <div className="flex items-start justify-between flex-wrap gap-3">
        <h1 className="text-2xl font-bold">Positions & risk</h1>
        <ModeBanner />
      </div>

      <div className="mt-5 bg-zinc-900 border border-zinc-800 rounded-xl p-5 text-sm">
        <span className="text-zinc-200 font-semibold">Operating modes.</span>
        <ul className="mt-2 space-y-1.5">
          {[
            ["RESEARCH", "signals and debates only, no orders placed anywhere"],
            ["PAPER", "an Alpaca paper account executes behind the risk gate — fake money, real discipline"],
            ["LIVE", "real money, owner's account only, enabled by a deliberate double interlock"],
          ].map(([mode, desc]) => (
            <li key={mode} className="text-zinc-400">
              <span className={mode === SITE_MODE
                ? "text-sky-400 font-semibold" : "text-zinc-300"}>
                {mode}
              </span>
              {mode === SITE_MODE && (
                <span className="ml-1.5 text-xs bg-sky-500/15 text-sky-400 border border-sky-500/30 px-1.5 py-0.5 rounded-full">
                  current
                </span>
              )}
              <span className="ml-2">{desc}</span>
            </li>
          ))}
        </ul>
      </div>

      <h2 className="text-lg font-semibold mt-8">Open positions</h2>
      {positions.length === 0 ? (
        <p className="text-zinc-500 mt-2 text-sm">
          No open positions — the engine has not found a trade worth taking
          yet. Restraint is the default; entries must earn their way past the
          judges and the gate.
        </p>
      ) : (
        <table className="w-full mt-3 bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden text-sm">
          <thead className="bg-zinc-800/50 text-zinc-400">
            <tr>
              <th className="text-left px-4 py-2 font-medium">Ticker</th>
              <th className="text-left px-4 py-2 font-medium">Qty</th>
              <th className="text-left px-4 py-2 font-medium">Entry</th>
              <th className="text-left px-4 py-2 font-medium">Since</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((p) => (
              <tr key={p.ticker} className="border-t border-zinc-800">
                <td className="px-4 py-2 font-medium">{p.ticker}</td>
                <td className="px-4 py-2">{p.qty}</td>
                <td className="px-4 py-2">{p.entry_price ?? "—"}</td>
                <td className="px-4 py-2">{p.entry_date?.slice(0, 10) ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <h2 className="text-lg font-semibold mt-8">Gate interventions</h2>
      <p className="text-sm text-zinc-500 mt-1">
        Trades the panels wanted but the hard-coded risk gate refused —
        discipline the models cannot override.
      </p>
      {blocks.length === 0 ? (
        <p className="text-zinc-500 mt-2 text-sm">No interventions recorded yet.</p>
      ) : (
        <ul className="mt-3 space-y-2">
          {blocks.map((b) => (
            <li key={b.id} className="bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-3 text-sm">
              <span className="font-medium">{b.ticker}</span>
              <span className="text-zinc-500 ml-2">{b.decided_at.slice(0, 10)}</span>
              <span className="ml-2 text-amber-400">{b.risk_gate_note}</span>
            </li>
          ))}
        </ul>
      )}
      <Disclaimer />
    </div>
  );
}
