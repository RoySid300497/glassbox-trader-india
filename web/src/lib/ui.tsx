// rendering shared dark-theme badges, cards, and banners
import { SITE_MODE } from "./supabase";

export function ActionBadge({ action }: { action: string }) {
  const styles: Record<string, string> = {
    BUY: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
    SELL: "bg-rose-500/15 text-rose-400 border-rose-500/30",
    NO_TRADE: "bg-zinc-500/15 text-zinc-400 border-zinc-500/30",
    Up: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
    Down: "bg-rose-500/15 text-rose-400 border-rose-500/30",
    Neutral: "bg-zinc-500/15 text-zinc-400 border-zinc-500/30",
  };
  return (
    <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold border ${styles[action] || styles.NO_TRADE}`}>
      {action.replace("_", " ")}
    </span>
  );
}

export function SentimentChip({ value }: { value: number | null }) {
  if (value === null || value === undefined)
    return <span className="text-xs text-zinc-500">—</span>;
  const cls = value > 0.15 ? "text-emerald-400" : value < -0.15 ? "text-rose-400" : "text-zinc-400";
  return <span className={`text-xs font-mono ${cls}`}>{value > 0 ? "+" : ""}{value.toFixed(2)}</span>;
}

export function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round((value || 0) * 100);
  return (
    <div className="flex items-center gap-2">
      <div className="w-24 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
        <div className="h-full bg-sky-500 rounded-full" style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-zinc-500">{pct}%</span>
    </div>
  );
}

export function StatCard({ label, value, sub }:
  { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4">
      <div className="text-2xl font-bold text-zinc-100">{value}</div>
      <div className="text-xs text-zinc-500 mt-1">{label}</div>
      {sub && <div className="text-xs text-zinc-600 mt-0.5">{sub}</div>}
    </div>
  );
}

export function ModeBanner() {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="px-2 py-0.5 rounded-full bg-sky-500/15 text-sky-400 border border-sky-500/30 font-semibold">
        {SITE_MODE} MODE
      </span>
      <span className="text-zinc-500">
        {SITE_MODE === "LIVE"
          ? "real orders on the owner's account — every trade accountable"
          : SITE_MODE === "PAPER"
            ? "simulated orders on a paper account — no real money"
            : "signals & debates only — no orders are placed"}
      </span>
    </div>
  );
}

export function Disclaimer() {
  return (
    <p className="text-xs text-zinc-600 mt-10 pb-8">
      Educational output only — nothing here is financial advice. Every
      decision shown was produced by AI models debating public data, gated
      by hard-coded risk rules.
    </p>
  );
}
