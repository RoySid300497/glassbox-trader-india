// showing the paper equity curve against spy and every closed trade
import { supabase, EquityPoint, Trade, SITE_MODE } from "@/lib/supabase";
import { Disclaimer, ModeBanner, StatCard } from "@/lib/ui";
import { EquityCurve } from "@/lib/charts";
import DownloadCsvButton from "@/lib/download";

export const dynamic = "force-dynamic";

export default async function Performance() {
  const [eq, tr] = await Promise.all([
    supabase.from("portfolio_history").select("*")
      .order("date", { ascending: true }).limit(400),
    supabase.from("trades").select("*")
      .order("exit_at", { ascending: false }).limit(200),
  ]);
  const equity = (eq.data || []) as EquityPoint[];
  const trades = (tr.data || []) as Trade[];

  const wins = trades.filter((t) => t.pnl > 0);
  const losses = trades.filter((t) => t.pnl <= 0);
  const totalPnl = trades.reduce((s, t) => s + t.pnl, 0);
  const nonzero = equity.filter((e) => e.equity > 0);
  const ret = nonzero.length > 1
    ? (nonzero[nonzero.length - 1].equity / nonzero[0].equity - 1) * 100
    : null;

  return (
    <div>
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold">Performance</h1>
          <p className="text-sm text-zinc-500 mt-1">
            Account equity against SPY, and every closed trade with its
            realized result.{" "}
            {SITE_MODE === "LIVE"
              ? "Real money — every trade accountable."
              : "Fake money, real discipline."}
          </p>
        </div>
        <ModeBanner />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
        <StatCard label="account return"
          value={ret !== null ? `${ret >= 0 ? "+" : ""}${ret.toFixed(2)}%` : "—"} />
        <StatCard label="realized P&L"
          value={trades.length ? `$${totalPnl.toFixed(2)}` : "—"} />
        <StatCard label="trade win rate"
          value={trades.length
            ? `${wins.length}/${trades.length} (${Math.round(100 * wins.length / trades.length)}%)`
            : "—"} />
        <StatCard label="avg win / avg loss"
          value={wins.length && losses.length
            ? `$${(wins.reduce((s, t) => s + t.pnl, 0) / wins.length).toFixed(0)} / $${Math.abs(losses.reduce((s, t) => s + t.pnl, 0) / losses.length).toFixed(0)}`
            : "—"} />
      </div>

      {equity.length > 1 ? (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 mt-6">
          <h2 className="text-sm font-semibold text-zinc-300 mb-3">
            Equity curve (normalized to 100)
          </h2>
          <EquityCurve equity={equity} />
        </div>
      ) : (
        <p className="text-zinc-500 mt-8 text-sm">
          The equity curve appears after the first days of paper trading
          accumulate.
        </p>
      )}

      <div className="flex items-center justify-between mt-8">
        <h2 className="text-lg font-semibold">Closed trades</h2>
        <DownloadCsvButton rows={trades as unknown as Record<string, unknown>[]}
          filename="glassbox_trades.csv" label="download trades CSV" />
      </div>
      {trades.length === 0 ? (
        <p className="text-zinc-500 mt-2 text-sm">
          No closed trades yet — a trade closes when its stop, target, or a
          SELL vote triggers. The first entries appear here automatically.
        </p>
      ) : (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden mt-3">
          <table className="w-full text-sm">
            <thead className="bg-zinc-800/50 text-zinc-400">
              <tr>
                <th className="text-left px-4 py-2 font-medium">Exit date</th>
                <th className="text-left px-4 py-2 font-medium">Ticker</th>
                <th className="text-left px-4 py-2 font-medium">Qty</th>
                <th className="text-left px-4 py-2 font-medium">Entry</th>
                <th className="text-left px-4 py-2 font-medium">Exit</th>
                <th className="text-left px-4 py-2 font-medium">P&L</th>
                <th className="text-left px-4 py-2 font-medium">Return</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t) => (
                <tr key={t.exit_fill_id} className="border-t border-zinc-800">
                  <td className="px-4 py-2 text-zinc-500">{t.exit_at?.slice(0, 10)}</td>
                  <td className="px-4 py-2 font-medium">{t.ticker}</td>
                  <td className="px-4 py-2">{t.qty}</td>
                  <td className="px-4 py-2 font-mono">{t.entry_price?.toFixed(2)}</td>
                  <td className="px-4 py-2 font-mono">{t.exit_price?.toFixed(2)}</td>
                  <td className={`px-4 py-2 font-mono ${t.pnl > 0 ? "text-emerald-400" : "text-rose-400"}`}>
                    {t.pnl > 0 ? "+" : ""}{t.pnl?.toFixed(2)}
                  </td>
                  <td className={`px-4 py-2 font-mono ${t.pnl_pct > 0 ? "text-emerald-400" : "text-rose-400"}`}>
                    {(t.pnl_pct * 100).toFixed(2)}%
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
