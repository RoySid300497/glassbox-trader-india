// showing archived headlines with sentiment per ticker
import { supabase, NewsItem } from "@/lib/supabase";
import { Disclaimer, SentimentChip } from "@/lib/ui";
import { SentimentBar } from "@/lib/charts";

export const dynamic = "force-dynamic";

export default async function News() {
  const { data } = await supabase.from("news_archive")
    .select("id,ticker,published_at,source,headline,url,sentiment")
    .order("published_at", { ascending: false }).limit(120);
  const rows = (data || []) as NewsItem[];

  // averaging sentiment per ticker where scores exist
  const agg: Record<string, { sum: number; n: number }> = {};
  for (const r of rows) {
    if (r.sentiment === null) continue;
    agg[r.ticker] ??= { sum: 0, n: 0 };
    agg[r.ticker].sum += r.sentiment;
    agg[r.ticker].n += 1;
  }
  const sentimentData = Object.entries(agg)
    .map(([ticker, v]) => ({ ticker, sentiment: +(v.sum / v.n).toFixed(2) }));

  return (
    <div>
      <h1 className="text-2xl font-bold">News & sentiment</h1>
      <p className="text-sm text-zinc-500 mt-1">
        Every headline the panels argued over, archived with a finance-aware
        sentiment score.
      </p>

      {sentimentData.length > 0 && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 mt-6">
          <h2 className="text-sm font-semibold text-zinc-300 mb-3">
            Average sentiment by ticker (recent)
          </h2>
          <SentimentBar data={sentimentData} />
        </div>
      )}

      <div className="mt-6 space-y-2">
        {rows.map((r) => (
          <div key={r.id}
            className="bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-3 flex items-start gap-4">
            <span className="font-bold text-sm w-14 shrink-0 mt-0.5">{r.ticker}</span>
            <div className="min-w-0 flex-1">
              {r.url ? (
                <a href={r.url} target="_blank" rel="noopener noreferrer"
                   className="text-sm text-zinc-200 hover:text-sky-400">
                  {r.headline}
                </a>
              ) : (
                <span className="text-sm text-zinc-200">{r.headline}</span>
              )}
              <div className="text-xs text-zinc-600 mt-0.5">
                {r.source} · {r.published_at?.slice(0, 16).replace("T", " ")}
              </div>
            </div>
            <SentimentChip value={r.sentiment} />
          </div>
        ))}
      </div>
      {rows.length === 0 && <p className="text-zinc-500 mt-6">No news archived yet.</p>}
      <Disclaimer />
    </div>
  );
}
