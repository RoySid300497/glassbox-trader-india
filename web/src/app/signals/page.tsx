// charting the cnn signal history per ticker
import { supabase, Decision, ModelPred, JudgeVote } from "@/lib/supabase";
import { ActionBadge, Disclaimer } from "@/lib/ui";
import { ConfidenceLines, ModelStandings, ModelRaceLines } from "@/lib/charts";

const VOTE_MATCH: Record<string, string> = {
  BUY: "Up", SELL: "Down", NO_TRADE: "Neutral",
};

export const dynamic = "force-dynamic";

export default async function Signals() {
  const [{ data }, predsRes, scoredRes] = await Promise.all([
    supabase.from("decisions")
      .select("decided_at,ticker,cnn_direction,cnn_confidence")
      .order("decided_at", { ascending: true }).limit(600),
    supabase.from("model_predictions").select("*")
      .not("scored_at", "is", null)
      .order("pred_date", { ascending: true }).limit(2000),
    supabase.from("decisions").select("judge_votes,outcome_label")
      .not("scored_at", "is", null).limit(1000),
  ]);
  const rows = (data || []) as Pick<Decision,
    "decided_at" | "ticker" | "cnn_direction" | "cnn_confidence">[];
  const preds = (predsRes.data || []) as ModelPred[];

  // aggregating the signal-model tournament standings and race
  const agg: Record<string, { correct: number; scored: number }> = {};
  const running: Record<string, { correct: number; scored: number }> = {};
  const raceByDate: Record<string, Record<string, string | number>> = {};
  for (const p of preds) {
    agg[p.model] ??= { correct: 0, scored: 0 };
    agg[p.model].scored += 1;
    if (p.was_correct) agg[p.model].correct += 1;
    running[p.model] ??= { correct: 0, scored: 0 };
    running[p.model].scored += 1;
    if (p.was_correct) running[p.model].correct += 1;
    const d = p.pred_date.slice(5);
    raceByDate[d] ??= { date: d };
    raceByDate[d][p.model] =
      +(running[p.model].correct / running[p.model].scored).toFixed(3);
  }
  const standings = Object.entries(agg)
    .map(([model, s]) => ({ model, rate: s.correct / s.scored,
                            scored: s.scored }))
    .sort((a, b) => b.rate - a.rate);
  const raceModels = standings.map((s) => s.model);
  const race = Object.values(raceByDate);

  // scoring each llm judge's votes against real outcomes
  const judgeAgg: Record<string, { correct: number; scored: number }> = {};
  for (const d of (scoredRes.data || []) as
       { judge_votes: JudgeVote[]; outcome_label: string | null }[]) {
    if (!d.outcome_label) continue;
    for (const v of d.judge_votes || []) {
      judgeAgg[v.provider] ??= { correct: 0, scored: 0 };
      judgeAgg[v.provider].scored += 1;
      if (VOTE_MATCH[v.vote] === d.outcome_label)
        judgeAgg[v.provider].correct += 1;
    }
  }
  const judgeStandings = Object.entries(judgeAgg)
    .map(([model, s]) => ({ model, rate: s.correct / s.scored,
                            scored: s.scored }))
    .sort((a, b) => b.rate - a.rate);

  // pivoting confidence per ticker into one date-indexed series
  const tickers = [...new Set(rows.map((r) => r.ticker))];
  const byDate: Record<string, Record<string, string | number>> = {};
  for (const r of rows) {
    const date = r.decided_at.slice(5, 10);
    byDate[date] ??= { date };
    byDate[date][r.ticker] = r.cnn_confidence;
  }
  const series = Object.values(byDate);

  // collecting the latest signal per ticker for the cards
  const latest: Record<string, typeof rows[number]> = {};
  for (const r of rows) latest[r.ticker] = r;

  return (
    <div>
      <h1 className="text-2xl font-bold">Signal engine</h1>
      <p className="text-sm text-zinc-500 mt-1">
        The CNN's direction call and confidence per ticker — the quantitative
        anchor every debate starts from.
      </p>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mt-6">
        {Object.values(latest).map((r) => (
          <div key={r.ticker} className="bg-zinc-900 border border-zinc-800 rounded-xl p-4">
            <div className="flex items-center justify-between">
              <span className="font-bold">{r.ticker}</span>
              <ActionBadge action={r.cnn_direction} />
            </div>
            <div className="text-xs text-zinc-500 mt-2">
              confidence <span className="text-zinc-300 font-mono">
                {Math.round(r.cnn_confidence * 100)}%</span>
            </div>
          </div>
        ))}
      </div>

      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 mt-6">
        <h2 className="text-sm font-semibold text-zinc-300 mb-3">
          CNN confidence over time
        </h2>
        <ConfidenceLines data={series} tickers={tickers} />
      </div>

      <p className="text-xs text-zinc-600 mt-4">
        Confidence near 33% means the model sees no edge between Up, Down, and
        Neutral. The weekly review tracks this model's live hit rate and
        triggers retraining when it decays toward random.
      </p>

      <h2 className="text-lg font-semibold mt-8">Model tournament</h2>
      {standings.length === 0 ? (
        <p className="text-sm text-zinc-500 mt-1">
          Standings appear once models have scored predictions. The tournament
          fills in as decisions are graded against real next-day outcomes.
        </p>
      ) : (
        <>
          <p className="text-sm text-zinc-500 mt-1">
            Every classifier predicts the same tickers on the same days; code
            scores them all. The weekly election seats the best routable model
            as the live signal engine. Dashed line: ~33% random baseline.
          </p>
          <div className="grid md:grid-cols-2 gap-4 mt-3">
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
              <h3 className="text-sm font-semibold text-zinc-300 mb-3">
                Standings
              </h3>
              <ModelStandings data={standings} />
            </div>
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
              <h3 className="text-sm font-semibold text-zinc-300 mb-3">
                Cumulative hit rate over time
              </h3>
              <ModelRaceLines data={race} models={raceModels} />
            </div>
          </div>
        </>
      )}

      {/* judge section */}
      <h2 className="text-lg font-semibold mt-8">LLM judge accuracy</h2>
      {judgeStandings.length === 0 ? (
        <p className="text-sm text-zinc-500 mt-1">
          Judge accuracy appears once judges have cast scored directional
          votes. Most early decisions are NO&nbsp;TRADE, so this fills in
          slowly as the panel takes real positions.
        </p>
      ) : (
        <>
          <p className="text-sm text-zinc-500 mt-1">
            Each judge's votes measured against what the market actually did
            next day (BUY↔Up, SELL↔Down, NO TRADE↔Neutral).
          </p>
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 mt-3">
            <ModelStandings data={judgeStandings} />
          </div>
        </>
      )}
      <Disclaimer />
    </div>
  );
}
