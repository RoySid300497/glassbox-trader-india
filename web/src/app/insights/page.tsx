// showing active theses and the distilled lessons memory
import { supabase, Thesis, Lesson } from "@/lib/supabase";
import { Disclaimer, ConfidenceBar } from "@/lib/ui";

export const dynamic = "force-dynamic";

const STATUS_STYLE: Record<string, string> = {
  ACTIVE: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  WEAKENING: "bg-amber-500/15 text-amber-400 border-amber-500/30",
  CLOSED: "bg-zinc-500/15 text-zinc-400 border-zinc-500/30",
};

export default async function Insights() {
  const [th, le] = await Promise.all([
    supabase.from("theses").select("*")
      .order("updated_at", { ascending: false }).limit(30),
    supabase.from("lessons").select("id,created_at,lesson_text")
      .eq("active", true).order("created_at", { ascending: false }).limit(30),
  ]);
  const theses = (th.data || []) as Thesis[];
  const lessons = (le.data || []) as Lesson[];

  return (
    <div>
      <h1 className="text-2xl font-bold">Insights</h1>
      <p className="text-sm text-zinc-500 mt-1">
        Long-horizon theses and the lessons the system has learned from its
        own scored mistakes.
      </p>

      <h2 className="text-lg font-semibold mt-6">Theses</h2>
      {theses.length === 0 ? (
        <p className="text-sm text-zinc-500 mt-2">
          No theses yet — they emerge only after repeated same-direction
          evidence accumulates. Conviction is earned, not conjured.
        </p>
      ) : (
        <div className="grid md:grid-cols-2 gap-4 mt-3">
          {theses.map((t) => (
            <div key={t.id} className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
              <div className="flex items-center justify-between">
                <span className="font-bold">{t.ticker}
                  <span className="ml-2 text-xs text-zinc-500">{t.direction}</span>
                </span>
                <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold border ${STATUS_STYLE[t.status]}`}>
                  {t.status}
                </span>
              </div>
              <p className="text-sm text-zinc-300 mt-3">{t.thesis_text}</p>
              <div className="mt-3"><ConfidenceBar value={t.confidence} /></div>
              <div className="text-xs text-zinc-600 mt-2">
                updated {t.updated_at?.slice(0, 10)}
              </div>
            </div>
          ))}
        </div>
      )}

      <h2 className="text-lg font-semibold mt-8">Lessons</h2>
      {lessons.length === 0 ? (
        <p className="text-sm text-zinc-500 mt-2">
          No lessons yet — these are distilled weekly from decisions the
          market proved wrong, by code-scored outcomes.
        </p>
      ) : (
        <ul className="mt-3 space-y-2">
          {lessons.map((l) => (
            <li key={l.id} className="bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-3 text-sm text-zinc-300">
              {l.lesson_text}
              <span className="text-xs text-zinc-600 ml-2">{l.created_at?.slice(0, 10)}</span>
            </li>
          ))}
        </ul>
      )}
      <Disclaimer />
    </div>
  );
}
