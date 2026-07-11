// creating a read-only supabase client and shared types
import { createClient } from "@supabase/supabase-js";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL || "http://localhost";
// server-side reads use the service key (server components only — never
// bundled to the browser); anon key remains as a harmless fallback
const key =
  process.env.SUPABASE_SERVICE_KEY ||
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ||
  "anon";

export const supabase = createClient(url, key);

export const SITE_MODE =
  process.env.NEXT_PUBLIC_SITE_MODE || "RESEARCH"; // PAPER via vercel env

export interface KeyPoint { claim: string; evidence_field: string; }

export interface PanelCase {
  stance: string; provider: string; confidence: number; key_points: KeyPoint[];
}

export interface JudgeVote {
  vote: "BUY" | "SELL" | "NO_TRADE"; reason: string;
  provider: string; confidence: number;
}

export interface Decision {
  id: number; decided_at: string; ticker: string;
  action: "BUY" | "SELL" | "NO_TRADE";
  cnn_direction: string; cnn_confidence: number;
  bull_case: { opening: PanelCase[]; rebuttal: PanelCase[] };
  bear_case: { opening: PanelCase[]; rebuttal: PanelCase[] };
  judge_votes: JudgeVote[]; risk_gate_note: string;
  outcome_return_1d: number | null; outcome_label: string | null;
  was_correct: boolean | null; scored_at: string | null;
}

export interface NewsItem {
  id: number; ticker: string; published_at: string; source: string;
  headline: string; summary: string | null; url: string | null;
  sentiment: number | null;
}

export interface Thesis {
  id: number; ticker: string; created_at: string; updated_at: string;
  thesis_text: string; direction: "LONG" | "SHORT";
  confidence: number; status: "ACTIVE" | "WEAKENING" | "CLOSED";
}

export interface Lesson { id: number; created_at: string; lesson_text: string; }

export interface ScreenRow {
  scan_date: string; ticker: string; direction: string;
  confidence: number; score: number;
}

export interface EquityPoint { date: string; equity: number; }

export interface Trade {
  exit_fill_id: string; ticker: string; qty: number;
  entry_price: number; exit_price: number;
  entry_at: string; exit_at: string; pnl: number; pnl_pct: number;
}

export interface WeeklyReport {
  week_of: string;
  stats: {
    decisions: number; scored: number; correct: number; trades: number;
    models: Record<string, { correct: number; scored: number }>;
    new_lessons: string[]; equity?: number; last_equity?: number;
  };
}

export interface ModelPred {
  pred_date: string; ticker: string; model: string;
  direction: string; confidence: number;
  outcome_label: string | null; was_correct: boolean | null;
}
