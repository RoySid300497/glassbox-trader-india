// managing the user watchlist config behind a server-side admin token
import { NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";

export const dynamic = "force-dynamic";

const TICKER = /^[A-Z]{1,5}(\.[A-Z])?$/;
const MAX_PINS = 30;

function admin() {
  const url =
    process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_KEY;
  if (!url || !key) throw new Error("service credentials not configured");
  return createClient(url, key);
}

export async function GET() {
  try {
    const db = admin();
    const { data } = await db
      .from("config")
      .select("key,value")
      .in("key", ["user_watchlist", "watchlist_fill"]);
    const map = new Map(
      ((data || []) as { key: string; value: string | null }[])
        .map((r) => [r.key, r.value ?? ""]),
    );
    const tickers = String(map.get("user_watchlist") || "")
      .split(",")
      .map((t) => t.trim().toUpperCase())
      .filter(Boolean);
    const fill = map.get("watchlist_fill") === "empty" ? "empty" : "screener";
    return NextResponse.json({ tickers, fill });
  } catch {
    return NextResponse.json({ tickers: [], fill: "screener" });
  }
}

export async function POST(req: Request) {
  const token = req.headers.get("x-admin-token") || "";
  const expected = process.env.WATCHLIST_ADMIN_TOKEN;
  if (!expected || token !== expected) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  let body: { tickers?: unknown; fill?: unknown };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "bad json" }, { status: 400 });
  }

  const raw: unknown[] = Array.isArray(body.tickers) ? body.tickers : [];
  const tickers = [
    ...new Set(raw.map((t) => String(t).trim().toUpperCase())),
  ]
    .filter((t) => TICKER.test(t))
    .slice(0, MAX_PINS);
  const fill = body.fill === "empty" ? "empty" : "screener";

  try {
    const db = admin();
    const { error } = await db.from("config").upsert([
      { key: "user_watchlist", value: tickers.join(",") },
      { key: "watchlist_fill", value: fill },
    ]);
    if (error) throw error;
    return NextResponse.json({ ok: true, tickers, fill });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "write failed" },
      { status: 500 },
    );
  }
}
