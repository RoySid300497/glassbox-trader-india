// proxying six months of daily candles from yahoo's public chart api
import { NextResponse } from "next/server";

export async function GET(_req: Request,
  { params }: { params: Promise<{ ticker: string }> }) {
  const { ticker } = await params;
  if (!/^[A-Za-z]{1,5}([.-][A-Za-z])?$/.test(ticker)) {
    return NextResponse.json({ error: "bad ticker" }, { status: 400 });
  }
  const sym = ticker.toUpperCase().replace(".", "-");
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${sym}?range=6mo&interval=1d`;
  try {
    const r = await fetch(url, { next: { revalidate: 900 } });
    const j = await r.json();
    const res = j?.chart?.result?.[0];
    const q = res?.indicators?.quote?.[0];
    const candles = (res?.timestamp || [])
      .map((t: number, i: number) => ({
        time: t, open: q.open[i], high: q.high[i],
        low: q.low[i], close: q.close[i],
      }))
      .filter((c: { open: number | null }) => c.open != null);
    return NextResponse.json({ candles });
  } catch {
    return NextResponse.json({ candles: [] });
  }
}
