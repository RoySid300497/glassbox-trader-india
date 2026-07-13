// proxying six months of daily candles from yahoo's public chart api
import { NextResponse } from "next/server";

export async function GET(_req: Request,
  { params }: { params: Promise<{ ticker: string }> }) {
  const { ticker } = await params;
  const ALLOW = new Set(["NIFTY", "NIFTYBANK", "INDIAVIX"]);
  if (!ALLOW.has(ticker.toUpperCase()) &&
      !/^[A-Za-z0-9]{1,12}([&.\-][A-Za-z0-9]{1,10})?(\.[A-Za-z]{1,3})?$/.test(ticker)) {
    return NextResponse.json({ error: "bad ticker" }, { status: 400 });
  }
  // india NSE symbols need the .NS suffix for yahoo's chart api; only fall
  // back to the US share-class dash form when no exchange suffix applies
  const raw = ticker.toUpperCase();
  // index aliases map to yahoo index symbols; equities get the .NS suffix
  const INDEX: Record<string, string> = {
    NIFTY: "%5ENSEI", NIFTYBANK: "%5ENSEBANK", INDIAVIX: "%5EINDIAVIX",
  };
  const sym = INDEX[raw] || (raw.endsWith(".NS") ? raw : raw + ".NS");
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