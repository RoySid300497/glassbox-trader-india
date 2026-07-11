// pinning tickers into tomorrow's debate and choosing how slots get filled
"use client";

import { useEffect, useState } from "react";
import { Disclaimer } from "@/lib/ui";

const TICKER = /^[A-Z]{1,5}(\.[A-Z])?$/;

export default function Watchlist() {
  const [tickers, setTickers] = useState<string[]>([]);
  const [fill, setFill] = useState<"screener" | "empty">("screener");
  const [input, setInput] = useState("");
  const [token, setToken] = useState("");
  const [status, setStatus] = useState("");
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setToken(window.localStorage.getItem("gb_admin_token") || "");
    fetch("/api/watchlist")
      .then((r) => r.json())
      .then((d) => {
        setTickers(d.tickers || []);
        setFill(d.fill === "empty" ? "empty" : "screener");
      })
      .catch(() => setStatus("could not load current watchlist"));
  }, []);

  function add() {
    const t = input.trim().toUpperCase();
    if (!TICKER.test(t)) {
      setStatus(`"${t}" is not a valid ticker`);
      return;
    }
    if (!tickers.includes(t)) {
      setTickers([...tickers, t]);
      setDirty(true);
    }
    setInput("");
    setStatus("");
  }

  function remove(t: string) {
    setTickers(tickers.filter((x) => x !== t));
    setDirty(true);
  }

  async function save() {
    setSaving(true);
    setStatus("");
    window.localStorage.setItem("gb_admin_token", token);
    try {
      const r = await fetch("/api/watchlist", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-admin-token": token,
        },
        body: JSON.stringify({ tickers, fill }),
      });
      const d = await r.json();
      if (!r.ok) {
        setStatus(d.error === "unauthorized"
          ? "wrong admin token"
          : `save failed: ${d.error}`);
      } else {
        setTickers(d.tickers);
        setDirty(false);
        setStatus("saved — the engine reads this on its next daily run");
      }
    } catch {
      setStatus("save failed: network error");
    }
    setSaving(false);
  }

  return (
    <div>
      <h1 className="text-2xl font-bold">Watchlist</h1>
      <p className="text-sm text-zinc-500 mt-1">
        Pinned tickers debate first every morning, ahead of the screener,
        bypassing the cooldown. Pins stay active until removed here.
      </p>

      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 mt-6">
        <div className="flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && add()}
            placeholder="add ticker (e.g. NVDA)"
            className="bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2
                       text-sm w-48 placeholder-zinc-600 focus:outline-none
                       focus:border-zinc-600"
          />
          <button
            onClick={add}
            className="px-4 py-2 rounded-lg text-sm font-semibold
                       bg-sky-500/15 text-sky-400 border border-sky-500/30
                       hover:bg-sky-500/25 transition-colors"
          >
            Add
          </button>
        </div>

        <div className="flex flex-wrap gap-2 mt-4 min-h-8">
          {tickers.length === 0 && (
            <span className="text-sm text-zinc-600">
              no pins — the screener fills all slots
            </span>
          )}
          {tickers.map((t) => (
            <span
              key={t}
              className="flex items-center gap-2 px-3 py-1 rounded-full
                         text-sm font-medium bg-zinc-800 border
                         border-zinc-700"
            >
              {t}
              <button
                onClick={() => remove(t)}
                aria-label={`remove ${t}`}
                className="text-zinc-500 hover:text-rose-400
                           transition-colors"
              >
                ×
              </button>
            </span>
          ))}
        </div>

        <div className="mt-6">
          <div className="text-xs text-zinc-500 mb-2">
            Remaining debate slots
          </div>
          <div className="flex gap-2">
            {(
              [
                ["screener", "Screener fills the rest"],
                ["empty", "Pins only"],
              ] as const
            ).map(([value, label]) => (
              <button
                key={value}
                onClick={() => {
                  setFill(value);
                  setDirty(true);
                }}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold
                            border transition-colors ${
                  fill === value
                    ? "bg-sky-500/15 text-sky-400 border-sky-500/30"
                    : "bg-zinc-950 text-zinc-500 border-zinc-800 hover:text-zinc-300"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          {fill === "empty" && (
            <p className="text-xs text-zinc-600 mt-2">
              Pins-only mode still runs the morning scan for the Scan page;
              with no pins set, the screener takes over so no day is wasted.
            </p>
          )}
        </div>

        <div className="mt-6 flex items-center gap-2">
          <input
            type="password"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            placeholder="admin token"
            className="bg-zinc-950 border border-zinc-800 rounded-lg px-3
                       py-2 text-sm w-48 placeholder-zinc-600
                       focus:outline-none focus:border-zinc-600"
          />
          <button
            onClick={save}
            disabled={saving || !dirty}
            className="px-4 py-2 rounded-lg text-sm font-semibold
                       bg-emerald-500/15 text-emerald-400 border
                       border-emerald-500/30 hover:bg-emerald-500/25
                       transition-colors disabled:opacity-40
                       disabled:cursor-not-allowed"
          >
            {saving ? "Saving…" : "Save"}
          </button>
          {status && (
            <span className="text-xs text-zinc-500">{status}</span>
          )}
        </div>
      </div>

      <Disclaimer />
    </div>
  );
}
