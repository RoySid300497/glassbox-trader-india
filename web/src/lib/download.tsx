"use client";
// exporting any row set as a downloadable csv file

export default function DownloadCsvButton({ rows, filename, label }:
  { rows: Record<string, unknown>[]; filename: string; label: string }) {
  const click = () => {
    if (!rows.length) return;
    const headers = Object.keys(rows[0]);
    const esc = (v: unknown) => {
      const s = v === null || v === undefined ? "" : String(v);
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const csv = [headers.join(","),
      ...rows.map((r) => headers.map((h) => esc(r[h])).join(","))].join("\n");
    const url = URL.createObjectURL(
      new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };
  return (
    <button onClick={click} disabled={!rows.length}
      className="text-xs px-3 py-1.5 rounded-lg border border-zinc-700 text-zinc-300 hover:border-zinc-500 hover:text-zinc-100 disabled:opacity-40 transition-colors">
      ↓ {label}
    </button>
  );
}
