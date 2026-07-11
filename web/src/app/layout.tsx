// wrapping every page with the dark shell and navigation
import "./globals.css";
import Link from "next/link";

export const metadata = {
  title: "glassbox trader",
  description: "Independent AI panels debate every stock decision — and show their work.",
};

const NAV = [
  ["/", "Briefing"],
  ["/scan", "Scan"],
  ["/signals", "Signals"],
  ["/news", "News"],
  ["/track", "Track record"],
  ["/performance", "Performance"],
  ["/reports", "Reports"],
  ["/insights", "Insights"],
  ["/positions", "Positions"],
  ["/watchlist", "Watchlist"],
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-zinc-950 text-zinc-100 min-h-screen antialiased">
        <nav className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur sticky top-0 z-10">
          <div className="max-w-6xl mx-auto px-6 py-4 flex items-center gap-8 overflow-x-auto">
            <Link href="/" className="font-bold text-lg tracking-tight whitespace-nowrap">
              glassbox<span className="text-zinc-500">trader</span>
            </Link>
            <div className="flex gap-6 text-sm text-zinc-400">
              {NAV.map(([href, label]) => (
                <Link key={href} href={href}
                      className="hover:text-zinc-100 whitespace-nowrap transition-colors">
                  {label}
                </Link>
              ))}
            </div>
          </div>
        </nav>
        <main className="max-w-6xl mx-auto px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
