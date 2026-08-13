import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { ThemeToggle } from "@/components/ThemeToggle";

export const metadata: Metadata = {
  title: "GridSight — vector-routed inspection registry",
  description:
    "MongoDB-backed registry of anomaly-detection specialists, routed by vector search over what each model was trained on.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:rounded focus:px-3 focus:py-2"
          style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
        >
          Skip to content
        </a>
        <header
          className="sticky top-0 z-40 backdrop-blur"
          style={{ background: "color-mix(in srgb, var(--bg) 88%, transparent)", borderBottom: "1px solid var(--border)" }}
        >
          <nav className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3 sm:px-6">
            <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight">
              <span
                aria-hidden
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{ background: "var(--accent)" }}
              />
              GridSight
            </Link>
            <div className="flex flex-1 items-center gap-4 text-sm">
              <Link href="/" className="hover:underline">
                Inspect
              </Link>
              <Link href="/registry" className="hover:underline">
                Registry
              </Link>
              <Link href="/trends" className="hover:underline">
                Trends
              </Link>
              <Link href="/voice" className="hover:underline">
                Voice
              </Link>
            </div>
            <ThemeToggle />
          </nav>
        </header>
        <main id="main" className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6">
          {children}
        </main>
      </body>
    </html>
  );
}
