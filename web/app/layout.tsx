import type { Metadata } from "next";
import Link from "next/link";
import { VoiceDock } from "@/components/VoiceDock";
import "./globals.css";

export const metadata: Metadata = {
  title: "Ambit — a registry of inspection competence",
  description:
    "A registry of anomaly-detection specialists on MongoDB Atlas. It knows what it can inspect, and refuses what it cannot.",
};

const NAV: ReadonlyArray<{ href: string; label: string }> = [
  { href: "/", label: "Inspect" },
  { href: "/registry", label: "Registry" },
  { href: "/trends", label: "Trends" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <a href="#main" className="skip-link">
          Skip to content
        </a>

        <header className="masthead">
          <div className="shell">
            <div className="masthead-inner">
              <Link href="/" className="wordmark">
                Ambit
              </Link>
              <nav className="nav" aria-label="Primary">
                {NAV.map((item) => (
                  <Link key={item.href} href={item.href}>
                    {item.label}
                  </Link>
                ))}
              </nav>
            </div>
          </div>
        </header>

        <main id="main" className="shell main">
          {children}
        </main>

        {/* Mounted in the layout, not in a route: this is the only point that
            survives an App Router navigation, which is what lets a live WebRTC
            session outlast moving between /, /registry and /trends. */}
        <VoiceDock />
      </body>
    </html>
  );
}
