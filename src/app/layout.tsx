import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Hipvan's Blog",
  description: "AI, code, and things I learn along the way.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <nav className="border-b border-[var(--color-border)] px-6 py-4">
          <div className="max-w-3xl mx-auto flex justify-between items-center">
            <a href="/" className="font-bold text-lg">
              Hipvan
            </a>
            <div className="flex gap-6 text-sm text-[var(--color-muted)]">
              <a href="/" className="hover:text-[var(--color-foreground)]">
                Blog
              </a>
              <a href="/about" className="hover:text-[var(--color-foreground)]">
                About
              </a>
              <a
                href="https://github.com/hipvan"
                target="_blank"
                className="hover:text-[var(--color-foreground)]"
              >
                GitHub
              </a>
            </div>
          </div>
        </nav>
        <main className="max-w-3xl mx-auto px-6 py-12">{children}</main>
      </body>
    </html>
  );
}
