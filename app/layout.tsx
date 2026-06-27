import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TOTO EDGE",
  description: "toto期待値分析MVP"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ja">
      <body>{children}</body>
    </html>
  );
}
