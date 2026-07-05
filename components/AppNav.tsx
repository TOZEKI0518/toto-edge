import Link from "next/link";

const navItems = [
  { href: "/", label: "Dashboard" },
  { href: "/backtest", label: "Backtest" },
  { href: "/runs", label: "AI Lab" },
  { href: "/debug", label: "Debug" },
];

export function AppNav() {
  return (
    <nav className="mt-6 flex flex-wrap gap-2">
      {navItems.map((item) => (
        <Link
          key={item.href}
          href={item.href}
          className="rounded-full border border-white/10 bg-white/[0.04] px-4 py-2 text-sm text-white/70 hover:border-cyan-300/40 hover:text-cyan-300"
        >
          {item.label}
        </Link>
      ))}
    </nav>
  );
}