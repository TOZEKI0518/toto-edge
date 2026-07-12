const navItems = [
  { href: "/", label: "Dashboard" },
  { href: "/backtest", label: "Backtest" },
  { href: "/backtest/summary", label: "Summary" },
  { href: "/backtest/inspect", label: "Inspect" },
  { href: "/runs", label: "AI Lab" },
  { href: "/debug", label: "Debug" },
];

export function AppNav() {
  return (
    <nav className="mt-6 flex flex-wrap gap-2">
      {navItems.map((item) => (
        <a
          key={item.href}
          href={item.href}
          className="inline-flex rounded-full border border-white/10 bg-white/[0.04] px-4 py-2 text-sm text-white/70 hover:border-cyan-300/40 hover:text-cyan-300"
        >
          {item.label}
        </a>
      ))}
    </nav>
  );
}
