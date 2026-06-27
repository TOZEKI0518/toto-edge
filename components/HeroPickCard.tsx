type HeroPickCardProps = {
  home: string;
  away: string;
  prediction: string;
  probability: number;
  edge: number;
};

export function HeroPickCard({
  home,
  away,
  prediction,
  probability,
  edge,
}: HeroPickCardProps) {
  return (
    <section className="mb-8 rounded-3xl border border-white/10 bg-gradient-to-br from-cyan-500/20 via-blue-500/10 to-purple-500/20 p-6 shadow-2xl">
      <p className="mb-2 text-sm text-white/60">Today&apos;s Best Pick</p>

      <div className="flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
        <div>
          <h2 className="text-4xl font-bold">
            {home} vs {away}
          </h2>
          <p className="mt-3 text-cyan-200">{prediction}</p>
        </div>

        <div className="text-right">
          <p className="text-6xl font-black">{probability}%</p>
          <p className="mt-2 text-sm text-white/60">Expected Value +{edge}%</p>
        </div>
      </div>
    </section>
  );
}