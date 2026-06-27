import { sampleMatches } from "@/data/sampleMatches";

  const matches = sampleMatches.map((match) => ({
    id: match.matchNo,
    home: match.homeTeam.shortName,
    away: match.awayTeam.shortName,
    prediction: "HOME WIN",
    probability: Math.round(
      50 +
        (match.awayStats.rank - match.homeStats.rank) * 2 +
        (match.homeStats.recentFormPoints - match.awayStats.recentFormPoints)
    ),
    edge: match.voteRates
      ? Math.round(
          50 +
            (match.awayStats.rank - match.homeStats.rank) * 2 +
            (match.homeStats.recentFormPoints - match.awayStats.recentFormPoints) -
            match.voteRates.home
        )
      : 0,
    confidence: "B+",
  }));

export default function Home() {
  const bestPick = matches[0];

  return (
    <main className="min-h-screen bg-[#05060A] text-white">
      <div className="mx-auto max-w-6xl px-5 py-8">
        <header className="mb-8 flex items-center justify-between">
          <div>
            <p className="text-sm text-cyan-300">AI Football Forecast</p>
            <h1 className="mt-1 text-3xl font-bold tracking-tight">
              TOTO EDGE
            </h1>
          </div>

          <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-white/70">
            Ver 0.1
          </div>
        </header>

        <section className="mb-8 rounded-3xl border border-white/10 bg-gradient-to-br from-cyan-500/20 via-blue-500/10 to-purple-500/20 p-6 shadow-2xl">
          <p className="mb-2 text-sm text-white/60">Today&apos;s Best Pick</p>

          <div className="flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
            <div>
              <h2 className="text-4xl font-bold">
                {bestPick.home} vs {bestPick.away}
              </h2>
              <p className="mt-3 text-cyan-200">{bestPick.prediction}</p>
            </div>

            <div className="text-right">
              <p className="text-6xl font-black">{bestPick.probability}%</p>
              <p className="mt-2 text-sm text-white/60">
                Expected Value +{bestPick.edge}%
              </p>
            </div>
          </div>
        </section>

        <section>
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-xl font-bold">Match Predictions</h3>
            <p className="text-sm text-white/50">第XXXX回 toto</p>
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            {matches.map((match) => (
              <article
                key={match.id}
                className="rounded-3xl border border-white/10 bg-white/[0.04] p-5 transition hover:-translate-y-1 hover:bg-white/[0.07]"
              >
                <div className="mb-4 flex items-center justify-between">
                  <span className="rounded-full bg-white/10 px-3 py-1 text-xs text-white/60">
                    Match {match.id}
                  </span>
                  <span className="rounded-full bg-cyan-400/10 px-3 py-1 text-xs text-cyan-300">
                    Confidence {match.confidence}
                  </span>
                </div>

                <h4 className="text-xl font-bold">
                  {match.home} vs {match.away}
                </h4>

                <div className="mt-5">
                  <div className="mb-2 flex justify-between text-sm">
                    <span className="text-white/60">{match.prediction}</span>
                    <span>{match.probability}%</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-white/10">
                    <div
                      className="h-full rounded-full bg-cyan-300"
                      style={{ width: `${match.probability}%` }}
                    />
                  </div>
                </div>

                <div className="mt-5 rounded-2xl bg-black/30 p-4">
                  <p className="text-sm text-white/50">Expected Value</p>
                  <p className="mt-1 text-2xl font-bold text-cyan-300">
                    +{match.edge}%
                  </p>
                </div>
              </article>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}