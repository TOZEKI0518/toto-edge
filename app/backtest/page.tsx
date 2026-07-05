import { getOutcomeLabel } from "@/services/fixturePredictionService";
import { runBacktestForRound } from "@/services/backtestService";

export default async function BacktestPage() {
  const result = await runBacktestForRound("1637");

  return (
    <main className="min-h-screen bg-[#05060A] p-6 text-white">
      <div className="mx-auto max-w-5xl">
        <p className="text-sm text-cyan-300">Backtest</p>
        <h1 className="mt-2 text-3xl font-bold">
          {result.round} Backtest Result
        </h1>

        <section className="mt-6 rounded-3xl border border-cyan-400/20 bg-cyan-400/10 p-6">
          <p className="text-sm text-white/60">Hit Rate</p>
          <p className="mt-2 text-6xl font-black">{result.hitRate}%</p>
          <p className="mt-2 text-white/70">
            {result.hitCount} / {result.totalMatches} matches
          </p>
        </section>

        <section className="mt-8 grid gap-4">
          {result.matches.map((match) => (
            <article
              key={match.matchNo}
              className="rounded-3xl border border-white/10 bg-white/[0.04] p-5"
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-sm text-white/50">
                    Match {match.matchNo}
                  </p>
                  <h2 className="mt-1 text-xl font-bold">
                    {match.homeTeam} vs {match.awayTeam}
                  </h2>
                  <p className="mt-2 text-white/60">
                    AI: {getOutcomeLabel(match.predictedOutcome)} / Result:{" "}
                    {getOutcomeLabel(match.actualOutcome)}
                  </p>
                </div>

                <div className="text-right">
                  <p
                    className={
                      match.isHit
                        ? "text-3xl font-black text-emerald-300"
                        : "text-3xl font-black text-red-300"
                    }
                  >
                    {match.isHit ? "✅" : "❌"}
                  </p>
                  <p className="mt-1 text-sm text-white/50">
                    {match.probability}% / {match.confidence}
                  </p>
                </div>
              </div>
            </article>
          ))}
        </section>
      </div>
    </main>
  );
}