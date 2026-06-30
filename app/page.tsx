import { AppHeader } from "@/components/AppHeader";
import { FixturePredictionCard } from "@/components/FixturePredictionCard";
import { sampleFixtures } from "@/data/sampleFixtures";
import { getJleagueStandings } from "@/services/jleagueStandingService";
import { getCurrentTotoRound } from "@/services/matchService";
import {
  getOutcomeLabel,
  predictFixtures,
} from "@/services/fixturePredictionService";
import { buildPredictionInputs } from "@/services/predictionInputService";

export default async function Home() {
  const [standings, totoRound] = await Promise.all([
    getJleagueStandings(),
    getCurrentTotoRound(),
  ]);

  const predictionInputs = buildPredictionInputs(sampleFixtures, standings);
  const predictions = predictFixtures(predictionInputs);

  const bestPrediction = [...predictions].sort(
    (a, b) => b.probability - a.probability
  )[0];

  return (
    <main className="min-h-screen bg-[#05060A] text-white">
      <div className="mx-auto max-w-6xl px-5 py-8">
        <AppHeader />

        <section className="mb-8 rounded-3xl border border-white/10 bg-gradient-to-br from-cyan-500/20 via-blue-500/10 to-purple-500/20 p-6 shadow-2xl">
          <p className="mb-2 text-sm text-white/60">Today&apos;s Best Pick</p>

          <div className="flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
            <div>
              <h2 className="text-4xl font-bold">
                {bestPrediction.homeTeam} vs {bestPrediction.awayTeam}
              </h2>
              <p className="mt-3 text-cyan-200">
                {getOutcomeLabel(bestPrediction.outcome)}
              </p>
            </div>

            <div className="text-right">
              <p className="text-6xl font-black">
                {bestPrediction.probability}%
              </p>
              <p className="mt-2 text-sm text-white/60">
                Confidence {bestPrediction.confidence}
              </p>
            </div>
          </div>
        </section>

        <section>
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h3 className="text-xl font-bold">AI Predictions</h3>
              <p className="mt-1 text-sm text-white/50">
                順位・勝点・得失点・ホーム補正によるV1予測
              </p>
            </div>
            <p className="text-sm text-white/50">{totoRound} toto</p>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            {predictions.map((prediction) => (
              <FixturePredictionCard
                key={prediction.matchNo}
                prediction={prediction}
              />
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}