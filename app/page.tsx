import { AppHeader } from "@/components/AppHeader";
import { HeroPickCard } from "@/components/HeroPickCard";
import { MatchCard } from "@/components/MatchCard";
import { sampleMatches } from "@/data/sampleMatches";
import { calculatePrediction, getPickLabel } from "@/lib/predictionEngine";

const matches = sampleMatches.map((match) => {
  const prediction = calculatePrediction(match);
  const pickKey = prediction.pick.toLowerCase() as "home" | "draw" | "away";

  return {
    id: match.matchNo,
    home: match.homeTeam.shortName,
    away: match.awayTeam.shortName,
    prediction: getPickLabel(prediction.pick),
    probability: prediction.probabilities[pickKey],
    edge: prediction.expectedValues?.[pickKey] ?? 0,
    confidence: prediction.confidence,
  };
});

export default function Home() {
  const bestPick = [...matches].sort((a, b) => b.edge - a.edge)[0];

  return (
    <main className="min-h-screen bg-[#05060A] text-white">
      <div className="mx-auto max-w-6xl px-5 py-8">
        <AppHeader />

        <HeroPickCard
          home={bestPick.home}
          away={bestPick.away}
          prediction={bestPick.prediction}
          probability={bestPick.probability}
          edge={bestPick.edge}
        />

        <section>
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-xl font-bold">Match Predictions</h3>
            <p className="text-sm text-white/50">第XXXX回 toto</p>
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            {matches.map((match) => (
              <MatchCard key={match.id} {...match} />
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}