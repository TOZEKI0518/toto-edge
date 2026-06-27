import { AppHeader } from "@/components/AppHeader";
import { HeroPickCard } from "@/components/HeroPickCard";
import { MatchCard } from "@/components/MatchCard";
import { getCurrentTotoRound, getMatches } from "@/services/matchService";
import {
  buildDashboardMatch,
  getBestPick,
} from "@/services/predictionService";

export default async function Home() {
  const rawMatches = await getMatches();
  const totoRound = await getCurrentTotoRound();

  const matches = rawMatches.map(buildDashboardMatch);
  const bestPick = getBestPick(matches);

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
            <p className="text-sm text-white/50">{totoRound} toto</p>
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