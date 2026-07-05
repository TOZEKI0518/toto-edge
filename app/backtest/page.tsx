import Link from "next/link";
import { getOutcomeLabel } from "@/services/fixturePredictionService";
import { runBacktestForRound } from "@/services/backtestService";
import { AppNav } from "@/components/AppNav";

type BacktestPageProps = {
  searchParams?: Promise<{
    round?: string;
  }>;
};

const presetRounds = ["1637", "1636", "1635", "1634", "1633"];

export default async function BacktestPage({ searchParams }: BacktestPageProps) {
  const params = await searchParams;
  const roundNumber = params?.round ?? "1637";
  const result = await runBacktestForRound(roundNumber);

  return (
    <main className="min-h-screen bg-[#05060A] p-6 text-white">
      <div className="mx-auto max-w-5xl">
        <p className="text-sm text-cyan-300">Backtest</p>
        <h1 className="mt-2 text-3xl font-bold">
          {result.round} Backtest Result
        </h1>

        <AppNav />

        <section className="mt-6 rounded-3xl border border-white/10 bg-white/[0.04] p-5">
          <p className="text-sm text-white/50">Round Select</p>
          <div className="mt-3 flex flex-wrap gap-2">
            {presetRounds.map((round) => (
              <Link
                key={round}
                href={`/backtest?round=${round}`}
                className={
                  round === roundNumber
                    ? "rounded-full bg-cyan-400 px-4 py-2 text-sm font-bold text-black"
                    : "rounded-full bg-white/10 px-4 py-2 text-sm text-white/70"
                }
              >
                第{round}回
              </Link>
            ))}
          </div>
        </section>

        <section className="mt-6 rounded-3xl border border-cyan-400/20 bg-cyan-400/10 p-6">
          <p className="text-sm text-white/60">Hit Rate</p>
          <p className="mt-2 text-6xl font-black">{result.hitRate}%</p>
          <p className="mt-2 text-white/70">
            {result.hitCount} / {result.totalMatches} J.League matches
          </p>
          <p className="mt-2 text-sm text-white/45">
            Jリーグ順位表と照合できた試合のみを検証対象にしています。
          </p>
        </section>

        {result.totalMatches === 0 && (
          <section className="mt-6 rounded-3xl border border-yellow-400/20 bg-yellow-400/10 p-5">
            <p className="text-yellow-200">
              この開催回はJリーグ順位表と照合できる試合がありません。
              ワールドカップ・代表戦・海外クラブ等はバックテスト対象外です。
            </p>
          </section>
        )}

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