import Link from "next/link";
import { AppNav } from "@/components/AppNav";
import { getOutcomeLabel } from "@/services/fixturePredictionService";
import { runBacktestForRound } from "@/services/backtestService";
import { leagueBadgeClass, leagueBadgeLabel } from "@/services/leagueClassifier";
import { runOptimizerTicketBacktest } from "@/services/optimizerBacktestService";

type BacktestPageProps = {
  searchParams?: Promise<{
    round?: string;
    filter?: string;
  }>;
};

const presetRounds = Array.from({ length: 138 }, (_, index) =>
  String(1637 - index)
);

export default async function BacktestPage({
  searchParams,
}: BacktestPageProps) {
  const params = await searchParams;
  const roundNumber = params?.round ?? "1637";
  const filter = params?.filter === "all" ? "all" : "jleague";

  let result: Awaited<ReturnType<typeof runBacktestForRound>> | null = null;
  let errorMessage = "";

  try {
    result = await runBacktestForRound(roundNumber, { filter });
  } catch (error) {
    errorMessage =
      error instanceof Error
        ? error.message
        : "バックテストの取得に失敗しました。";
  }

  const optimizerBacktest = result
    ? runOptimizerTicketBacktest(result, 50)
    : null;

  return (
    <main className="min-h-screen bg-[#05060A] p-6 text-white">
      <div className="mx-auto max-w-5xl">
        <p className="text-sm text-cyan-300">Backtest</p>
        <h1 className="mt-2 text-3xl font-bold">
          {result ? `${result.round} Backtest Result` : "Backtest"}
        </h1>

        <AppNav />

        <div className="mt-6 flex flex-wrap gap-2">
          <a className={filter === "jleague" ? "rounded-full bg-cyan-400 px-4 py-2 text-sm font-bold text-black" : "rounded-full border border-white/10 bg-white/[0.04] px-4 py-2 text-sm text-white/70"} href={`/backtest?round=${roundNumber}&filter=jleague`}>
            ⚽ Jリーグのみ
          </a>
          <a className={filter === "all" ? "rounded-full bg-cyan-400 px-4 py-2 text-sm font-bold text-black" : "rounded-full border border-white/10 bg-white/[0.04] px-4 py-2 text-sm text-white/70"} href={`/backtest?round=${roundNumber}&filter=all`}>
            全試合
          </a>
        </div>

        <section className="mt-6 rounded-3xl border border-white/10 bg-white/[0.04] p-5">
          <p className="text-sm text-white/50">Round Select</p>
          <div className="mt-3 flex max-h-40 flex-wrap gap-2 overflow-y-auto pr-2">
            {presetRounds.map((round) => (
              <Link
                key={round}
                href={`/backtest?round=${round}&filter=${filter}`}
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

        {errorMessage && (
          <section className="mt-6 rounded-3xl border border-red-500/20 bg-red-500/10 p-6">
            <h2 className="text-xl font-bold text-red-300">
              バックテストを実行できませんでした
            </h2>
            <pre className="mt-4 overflow-auto rounded-xl bg-black/30 p-4 text-xs text-red-200">
              {errorMessage}
            </pre>
          </section>
        )}

        {result && (
          <>
            <section className="mt-6 rounded-3xl border border-cyan-400/20 bg-cyan-400/10 p-6">
              <p className="text-sm text-white/60">Hit Rate</p>
              <p className="mt-2 text-6xl font-black">{result.hitRate}%</p>
              <p className="mt-2 text-white/70">
                的中: {result.hitCount} / {result.totalMatches} {filter === "jleague" ? "J.League matches" : "matches"}
              </p>
              <p className="mt-2 text-white/70">
                照合率: {result.totalMatches} / {result.totalFixtures} matches (
                {result.coverageRate}%)
              </p>
              <p className="mt-2 text-sm text-white/45">
                使用データ：
                {result.dataSource === "date_cut" && result.snapshotDate
                  ? `${result.snapshotDate}以前（過去${result.sourceRoundCount ?? 0}回）`
                  : result.dataSource === "round_snapshot" && result.snapshotRoundNo
                  ? `第${result.snapshotRoundNo}回 Snapshot`
                  : result.dataSource === "snapshot" && result.snapshotDate
                  ? `${result.snapshotDate} Snapshot`
                  : result.dataSource === "current"
                  ? "現在順位表"
                  : "なし"}
              </p>
            </section>


            {optimizerBacktest && (
              <section className="mt-6 rounded-3xl border border-emerald-400/20 bg-emerald-400/[0.06] p-6">
                <div className="flex flex-col gap-5 md:flex-row md:items-end md:justify-between">
                  <div>
                    <p className="text-sm text-emerald-200">
                      Recommended 50 Tickets Backtest
                    </p>
                    <h2 className="mt-2 text-2xl font-black">
                      50口買っていた場合
                    </h2>
                    <p className="mt-2 text-sm text-white/50">
                      投資額 {optimizerBacktest.investmentYen.toLocaleString("ja-JP")}円
                      {" / "}実結果 {optimizerBacktest.actualPicks || "-"}
                    </p>
                  </div>
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                    <div className="rounded-2xl bg-black/20 p-3">
                      <p className="text-xs text-white/45">Best Hits</p>
                      <p className="mt-1 text-2xl font-black text-cyan-200">
                        {optimizerBacktest.maxHitCount} / {optimizerBacktest.matchCount}
                      </p>
                    </div>
                    <div className="rounded-2xl bg-black/20 p-3">
                      <p className="text-xs text-white/45">
                        {optimizerBacktest.matchCount === 13
                          ? "1等相当"
                          : `${optimizerBacktest.matchCount}/${optimizerBacktest.matchCount}`}
                      </p>
                      <p className="mt-1 text-2xl font-black">
                        {optimizerBacktest.matchCount === 13
                          ? optimizerBacktest.firstPrizeEquivalentCount
                          : optimizerBacktest.tickets.filter(
                              (ticket) =>
                                ticket.hitCount === optimizerBacktest.matchCount
                            ).length}
                      </p>
                    </div>
                    <div className="rounded-2xl bg-black/20 p-3">
                      <p className="text-xs text-white/45">
                        {optimizerBacktest.matchCount === 13
                          ? "2等相当"
                          : `${Math.max(optimizerBacktest.matchCount - 1, 0)}/${optimizerBacktest.matchCount}`}
                      </p>
                      <p className="mt-1 text-2xl font-black">
                        {optimizerBacktest.matchCount === 13
                          ? optimizerBacktest.secondPrizeEquivalentCount
                          : optimizerBacktest.tickets.filter(
                              (ticket) =>
                                ticket.hitCount === optimizerBacktest.matchCount - 1
                            ).length}
                      </p>
                    </div>
                    <div className="rounded-2xl bg-black/20 p-3">
                      <p className="text-xs text-white/45">
                        {optimizerBacktest.matchCount === 13
                          ? "3等相当"
                          : `${Math.max(optimizerBacktest.matchCount - 2, 0)}/${optimizerBacktest.matchCount}`}
                      </p>
                      <p className="mt-1 text-2xl font-black">
                        {optimizerBacktest.matchCount === 13
                          ? optimizerBacktest.thirdPrizeEquivalentCount
                          : optimizerBacktest.tickets.filter(
                              (ticket) =>
                                ticket.hitCount === optimizerBacktest.matchCount - 2
                            ).length}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="mt-5 overflow-x-auto">
                  <table className="min-w-full text-left text-sm">
                    <thead className="text-xs text-white/40">
                      <tr className="border-b border-white/10">
                        <th className="px-3 py-3">#</th>
                        <th className="px-3 py-3">Picks</th>
                        <th className="px-3 py-3">Hits</th>
                        <th className="px-3 py-3">判定</th>
                      </tr>
                    </thead>
                    <tbody>
                      {optimizerBacktest.tickets.map((ticket) => (
                        <tr key={ticket.rank} className="border-b border-white/[0.06]">
                          <td className="px-3 py-3 text-white/45">{ticket.rank}</td>
                          <td className="px-3 py-3 font-mono tracking-wider">{ticket.picks}</td>
                          <td className="px-3 py-3 font-bold text-cyan-200">
                            {ticket.hitCount} / {optimizerBacktest.matchCount}
                          </td>
                          <td className="px-3 py-3">
                            {optimizerBacktest.matchCount === 13
                              ? ticket.isFirstPrizeEquivalent
                                ? "1等相当"
                                : ticket.isSecondPrizeEquivalent
                                ? "2等相当"
                                : ticket.isThirdPrizeEquivalent
                                ? "3等相当"
                                : "-"
                              : ticket.hitCount === optimizerBacktest.matchCount
                              ? `${optimizerBacktest.matchCount}/${optimizerBacktest.matchCount}`
                              : ticket.hitCount === optimizerBacktest.matchCount - 1
                              ? `${optimizerBacktest.matchCount - 1}/${optimizerBacktest.matchCount}`
                              : ticket.hitCount === optimizerBacktest.matchCount - 2
                              ? `${optimizerBacktest.matchCount - 2}/${optimizerBacktest.matchCount}`
                              : "-"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <p className="mt-4 text-xs leading-5 text-white/40">
                  {optimizerBacktest.monetaryRoiNote}
                </p>
              </section>
            )}

            {result.totalMatches === 0 && (
              <section className="mt-6 rounded-3xl border border-yellow-400/20 bg-yellow-400/10 p-5">
                <p className="text-yellow-200">
                  この開催回は対象条件で照合できる試合がありません。
                </p>
              </section>
            )}

            {result.unmatchedTeams && result.unmatchedTeams.length > 0 && (
              <section className="mt-6 rounded-3xl border border-yellow-400/20 bg-yellow-400/10 p-5">
                <p className="text-sm text-yellow-100">未照合チーム</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  {result.unmatchedTeams.map((team) => (
                    <span
                      key={team}
                      className="rounded-full bg-black/30 px-3 py-1 text-sm text-yellow-100"
                    >
                      {team}
                    </span>
                  ))}
                </div>
              </section>
            )}

            <section className="mt-8 grid gap-4">
              {result.matches.map((match) => (
                <article
                  key={match.matchNo}
                  className="rounded-3xl border border-cyan-400/20 bg-cyan-400/[0.045] p-5 shadow-[inset_4px_0_0_rgba(34,211,238,0.75)]"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <div className="flex flex-wrap items-center gap-2">
                        <p className="text-sm text-white/50">Match {match.matchNo}</p>
                        <span className={`rounded-full px-2 py-1 text-xs ${leagueBadgeClass(match.league)}`}>{leagueBadgeLabel(match.league)}</span>
                      </div>
                      <h2 className="mt-1 text-xl font-bold">
                        {match.homeTeam} vs {match.awayTeam}
                      </h2>
                      <p className="mt-2 text-white/60">
                        AI: {getOutcomeLabel(match.predictedOutcome)}
                        {" / "}
                        Result: {getOutcomeLabel(match.actualOutcome)}
                      </p>
                      {match.factors && match.factors.length > 0 && (
                        <div className="mt-4 grid gap-2 md:grid-cols-2">
                          {match.factors
                            .filter((factor) => ["総合ELO", "Home/Away ELO", "直近5試合フォーム", "対戦相性"].includes(factor.label))
                            .map((factor) => (
                              <div key={factor.label} className="rounded-2xl bg-black/20 px-3 py-2">
                                <div className="flex items-center justify-between gap-3">
                                  <span className="text-xs text-white/50">{factor.label}</span>
                                  <span className={factor.score >= 0 ? "text-sm font-bold text-cyan-200" : "text-sm font-bold text-red-200"}>
                                    {factor.score >= 0 ? "+" : ""}{factor.score}
                                  </span>
                                </div>
                                <p className="mt-1 text-xs text-white/45">{factor.description}</p>
                              </div>
                            ))}
                        </div>
                      )}
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
          </>
        )}
      </div>
    </main>
  );
}
