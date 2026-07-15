import Link from "next/link";
import { AppNav } from "@/components/AppNav";
import { getOutcomeLabel } from "@/services/fixturePredictionService";
import { runBacktestSummary } from "@/services/backtestService";
import { leagueBadgeClass, leagueBadgeLabel } from "@/services/leagueClassifier";
import { summarizeOptimizerTicketBacktests } from "@/services/optimizerBacktestService";

type BacktestSummaryPageProps = {
  searchParams?: Promise<{
    from?: string;
    to?: string;
    filter?: string;
  }>;
};

function buildRoundRange(from?: string | number, to?: string | number) {
  const hasRange = from !== undefined || to !== undefined;
  const start = Number(from ?? 1618);
  const end = Number(to ?? 1637);

  if (!Number.isInteger(start) || !Number.isInteger(end)) {
    return Array.from({ length: 20 }, (_, index) => String(1637 - index));
  }

  const max = Math.max(start, end);
  const min = Math.min(start, end);

  const rounds = Array.from({ length: max - min + 1 }, (_, index) =>
    String(max - index)
  );

  return hasRange ? rounds : rounds.slice(0, 20);
}

function dataLabel(result: Awaited<ReturnType<typeof runBacktestSummary>>["results"][number]) {
  if (result.dataSource === "date_cut" && result.snapshotDate) return `${result.snapshotDate}以前`;
  if (result.dataSource === "round_snapshot" && result.snapshotRoundNo) return `第${result.snapshotRoundNo}回`;
  if (result.dataSource === "snapshot" && result.snapshotDate) return result.snapshotDate;
  if (result.dataSource === "current") return "現在";
  return "なし";
}

export default async function BacktestSummaryPage({
  searchParams,
}: BacktestSummaryPageProps) {
  const params = await searchParams;
  const targetRounds = buildRoundRange(params?.from, params?.to);
  const filter = params?.filter === "all" ? "all" : "jleague";

  let summary: Awaited<ReturnType<typeof runBacktestSummary>> | null = null;
  let errorMessage = "";

  try {
    summary = await runBacktestSummary(targetRounds, { filter });
  } catch (error) {
    errorMessage =
      error instanceof Error
        ? error.message
        : "バックテスト集計の取得に失敗しました。";
  }

  const optimizerSummary = summary
    ? summarizeOptimizerTicketBacktests(summary.results, 50)
    : null;

  return (
    <main className="min-h-screen bg-[#05060A] p-6 text-white">
      <div className="mx-auto max-w-6xl">
        <p className="text-sm text-cyan-300">Backtest</p>
        <h1 className="mt-2 text-3xl font-bold">Backtest Summary</h1>

        <AppNav />

        <div className="mt-6 flex flex-wrap gap-2">
          <a
            href="/backtest/summary?filter=jleague"
            className={
              filter === "jleague"
                ? "rounded-full bg-cyan-400 px-4 py-2 text-sm font-bold text-black"
                : "rounded-full border border-white/10 bg-white/[0.04] px-4 py-2 text-sm text-white/70"
            }
          >
            ⚽ Jリーグのみ
          </a>
          <a
            href="/backtest/summary?filter=all"
            className={
              filter === "all"
                ? "rounded-full bg-cyan-400 px-4 py-2 text-sm font-bold text-black"
                : "rounded-full border border-white/10 bg-white/[0.04] px-4 py-2 text-sm text-white/70"
            }
          >
            全試合
          </a>
        </div>

        <section className="mt-6 rounded-3xl border border-cyan-400/20 bg-cyan-400/10 p-6">
          <p className="text-sm text-white/60">Average Hit Rate</p>
          <p className="mt-2 text-6xl font-black">
            {summary ? summary.averageHitRate : 0}%
          </p>
          <p className="mt-2 text-white/70">
            {summary ? summary.totalHits : 0} / {summary ? summary.totalMatches : 0} {filter === "jleague" ? "J.League matches" : "matches"}
          </p>
          <p className="mt-2 text-sm text-white/45">
            Target rounds: {targetRounds[targetRounds.length - 1]} - {targetRounds[0]} / default recent 20 rounds
          </p>
        </section>


        {optimizerSummary && (
          <section className="mt-6 rounded-3xl border border-emerald-400/20 bg-emerald-400/[0.06] p-6">
            <p className="text-sm text-emerald-200">
              Recommended 50 Tickets Backtest
            </p>
            <h2 className="mt-2 text-2xl font-black">
              おすすめ50口の成績
            </h2>
            <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <div className="rounded-2xl bg-black/20 p-4">
                <p className="text-xs text-white/45">評価開催回</p>
                <p className="mt-2 text-2xl font-black">
                  {optimizerSummary.evaluatedRounds}
                </p>
              </div>
              <div className="rounded-2xl bg-black/20 p-4">
                <p className="text-xs text-white/45">平均 Best Hits</p>
                <p className="mt-2 text-2xl font-black text-cyan-200">
                  {optimizerSummary.averageBestHits}
                </p>
              </div>
              <div className="rounded-2xl bg-black/20 p-4">
                <p className="text-xs text-white/45">当せん相当回</p>
                <p className="mt-2 text-2xl font-black text-emerald-200">
                  {optimizerSummary.winningRounds}
                </p>
              </div>
              <div className="rounded-2xl bg-black/20 p-4">
                <p className="text-xs text-white/45">累計投資額</p>
                <p className="mt-2 text-2xl font-black">
                  {optimizerSummary.totalInvestmentYen.toLocaleString("ja-JP")}円
                </p>
              </div>
            </div>
            <p className="mt-4 text-xs leading-5 text-white/40">
              金額ROIは開催回ごとの払戻金データをまだ保存していないため未表示です。
              現在は各回の最高一致数と1〜3等相当の有無を確認できます。
            </p>
          </section>
        )}


        {summary && (
          <section className="mt-6 grid gap-4 md:grid-cols-3">
            <div className="rounded-3xl border border-cyan-400/20 bg-cyan-400/10 p-5">
              <p className="text-sm text-cyan-200">League Breakdown</p>
              {summary.leagueStats.map((item) => (
                <div key={item.league} className="mt-3 rounded-2xl bg-black/20 p-4">
                  <p className="text-sm text-white/60">{leagueBadgeLabel(item.league)}</p>
                  <p className="mt-1 text-3xl font-black">{item.rate}%</p>
                  <p className="mt-1 text-sm text-white/50">{item.hits} / {item.total}</p>
                </div>
              ))}
            </div>

            <div className="rounded-3xl border border-white/10 bg-white/[0.04] p-5">
              <p className="text-sm text-white/60">Top Accuracy Teams</p>
              <div className="mt-3 space-y-2">
                {summary.topTeams.map((item, index) => (
                  <div key={item.team} className="flex items-center justify-between rounded-2xl bg-cyan-400/10 px-3 py-2 text-sm">
                    <span>{index + 1}. {item.team}</span>
                    <span className="font-bold text-cyan-300">{item.rate}%</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-3xl border border-white/10 bg-white/[0.04] p-5">
              <p className="text-sm text-white/60">Home / Draw / Away</p>
              <div className="mt-3 space-y-2">
                {summary.outcomeAccuracy.map((item) => (
                  <div key={item.outcome} className="flex items-center justify-between rounded-2xl bg-white/[0.04] px-3 py-2 text-sm">
                    <span>{getOutcomeLabel(item.outcome as never)}</span>
                    <span className="font-bold text-cyan-300">{item.rate}%</span>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}

        {errorMessage && (
          <section className="mt-6 rounded-3xl border border-yellow-400/20 bg-yellow-400/10 p-6">
            <p className="text-sm font-bold text-yellow-300">Error</p>
            <p className="mt-2 break-words text-sm text-white/70">{errorMessage}</p>
          </section>
        )}

        {summary && (
          <section className="mt-8 overflow-hidden rounded-3xl border border-white/10">
            <table className="w-full border-collapse text-left text-sm">
              <thead className="bg-white/[0.06] text-white/60">
                <tr>
                  <th className="px-4 py-3">Round</th>
                  <th className="px-4 py-3">Hit</th>
                  <th className="px-4 py-3">Rate</th>
                  <th className="px-4 py-3">Best Hits</th>
                  <th className="px-4 py-3">Data</th>
                  <th className="px-4 py-3">League</th>
                  <th className="px-4 py-3">Detail</th>
                </tr>
              </thead>
              <tbody>
                {summary.results.map((result) => (
                  <tr key={result.round} className="border-t border-cyan-400/15 bg-cyan-400/[0.035]">
                    <td className="border-l-4 border-cyan-400 px-4 py-3 font-bold">{result.round}</td>
                    <td className="px-4 py-3">{result.hitCount} / {result.totalMatches}</td>
                    <td className="px-4 py-3 font-bold text-cyan-300">{result.hitRate}%</td>
                    <td className="px-4 py-3 font-bold text-emerald-300">
                      {optimizerSummary?.rounds.find((item) => item.round === result.round)?.result.maxHitCount ?? "-"} / {result.totalMatches}
                    </td>
                    <td className="px-4 py-3 text-white/60">{dataLabel(result)}</td>
                    <td className="px-4 py-3"><span className={`rounded-full px-2 py-1 text-xs ${leagueBadgeClass(result.matches[0]?.league)}`}>{leagueBadgeLabel(result.matches[0]?.league)}</span></td>
                    <td className="px-4 py-3">
                      <Link href={`/backtest?round=${result.round.replace(/[^\d]/g, "")}&filter=${filter}`} className="text-cyan-300 underline">
                        詳細
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}
      </div>
    </main>
  );
}
