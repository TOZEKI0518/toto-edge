import Link from "next/link";
import { AppNav } from "@/components/AppNav";
import { runBacktestSummary } from "@/services/backtestService";

type BacktestSummaryPageProps = {
  searchParams?: Promise<{
    from?: string;
    to?: string;
  }>;
};

function buildRoundRange(from: string, to: string) {
  const start = Number(from);
  const end = Number(to);

  if (!Number.isInteger(start) || !Number.isInteger(end)) {
    return ["1637", "1636", "1635", "1634", "1633", "1632", "1631", "1630"];
  }

  const max = Math.max(start, end);
  const min = Math.min(start, end);

  return Array.from({ length: max - min + 1 }, (_, index) =>
    String(max - index)
  );
}

export default async function BacktestSummaryPage({
  searchParams,
}: BacktestSummaryPageProps) {
  const params = await searchParams;
  const targetRounds = buildRoundRange(
    params?.from ?? "1630",
    params?.to ?? "1637"
  );

  const summary = await runBacktestSummary(targetRounds);

  return (
    <main className="min-h-screen bg-[#05060A] p-6 text-white">
      <div className="mx-auto max-w-5xl">
        <p className="text-sm text-cyan-300">Backtest</p>
        <h1 className="mt-2 text-3xl font-bold">Backtest Summary</h1>

        <AppNav />

        <section className="mt-6 rounded-3xl border border-cyan-400/20 bg-cyan-400/10 p-6">
          <p className="text-sm text-white/60">Average Hit Rate</p>
          <p className="mt-2 text-6xl font-black">{summary.averageHitRate}%</p>
          <p className="mt-2 text-white/70">
            {summary.totalHits} / {summary.totalMatches} J.League matches
          </p>
          <p className="mt-2 text-sm text-white/45">
            Target rounds: {targetRounds[targetRounds.length - 1]} - {targetRounds[0]}
          </p>
        </section>

        <section className="mt-8 overflow-hidden rounded-3xl border border-white/10">
          <table className="w-full border-collapse text-left text-sm">
            <thead className="bg-white/[0.06] text-white/60">
              <tr>
                <th className="px-4 py-3">Round</th>
                <th className="px-4 py-3">Hit</th>
                <th className="px-4 py-3">Rate</th>
                <th className="px-4 py-3">Detail</th>
              </tr>
            </thead>
            <tbody>
              {summary.results.map((result) => (
                <tr key={result.round} className="border-t border-white/10">
                  <td className="px-4 py-3">{result.round}</td>
                  <td className="px-4 py-3">
                    {result.hitCount} / {result.totalMatches}
                  </td>
                  <td className="px-4 py-3 font-bold text-cyan-300">
                    {result.hitRate}%
                  </td>
                  <td className="px-4 py-3">
                    <Link
                      href={`/backtest?round=${result.round.replace(/[^\d]/g, "")}`}
                      className="text-cyan-300 underline"
                    >
                      詳細
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>
    </main>
  );
}