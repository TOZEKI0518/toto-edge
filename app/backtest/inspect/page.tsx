import Link from "next/link";
import { AppNav } from "@/components/AppNav";
import { inspectBacktestRounds } from "@/services/backtestService";

type BacktestInspectPageProps = {
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

export default async function BacktestInspectPage({
  searchParams,
}: BacktestInspectPageProps) {
  const params = await searchParams;
  const targetRounds = buildRoundRange(
    params?.from ?? "1600",
    params?.to ?? "1637"
  );

  const results = await inspectBacktestRounds(targetRounds);
  const candidates = results.filter((item) => item.isJleagueCandidate);

  return (
    <main className="min-h-screen bg-[#05060A] p-6 text-white">
      <div className="mx-auto max-w-5xl">
        <p className="text-sm text-cyan-300">Backtest</p>
        <h1 className="mt-2 text-3xl font-bold">J.League Round Inspector</h1>

        <AppNav />

        <section className="mt-6 rounded-3xl border border-cyan-400/20 bg-cyan-400/10 p-6">
          <p className="text-sm text-white/60">J.League Candidates</p>
          <p className="mt-2 text-6xl font-black">{candidates.length}</p>
          <p className="mt-2 text-white/70">
            Target rounds: {targetRounds[targetRounds.length - 1]} -{" "}
            {targetRounds[0]}
          </p>
        </section>

        <section className="mt-8 overflow-hidden rounded-3xl border border-white/10">
          <table className="w-full border-collapse text-left text-sm">
            <thead className="bg-white/[0.06] text-white/60">
              <tr>
                <th className="px-4 py-3">Round</th>
                <th className="px-4 py-3">Fixtures</th>
                <th className="px-4 py-3">J.League Matches</th>
                <th className="px-4 py-3">Hit</th>
                <th className="px-4 py-3">Rate</th>
                <th className="px-4 py-3">Coverage</th>
                <th className="px-4 py-3">Data</th>
                <th className="px-4 py-3">Detail</th>
              </tr>
            </thead>
            <tbody>
              {results.map((result) => (
                <tr
                  key={result.roundNumber}
                  className={
                    result.isJleagueCandidate
                      ? "border-t border-cyan-300/20 bg-cyan-300/[0.04]"
                      : "border-t border-white/10"
                  }
                >
                  <td className="px-4 py-3">{result.round}</td>
                  <td className="px-4 py-3">{result.totalFixtures}</td>
                  <td className="px-4 py-3">{result.totalMatches}</td>
                  <td className="px-4 py-3">
                    {result.hitCount} / {result.totalMatches}
                  </td>
                  <td className="px-4 py-3 font-bold text-cyan-300">
                    {result.hitRate}%
                  </td>
                  <td className="px-4 py-3">{result.coverageRate}%</td>
                  <td className="px-4 py-3 text-white/60">
                    {result.dataSource === "date_cut" && result.snapshotDate
                      ? `${result.snapshotDate}以前`
                      : result.dataSource === "round_snapshot" && result.snapshotRoundNo
                      ? `第${result.snapshotRoundNo}回`
                      : result.dataSource === "snapshot" && result.snapshotDate
                      ? result.snapshotDate
                      : result.dataSource === "current"
                      ? "現在"
                      : "なし"}
                  </td>
                  <td className="px-4 py-3">
                    <Link
                      href={`/backtest?round=${result.roundNumber}`}
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
