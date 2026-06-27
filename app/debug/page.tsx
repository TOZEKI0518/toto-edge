import { getTotoLivePageSummary } from "@/services/totoLiveService";
import {
  getLatestYahooTotoRound,
  getYahooTotoRounds,
} from "@/services/yahooTotoService";

export default async function DebugPage() {
  const [sources, rounds, latestRound] = await Promise.all([
    Promise.all([
      getTotoLivePageSummary("rakutenSchedule"),
      getTotoLivePageSummary("yahooSchedule"),
      getTotoLivePageSummary("jleagueMatches"),
      getTotoLivePageSummary("jleagueStandings"),
    ]),
    getYahooTotoRounds(),
    getLatestYahooTotoRound(),
  ]);

  return (
    <main className="min-h-screen bg-[#05060A] p-8 text-white">
      <div className="mx-auto max-w-5xl">
        <p className="text-sm text-cyan-300">Debug</p>
        <h1 className="mt-2 text-3xl font-bold">TOTO Data Debug</h1>

        {latestRound && (
          <section className="mt-6 rounded-3xl border border-cyan-400/20 bg-cyan-400/10 p-6">
            <p className="text-sm text-cyan-300">Latest Round</p>
            <h2 className="mt-2 text-3xl font-bold">{latestRound.round}</h2>
            <p className="mt-2 text-white/70">
              {latestRound.salesStart} - {latestRound.salesEnd}
            </p>
            <p className="mt-3">
              <span className="rounded-full bg-white/10 px-3 py-1 text-xs text-white/70">
                {latestRound.status}
              </span>
            </p>
          </section>
        )}

        <section className="mt-6">
          <h2 className="text-xl font-bold">Source Fetch Status</h2>

          <div className="mt-4 grid gap-4 md:grid-cols-2">
            {sources.map((source) => (
              <article
                key={source.sourceKey}
                className="rounded-3xl border border-white/10 bg-white/[0.04] p-6"
              >
                <div className="mb-4 flex items-center justify-between gap-4">
                  <h3 className="text-lg font-bold">{source.sourceKey}</h3>
                  <span
                    className={
                      source.ok
                        ? "rounded-full bg-emerald-400/10 px-3 py-1 text-xs text-emerald-300"
                        : "rounded-full bg-red-400/10 px-3 py-1 text-xs text-red-300"
                    }
                  >
                    {source.ok ? "OK" : "FAILED"}
                  </span>
                </div>

                <div className="space-y-2 text-sm text-white/70">
                  <p>
                    <span className="text-white/40">Title:</span>{" "}
                    {source.title}
                  </p>
                  <p>
                    <span className="text-white/40">Length:</span>{" "}
                    {source.textLength.toLocaleString()}
                  </p>
                  <p>
                    <span className="text-white/40">Fetched:</span>{" "}
                    {source.fetchedAt}
                  </p>
                  <p className="break-all">
                    <span className="text-white/40">URL:</span> {source.url}
                  </p>

                  {!source.ok && (
                    <p className="text-red-300">{source.errorMessage}</p>
                  )}
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="mt-10">
          <div className="flex items-end justify-between">
            <div>
              <p className="text-sm text-cyan-300">Yahoo! toto</p>
              <h2 className="text-xl font-bold">Parsed Rounds</h2>
            </div>
            <p className="text-sm text-white/50">{rounds.length} rounds</p>
          </div>

          <div className="mt-4 overflow-hidden rounded-3xl border border-white/10">
            <table className="w-full border-collapse text-left text-sm">
              <thead className="bg-white/[0.06] text-white/60">
                <tr>
                  <th className="px-4 py-3">Round</th>
                  <th className="px-4 py-3">Sales Start</th>
                  <th className="px-4 py-3">Sales End</th>
                  <th className="px-4 py-3">Status</th>
                </tr>
              </thead>
              <tbody>
                {rounds.slice(0, 20).map((round) => (
                  <tr key={round.round} className="border-t border-white/10">
                    <td className="px-4 py-3 font-medium">{round.round}</td>
                    <td className="px-4 py-3 text-white/70">
                      {round.salesStart}
                    </td>
                    <td className="px-4 py-3 text-white/70">
                      {round.salesEnd}
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={
                          round.status === "open"
                            ? "rounded-full bg-emerald-400/10 px-3 py-1 text-xs text-emerald-300"
                            : "rounded-full bg-white/10 px-3 py-1 text-xs text-white/60"
                        }
                      >
                        {round.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </main>
  );
}