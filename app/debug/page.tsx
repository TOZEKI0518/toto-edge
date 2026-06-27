import { getTotoLivePageSummary } from "@/services/totoLiveService";

export default async function DebugPage() {
  const sources = await Promise.all([
    getTotoLivePageSummary("rakutenSchedule"),
    getTotoLivePageSummary("yahooSchedule"),
    getTotoLivePageSummary("jleagueMatches"),
    getTotoLivePageSummary("jleagueStandings"),
  ]);

  return (
    <main className="min-h-screen bg-[#05060A] p-8 text-white">
      <div className="mx-auto max-w-5xl">
        <p className="text-sm text-cyan-300">Debug</p>
        <h1 className="mt-2 text-3xl font-bold">TOTO Source Fetch Status</h1>

        <div className="mt-6 grid gap-4 md:grid-cols-2">
          {sources.map((source) => (
            <article
              key={source.sourceKey}
              className="rounded-3xl border border-white/10 bg-white/[0.04] p-6"
            >
              <div className="mb-4 flex items-center justify-between gap-4">
                <h2 className="text-lg font-bold">{source.sourceKey}</h2>
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
                  <span className="text-white/40">Title:</span> {source.title}
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
      </div>
    </main>
  );
}