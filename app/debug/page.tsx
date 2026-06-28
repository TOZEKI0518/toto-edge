import { getTotoLivePageSummary } from "@/services/totoLiveService";
import {
  getLatestYahooTotoRound,
  getYahooTotoRounds,
} from "@/services/yahooTotoService";
import { getJleagueStandings } from "@/services/jleagueStandingService";
import { sampleFixtures } from "@/data/sampleFixtures";
import { buildPredictionInputs } from "@/services/predictionInputService";
import {
  getOutcomeLabel,
  predictFixtures,
} from "@/services/fixturePredictionService";


export default async function DebugPage() {
  const [sources, rounds, latestRound, standings] = await Promise.all([
    Promise.all([
      getTotoLivePageSummary("rakutenSchedule"),
      getTotoLivePageSummary("yahooSchedule"),
      getTotoLivePageSummary("jleagueMatches"),
      getTotoLivePageSummary("jleagueStandings"),
    ]),
    getYahooTotoRounds(),
    getLatestYahooTotoRound(),
    getJleagueStandings(),
  ]);

  const predictionInputs = buildPredictionInputs(sampleFixtures, standings);
  const fixturePredictions = predictFixtures(predictionInputs);
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
        <section className="mt-10">
          <div className="flex items-end justify-between">
            <div>
              <p className="text-sm text-cyan-300">J.LEAGUE</p>
              <h2 className="text-xl font-bold">Parsed Standings</h2>
            </div>
            <p className="text-sm text-white/50">{standings.length} teams</p>
          </div>

          <div className="mt-4 overflow-hidden rounded-3xl border border-white/10">
            <table className="w-full border-collapse text-left text-sm">
              <thead className="bg-white/[0.06] text-white/60">
                <tr>
                  <th className="px-4 py-3">Rank</th>
                  <th className="px-4 py-3">Team</th>
                  <th className="px-4 py-3">Points</th>
                  <th className="px-4 py-3">Goal Diff</th>
                </tr>
              </thead>
              <tbody>
                {standings.map((team) => (
                  <tr key={team.teamName} className="border-t border-white/10">
                    <td className="px-4 py-3 font-medium">{team.rank}</td>
                    <td className="px-4 py-3 text-white/80">{team.teamName}</td>
                    <td className="px-4 py-3 text-white/70">{team.points}</td>
                    <td className="px-4 py-3 text-white/70">
                      {team.goalDifference}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
        <section className="mt-10">
          <div className="flex items-end justify-between">
            <div>
              <p className="text-sm text-cyan-300">Prediction</p>
              <h2 className="text-xl font-bold">Prediction Inputs</h2>
            </div>
            <p className="text-sm text-white/50">
              {predictionInputs.length} matches
            </p>
          </div>

          <div className="mt-4 grid gap-4">
            {predictionInputs.map((input) => (
              <article
                key={input.fixture.matchNo}
                className="rounded-3xl border border-white/10 bg-white/[0.04] p-5"
              >
                <p className="text-sm text-white/50">
                  Match {input.fixture.matchNo}
                </p>
                <h3 className="mt-1 text-xl font-bold">
                  {input.fixture.homeTeam} vs {input.fixture.awayTeam}
                </h3>
                <p className="mt-3 text-sm text-white/70">
                  Home: {input.homeStanding?.rank ?? "-"}位 / 勝点{" "}
                  {input.homeStanding?.points ?? "-"} / 得失点{" "}
                  {input.homeStanding?.goalDifference ?? "-"}
                </p>
                <p className="mt-1 text-sm text-white/70">
                  Away: {input.awayStanding?.rank ?? "-"}位 / 勝点{" "}
                  {input.awayStanding?.points ?? "-"} / 得失点{" "}
                  {input.awayStanding?.goalDifference ?? "-"}
                </p>
              </article>
            ))}
          </div>
        </section>
        <section className="mt-10">
          <div className="flex items-end justify-between">
            <div>
              <p className="text-sm text-cyan-300">AI Engine V1</p>
              <h2 className="text-xl font-bold">Fixture Predictions</h2>
            </div>
            <p className="text-sm text-white/50">
              {fixturePredictions.length} predictions
            </p>
          </div>

          <div className="mt-4 grid gap-4">
            {fixturePredictions.map((prediction) => (
              <article
                key={prediction.matchNo}
                className="rounded-3xl border border-white/10 bg-white/[0.04] p-5"
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="text-sm text-white/50">
                      Match {prediction.matchNo}
                    </p>
                    <h3 className="mt-1 text-xl font-bold">
                      {prediction.homeTeam} vs {prediction.awayTeam}
                    </h3>
                    <p className="mt-2 text-cyan-300">
                      {getOutcomeLabel(prediction.outcome)}
                    </p>
                  </div>

                  <div className="text-right">
                    <p className="text-4xl font-black">{prediction.probability}%</p>
                    <p className="mt-1 text-sm text-white/50">
                      Confidence {prediction.confidence}
                    </p>
                  </div>
                </div>

                <div className="mt-5 rounded-2xl bg-black/30 p-4">
                  <p className="text-sm text-white/50">
                    Total Score: {prediction.totalScore}
                  </p>

                  <div className="mt-3 space-y-2">
                    {prediction.factors.map((factor) => (
                      <div
                        key={factor.label}
                        className="flex items-start justify-between gap-4 text-sm"
                      >
                        <div>
                          <p className="font-medium text-white/80">
                            {factor.label}
                          </p>
                          <p className="text-white/45">{factor.description}</p>
                        </div>
                        <p
                          className={
                            factor.score >= 0
                              ? "font-bold text-cyan-300"
                              : "font-bold text-red-300"
                          }
                        >
                          {factor.score >= 0 ? "+" : ""}
                          {factor.score}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </article>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}