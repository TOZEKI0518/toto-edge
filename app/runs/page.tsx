import { AppNav } from "@/components/AppNav";
import Link from "next/link";
import { getPredictionRuns } from "@/services/predictionRunService";

export default async function RunsPage() {
  let runs: Awaited<ReturnType<typeof getPredictionRuns>> = [];
  let errorMessage = "";

  try {
    runs = await getPredictionRuns();
  } catch (error) {
    errorMessage =
      error instanceof Error
        ? error.message
        : "Prediction Runs の取得に失敗しました。";
  }

  return (
    <main className="min-h-screen bg-[#05060A] p-6 text-white">
      <div className="mx-auto max-w-5xl">
        <p className="text-sm text-cyan-300">AI Lab</p>
        <h1 className="mt-2 text-3xl font-bold">Prediction Runs</h1>

        <AppNav />

        {errorMessage && (
          <section className="mt-6 rounded-3xl border border-yellow-400/20 bg-yellow-400/10 p-5">
            <p className="text-yellow-200">{errorMessage}</p>
          </section>
        )}

        <section className="mt-6 overflow-hidden rounded-3xl border border-white/10">
          <table className="w-full border-collapse text-left text-sm">
            <thead className="bg-white/[0.06] text-white/60">
              <tr>
                <th className="px-4 py-3">Run</th>
                <th className="px-4 py-3">Round</th>
                <th className="px-4 py-3">Snapshot</th>
                <th className="px-4 py-3">Version</th>
                <th className="px-4 py-3">Hit</th>
                <th className="px-4 py-3">Rate</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.id} className="border-t border-white/10">
                  <td className="px-4 py-3">
                    <Link
                      href={`/runs/${run.id}`}
                      className="text-cyan-300 underline"
                    >
                      #{run.id}
                    </Link>
                  </td>
                  <td className="px-4 py-3">第{run.round_no}回</td>
                  <td className="px-4 py-3">
                    {run.data_snapshot_round
                      ? `第${run.data_snapshot_round}回`
                      : "-"}
                  </td>
                  <td className="px-4 py-3">{run.algorithm_version}</td>
                  <td className="px-4 py-3">
                    {run.hit_count ?? 0} / {run.total_matches ?? 0}
                  </td>
                  <td className="px-4 py-3 font-bold text-cyan-300">
                    {run.hit_rate ?? 0}%
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
