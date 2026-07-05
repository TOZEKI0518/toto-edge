import Link from "next/link";
import { getPredictionHistoryByRunId } from "@/services/predictionRunService";

type RunDetailPageProps = {
  params: Promise<{
    runId: string;
  }>;
};

const labelOutcome = (value: string | null) => {
  if (value === "HOME") return "ホーム勝ち";
  if (value === "AWAY") return "アウェイ勝ち";
  if (value === "DRAW") return "引き分け";
  return "-";
};

export default async function RunDetailPage({ params }: RunDetailPageProps) {
  const { runId } = await params;
  const rows = await getPredictionHistoryByRunId(runId);

  return (
    <main className="min-h-screen bg-[#05060A] p-6 text-white">
      <div className="mx-auto max-w-5xl">
        <Link href="/runs" className="text-sm text-cyan-300">
          ← Prediction Runs
        </Link>

        <h1 className="mt-3 text-3xl font-bold">Run #{runId}</h1>

        <section className="mt-6 grid gap-4">
          {rows.map((row) => (
            <article
              key={row.id}
              className="rounded-3xl border border-white/10 bg-white/[0.04] p-5"
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-sm text-white/50">Match {row.match_no}</p>
                  <h2 className="mt-1 text-xl font-bold">
                    {row.home_team} vs {row.away_team}
                  </h2>
                  <p className="mt-2 text-white/60">
                    AI: {labelOutcome(row.predicted_outcome)} / Result:{" "}
                    {labelOutcome(row.actual_outcome)}
                  </p>
                </div>

                <div className="text-right">
                  <p
                    className={
                      row.hit
                        ? "text-3xl font-black text-emerald-300"
                        : "text-3xl font-black text-red-300"
                    }
                  >
                    {row.hit ? "✅" : "❌"}
                  </p>
                  <p className="mt-1 text-sm text-white/50">
                    {row.probability ?? 0}% / {row.confidence ?? "-"}
                  </p>
                </div>
              </div>
            </article>
          ))}

          {rows.length === 0 && (
            <section className="rounded-3xl border border-yellow-400/20 bg-yellow-400/10 p-5">
              <p className="text-yellow-200">
                このRunには保存された試合別予測がありません。
              </p>
            </section>
          )}
        </section>
      </div>
    </main>
  );
}