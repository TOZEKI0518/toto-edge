type Row = Record<string, string>;
type JsonRecord = Record<string, unknown>;

export const dynamic = "force-dynamic";

async function getDashboard() {
  const baseUrl =
    process.env.VERCEL_URL
      ? `https://${process.env.VERCEL_URL}`
      : process.env.NEXT_PUBLIC_BASE_URL ?? "http://localhost:3000";

  const response = await fetch(`${baseUrl}/api/toto-quant`, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`Dashboard API failed: ${response.status}`);
  }

  return response.json() as Promise<{
    generatedAt: string;
    predictions: Row[];
    teamFeatures: Row[];
    probability: JsonRecord | null;
    expectedValue: JsonRecord | null;
    monteCarlo: JsonRecord | null;
  }>;
}

function numberValue(
  source: JsonRecord | null,
  candidates: string[],
): number | null {
  if (!source) return null;

  for (const key of candidates) {
    const value = source[key];
    if (typeof value === "number") return value;
    if (typeof value === "string" && value.trim() !== "") {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) return parsed;
    }
  }

  return null;
}

function percent(value: number | null, digits = 3) {
  return value === null ? "—" : `${value.toFixed(digits)}%`;
}

function predictionLabel(value: string) {
  if (value === "H") return "ホーム勝ち";
  if (value === "D") return "引き分け";
  if (value === "A") return "アウェイ勝ち";
  return value || "—";
}

export default async function TotoQuantPage() {
  const data = await getDashboard();

  const firstPrize = numberValue(data.probability, [
    "probability_13",
    "first_probability",
    "all_correct_probability",
    "p13",
  ]);
  const secondOrBetter = numberValue(data.probability, [
    "probability_12_or_better",
    "second_or_better",
    "p12_or_better",
  ]);
  const thirdOrBetter = numberValue(data.probability, [
    "probability_11_or_better",
    "third_or_better",
    "p11_or_better",
  ]);
  const expectedRoi = numberValue(data.expectedValue, [
    "expected_roi",
    "roi",
    "expected_roi_percent",
  ]);
  const meanHits = numberValue(data.monteCarlo, [
    "mean_hits",
    "meanHits",
  ]);

  return (
    <main className="min-h-screen bg-[#05060a] text-white">
      <div className="mx-auto max-w-5xl px-4 py-6 sm:px-6">
        <header className="mb-6">
          <p className="text-xs font-semibold uppercase tracking-[0.25em] text-cyan-300">
            Project Alpha
          </p>
          <h1 className="mt-2 text-3xl font-black sm:text-4xl">
            Toto Quant
          </h1>
          <p className="mt-2 text-sm text-white/50">
            AI予測・当選確率・Player Intelligence
          </p>
        </header>

        <section className="grid grid-cols-2 gap-3 sm:grid-cols-5">
          <Metric label="1等" value={percent(firstPrize, 4)} />
          <Metric label="2等以上" value={percent(secondOrBetter, 3)} />
          <Metric label="3等以上" value={percent(thirdOrBetter, 3)} />
          <Metric
            label="期待ROI"
            value={expectedRoi === null ? "—" : `${expectedRoi.toFixed(1)}%`}
          />
          <Metric
            label="平均的中"
            value={meanHits === null ? "—" : `${meanHits.toFixed(2)}/13`}
          />
        </section>

        <section className="mt-8">
          <div className="mb-3 flex items-end justify-between gap-3">
            <div>
              <h2 className="text-xl font-bold">最新予測</h2>
              <p className="text-xs text-white/40">
                {data.predictions.length}試合
              </p>
            </div>
            <p className="text-[11px] text-white/35">
              {new Date(data.generatedAt).toLocaleString("ja-JP")}
            </p>
          </div>

          {data.predictions.length === 0 ? (
            <Empty text="ml/round_predictions.csv がまだありません。" />
          ) : (
            <div className="grid gap-3 md:grid-cols-2">
              {data.predictions.map((row, index) => {
                const home = row.homeTeam || row.home_team || "Home";
                const away = row.awayTeam || row.away_team || "Away";
                const prediction = row.prediction || "";
                const confidence = Number(row.confidence || 0);
                const difficulty = row.difficultyClass || row.difficulty || "";

                return (
                  <article
                    key={`${home}-${away}-${index}`}
                    className="rounded-2xl border border-white/10 bg-white/[0.045] p-4"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <p className="text-sm font-semibold">
                          {index + 1}. {home}
                        </p>
                        <p className="mt-1 text-sm text-white/55">
                          vs {away}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-lg font-black text-cyan-300">
                          {predictionLabel(prediction)}
                        </p>
                        <p className="text-xs text-white/45">
                          {(confidence * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>

                    <div className="mt-4 grid grid-cols-3 gap-2 text-center">
                      <Probability label="H" value={row.probH} />
                      <Probability label="D" value={row.probD} />
                      <Probability label="A" value={row.probA} />
                    </div>

                    {(difficulty || row.difficultyStars) && (
                      <p className="mt-3 text-xs text-white/45">
                        難易度: {row.difficultyStars || ""} {difficulty}
                      </p>
                    )}
                  </article>
                );
              })}
            </div>
          )}
        </section>

        <section className="mt-8">
          <h2 className="mb-3 text-xl font-bold">Player Intelligence</h2>
          {data.teamFeatures.length === 0 ? (
            <Empty text="team_features.csv がまだありません。" />
          ) : (
            <div className="overflow-hidden rounded-2xl border border-white/10">
              <div className="overflow-x-auto">
                <table className="min-w-full text-left text-sm">
                  <thead className="bg-white/[0.06] text-white/55">
                    <tr>
                      <th className="px-4 py-3">Team</th>
                      <th className="px-4 py-3">Core</th>
                      <th className="px-4 py-3">Momentum</th>
                      <th className="px-4 py-3">Suspension</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.teamFeatures.map((row, index) => (
                      <tr
                        key={`${row.team}-${index}`}
                        className="border-t border-white/10"
                      >
                        <td className="whitespace-nowrap px-4 py-3 font-semibold">
                          {row.team}
                        </td>
                        <td className="px-4 py-3">
                          {Number(row.homeCoreScore || 0).toFixed(3)}
                        </td>
                        <td className="px-4 py-3">
                          {Number(row.homeMomentum || 0).toFixed(3)}
                        </td>
                        <td className="px-4 py-3">
                          {Number(row.homeSuspensionLoss || 0).toFixed(3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-gradient-to-br from-cyan-500/15 to-blue-500/5 p-4">
      <p className="text-xs text-white/45">{label}</p>
      <p className="mt-2 text-xl font-black">{value}</p>
    </div>
  );
}

function Probability({ label, value }: { label: string; value?: string }) {
  const numeric = Number(value || 0);
  return (
    <div className="rounded-xl bg-black/25 px-2 py-2">
      <p className="text-[11px] text-white/40">{label}</p>
      <p className="mt-1 font-bold">{(numeric * 100).toFixed(1)}%</p>
    </div>
  );
}

function Empty({ text }: { text: string }) {
  return (
    <div className="rounded-2xl border border-dashed border-white/15 p-6 text-sm text-white/45">
      {text}
    </div>
  );
}
