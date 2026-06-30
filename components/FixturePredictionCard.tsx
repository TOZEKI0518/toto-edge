import { getOutcomeLabel } from "@/services/fixturePredictionService";
import type { FixturePrediction } from "@/types/prediction";

type FixturePredictionCardProps = {
  prediction: FixturePrediction;
};

const getOutcomeIcon = (outcome: FixturePrediction["outcome"]) => {
  if (outcome === "HOME") return "🏠";
  if (outcome === "AWAY") return "🚗";
  return "🤝";
};

export function FixturePredictionCard({
  prediction,
}: FixturePredictionCardProps) {
  return (
    <article className="rounded-3xl border border-white/10 bg-white/[0.04] p-5 shadow-xl">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs text-white/40">Match {prediction.matchNo}</p>
          <h3 className="mt-1 text-lg font-bold leading-snug md:text-xl">
            {prediction.homeTeam}
            <span className="mx-2 text-white/30">vs</span>
            {prediction.awayTeam}
          </h3>
        </div>

        <span className="rounded-full bg-cyan-400/10 px-3 py-1 text-xs font-medium text-cyan-300">
          {prediction.confidence}
        </span>
      </div>

      <div className="mt-5 rounded-2xl bg-black/30 p-4">
        <div className="flex items-center justify-between gap-4">
          <div>
            <p className="text-sm text-white/50">AI Prediction</p>
            <p className="mt-1 text-2xl font-bold text-cyan-300">
              {getOutcomeIcon(prediction.outcome)}{" "}
              {getOutcomeLabel(prediction.outcome)}
            </p>
          </div>

          <div className="text-right">
            <p className="text-4xl font-black">{prediction.probability}%</p>
            <p className="text-xs text-white/40">
              AI Score {prediction.totalScore}
            </p>
          </div>
        </div>
      </div>

      <details className="mt-4 rounded-2xl bg-white/[0.03] p-4">
        <summary className="cursor-pointer text-sm font-medium text-white/70">
          予想理由を見る
        </summary>

        <div className="mt-4 space-y-3">
          {prediction.factors.map((factor) => (
            <div
              key={factor.label}
              className="flex items-start justify-between gap-4 text-sm"
            >
              <div>
                <p className="font-medium text-white/80">{factor.label}</p>
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
      </details>
    </article>
  );
}