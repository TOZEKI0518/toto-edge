import { getOutcomeLabel } from "@/services/fixturePredictionService";
import type { FixturePrediction } from "@/types/prediction";

type FixturePredictionCardProps = {
  prediction: FixturePrediction;
};

export function FixturePredictionCard({
  prediction,
}: FixturePredictionCardProps) {
  return (
    <article className="rounded-3xl border border-white/10 bg-white/[0.04] p-5 shadow-xl">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm text-white/50">Match {prediction.matchNo}</p>
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
          AI Score: {prediction.totalScore}
        </p>

        <div className="mt-3 space-y-2">
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
      </div>
    </article>
  );
}