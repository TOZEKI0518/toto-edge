import { PredictionBar } from "@/components/PredictionBar";

type MatchCardProps = {
  id: number;
  home: string;
  away: string;
  prediction: string;
  probability: number;
  edge: number;
  confidence: string;
};

export function MatchCard({
  id,
  home,
  away,
  prediction,
  probability,
  edge,
  confidence,
}: MatchCardProps) {
  return (
    <article className="rounded-3xl border border-white/10 bg-white/[0.04] p-5 transition hover:-translate-y-1 hover:bg-white/[0.07]">
      <div className="mb-4 flex items-center justify-between">
        <span className="rounded-full bg-white/10 px-3 py-1 text-xs text-white/60">
          Match {id}
        </span>
        <span className="rounded-full bg-cyan-400/10 px-3 py-1 text-xs text-cyan-300">
          Confidence {confidence}
        </span>
      </div>

      <h4 className="text-xl font-bold">
        {home} vs {away}
      </h4>

      <PredictionBar label={prediction} value={probability} />

      <div className="mt-5 rounded-2xl bg-black/30 p-4">
        <p className="text-sm text-white/50">Expected Value</p>
        <p className="mt-1 text-2xl font-bold text-cyan-300">+{edge}%</p>
      </div>
    </article>
  );
}