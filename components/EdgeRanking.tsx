import type { TotoMatch } from "@/types/toto";
import { labelOutcome, predictMatch, sign } from "@/lib/predict";

export function EdgeRanking({ matches }: { matches: TotoMatch[] }) {
  const rows = matches.map((m) => ({ match: m, prediction: predictMatch(m) })).sort((a, b) => b.prediction.edge - a.prediction.edge).slice(0, 5);
  return (
    <section className="rounded-3xl border border-emerald-500/20 bg-emerald-950/20 p-5">
      <h2 className="text-lg font-bold text-white">期待値ランキング</h2>
      <p className="mt-1 text-sm text-slate-400">AI確率とtoto投票率の差が大きい順です。</p>
      <div className="mt-4 space-y-3">
        {rows.map(({ match, prediction }, idx) => (
          <div key={match.id} className="flex items-center justify-between rounded-2xl bg-slate-950/70 p-3">
            <div>
              <p className="text-sm font-bold">{idx + 1}. {match.home.name} vs {match.away.name}</p>
              <p className="text-xs text-slate-400">{labelOutcome(prediction.pick)} / 信頼度 {prediction.confidence}</p>
            </div>
            <div className="text-right">
              <p className="text-lg font-black text-emerald-300">{sign(prediction.edge)}%</p>
              <p className="text-xs text-slate-500">EDGE</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
