import type { TotoMatch } from "@/types/toto";
import { labelOutcome, predictMatch, sign } from "@/lib/predict";
import { ProbabilityBar } from "./ProbabilityBar";

export function MatchCard({ match }: { match: TotoMatch }) {
  const p = predictMatch(match);
  const topReasons = [...p.reasons].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 4);
  return (
    <article className="rounded-3xl border border-slate-800 bg-slate-900/70 p-5 shadow-glow backdrop-blur">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <p className="text-xs text-slate-400">No.{match.id} / {match.league} / {match.kickoff}</p>
          <h2 className="mt-1 text-xl font-bold text-white">{match.home.name} <span className="text-slate-500">vs</span> {match.away.name}</h2>
          <p className="mt-1 text-sm text-slate-400">{match.venue}・{match.weather}</p>
        </div>
        <div className="rounded-2xl bg-emerald-400 px-3 py-2 text-center text-slate-950">
          <p className="text-xs font-bold">EDGE</p>
          <p className="text-lg font-black">{sign(p.edge)}%</p>
        </div>
      </div>

      <ProbabilityBar probabilities={p.probabilities} voteRate={match.voteRate} />

      <div className="mt-5 grid gap-3 rounded-2xl bg-slate-950/70 p-4 md:grid-cols-3">
        <div>
          <p className="text-xs text-slate-500">おすすめ</p>
          <p className="text-lg font-bold text-emerald-300">{labelOutcome(p.pick)}</p>
        </div>
        <div>
          <p className="text-xs text-slate-500">信頼度</p>
          <p className="text-lg font-bold">{p.confidence}</p>
        </div>
        <div>
          <p className="text-xs text-slate-500">AI確率</p>
          <p className="text-lg font-bold">{p.probabilities[p.pick]}%</p>
        </div>
      </div>

      <div className="mt-4">
        <p className="mb-2 text-sm font-semibold text-slate-300">主な理由</p>
        <div className="space-y-2">
          {topReasons.map((r) => (
            <div key={r.label} className="flex items-center justify-between rounded-xl bg-slate-800/70 px-3 py-2 text-sm">
              <span>{r.label}<span className="ml-2 text-slate-500">{r.note}</span></span>
              <span className={r.value >= 0 ? "font-bold text-emerald-300" : "font-bold text-rose-300"}>{sign(r.value)}</span>
            </div>
          ))}
        </div>
      </div>
    </article>
  );
}
