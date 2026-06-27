import type { Outcome } from "@/types/toto";
import { labelOutcome } from "@/lib/predict";

export function ProbabilityBar({ probabilities, voteRate }: { probabilities: Record<Outcome, number>; voteRate: Record<Outcome, number> }) {
  const rows: Outcome[] = ["home", "draw", "away"];
  return (
    <div className="space-y-3">
      {rows.map((o) => (
        <div key={o}>
          <div className="mb-1 flex items-center justify-between text-sm">
            <span className="text-slate-300">{labelOutcome(o)}</span>
            <span className="font-semibold">AI {probabilities[o]}% <span className="text-slate-500">/ 投票 {voteRate[o]}%</span></span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-slate-800">
            <div className="h-full rounded-full bg-emerald-400" style={{ width: `${probabilities[o]}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}
