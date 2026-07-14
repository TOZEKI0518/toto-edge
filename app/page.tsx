import { AppHeader } from "@/components/AppHeader";
import { AppNav } from "@/components/AppNav";
import { getLatestCurrentRoundDashboard } from "@/services/currentRoundDashboardService";
import type {
  ConfidenceLevel,
  TotoMatchPrediction,
  TotoOutcome,
} from "@/types/currentRoundDashboard";

export const dynamic = "force-dynamic";

const OUTCOME_LABELS: Record<TotoOutcome, string> = {
  H: "ホーム勝ち",
  D: "引き分け",
  A: "アウェイ勝ち",
};

const CONFIDENCE_STYLES: Record<ConfidenceLevel, string> = {
  HIGH: "border-emerald-400/40 bg-emerald-400/10 text-emerald-200",
  MEDIUM: "border-amber-400/40 bg-amber-400/10 text-amber-200",
  LOW: "border-rose-400/40 bg-rose-400/10 text-rose-200",
};

function percentage(value: number | null | undefined, digits = 1) {
  if (value == null || !Number.isFinite(value)) return "-";
  return `${(value * 100).toFixed(digits)}%`;
}

function decimal(value: number | null | undefined, digits = 3) {
  if (value == null || !Number.isFinite(value)) return "-";
  return value.toFixed(digits);
}

function yen(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return "-";
  return `${Math.round(value).toLocaleString("ja-JP")}円`;
}

function outcomeProbability(match: TotoMatchPrediction, outcome: TotoOutcome) {
  if (outcome === "H") return match.prob_home;
  if (outcome === "D") return match.prob_draw;
  return match.prob_away;
}

function marketProbability(match: TotoMatchPrediction, outcome: TotoOutcome) {
  if (outcome === "H") return match.market_prob_home;
  if (outcome === "D") return match.market_prob_draw;
  return match.market_prob_away;
}

function ProbabilityBar({
  label,
  ai,
  market,
}: {
  label: string;
  ai: number;
  market: number;
}) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="text-white/65">{label}</span>
        <span>
          <span className="font-semibold text-cyan-200">{percentage(ai)}</span>
          <span className="ml-2 text-white/35">
            市場 {percentage(market)}
          </span>
        </span>
      </div>
      <div className="relative h-2 overflow-hidden rounded-full bg-white/10">
        <div
          className="absolute inset-y-0 left-0 rounded-full bg-cyan-400"
          style={{ width: `${Math.min(ai * 100, 100)}%` }}
        />
        <div
          className="absolute inset-y-0 border-r border-amber-300"
          style={{ width: `${Math.min(market * 100, 100)}%` }}
        />
      </div>
    </div>
  );
}

function MatchCard({ match }: { match: TotoMatchPrediction }) {
  const confidence = match.confidence_level ?? "LOW";
  const predictionProbability = outcomeProbability(match, match.prediction);
  const predictionMarket = marketProbability(match, match.prediction);

  return (
    <article className="rounded-3xl border border-white/10 bg-white/[0.035] p-5 shadow-xl">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs text-white/40">
            Match {match.toto_match_no}
            {match.match_date ? ` · ${match.match_date}` : ""}
          </p>
          <h3 className="mt-1 text-lg font-bold">
            {match.home_team}
            <span className="mx-2 text-white/35">vs</span>
            {match.away_team}
          </h3>
        </div>
        <span
          className={`rounded-full border px-3 py-1 text-xs font-semibold ${CONFIDENCE_STYLES[confidence]}`}
        >
          {confidence}
        </span>
      </div>

      <div className="mt-5 rounded-2xl border border-cyan-300/20 bg-cyan-300/[0.06] p-4">
        <div className="flex items-end justify-between gap-4">
          <div>
            <p className="text-xs text-cyan-100/60">AI Pick</p>
            <p className="mt-1 text-xl font-black text-cyan-100">
              {OUTCOME_LABELS[match.prediction]}
            </p>
          </div>
          <div className="text-right">
            <p className="text-3xl font-black">
              {percentage(predictionProbability)}
            </p>
            <p className="text-xs text-white/40">
              市場 {percentage(predictionMarket)}
            </p>
          </div>
        </div>
      </div>

      <div className="mt-5 space-y-3">
        <ProbabilityBar
          label="ホーム"
          ai={match.prob_home}
          market={match.market_prob_home}
        />
        <ProbabilityBar
          label="引き分け"
          ai={match.prob_draw}
          market={match.market_prob_draw}
        />
        <ProbabilityBar
          label="アウェイ"
          ai={match.prob_away}
          market={match.market_prob_away}
        />
      </div>

      <div className="mt-5 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
        <div className="rounded-2xl bg-white/[0.04] p-3">
          <p className="text-xs text-white/40">Best Edge</p>
          <p className="mt-1 font-bold text-emerald-300">
            {percentage(match.best_edge)}
          </p>
        </div>
        <div className="rounded-2xl bg-white/[0.04] p-3">
          <p className="text-xs text-white/40">Value Ratio</p>
          <p className="mt-1 font-bold">
            {decimal(match.best_value_ratio, 2)}
          </p>
        </div>
        <div className="rounded-2xl bg-white/[0.04] p-3">
          <p className="text-xs text-white/40">Confidence</p>
          <p className="mt-1 font-bold">
            {percentage(match.confidence_score)}
          </p>
        </div>
        <div className="rounded-2xl bg-white/[0.04] p-3">
          <p className="text-xs text-white/40">Coverage</p>
          <p className="mt-1 font-bold">
            {match.coverage_recommendation ?? "-"}
          </p>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-2 text-xs">
        <span className="rounded-full border border-white/10 px-3 py-1 text-white/60">
          RF {match.rf_prediction ?? "-"}
        </span>
        <span className="rounded-full border border-white/10 px-3 py-1 text-white/60">
          LGBM {match.lgbm_prediction ?? "-"}
        </span>
        <span className="rounded-full border border-white/10 px-3 py-1 text-white/60">
          推奨 {match.recommended_combination ?? "-"}
        </span>
        <span className="rounded-full border border-white/10 px-3 py-1 text-white/60">
          Value Pick {match.best_value_pick ?? "-"}
        </span>
      </div>
    </article>
  );
}

export default async function Home() {
  const { run, matches, tickets } =
    await getLatestCurrentRoundDashboard();

  if (!run || matches.length === 0) {
    return (
      <main className="min-h-screen bg-[#05060A] text-white">
        <div className="mx-auto max-w-7xl px-5 py-8">
          <AppHeader />
          <AppNav />
          <section className="mt-8 rounded-3xl border border-white/10 bg-white/[0.04] p-8">
            <p className="text-sm text-cyan-300">Project Alpha v6</p>
            <h1 className="mt-2 text-3xl font-black">
              Current round data is not available
            </h1>
            <p className="mt-3 text-white/55">
              Supabaseへ現在回の予測結果をアップロードすると、ここに表示されます。
            </p>
          </section>
        </div>
      </main>
    );
  }

  const bestValueMatch = [...matches].sort(
    (left, right) =>
      (right.roi_priority_score ?? 0) -
      (left.roi_priority_score ?? 0)
  )[0];

  return (
    <main className="min-h-screen bg-[#05060A] text-white">
      <div className="mx-auto max-w-7xl px-5 py-8">
        <AppHeader />
        <AppNav />

        <section className="mt-8 overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-cyan-500/20 via-blue-500/10 to-purple-500/20 p-6 shadow-2xl md:p-8">
          <div className="grid gap-8 lg:grid-cols-[1.4fr_1fr] lg:items-end">
            <div>
              <div className="flex flex-wrap items-center gap-3">
                <span className="rounded-full border border-cyan-300/30 bg-cyan-300/10 px-3 py-1 text-xs font-semibold text-cyan-100">
                  第{run.round_id}回
                </span>
                <span
                  className={`rounded-full border px-3 py-1 text-xs font-semibold ${
                    run.decision === "BUY"
                      ? "border-emerald-300/40 bg-emerald-300/10 text-emerald-100"
                      : "border-rose-300/40 bg-rose-300/10 text-rose-100"
                  }`}
                >
                  {run.decision}
                </span>
              </div>

              <p className="mt-6 text-sm text-white/50">
                Highest Value Match
              </p>
              <h1 className="mt-2 text-3xl font-black md:text-5xl">
                {bestValueMatch.home_team}
                <span className="mx-3 text-white/30">vs</span>
                {bestValueMatch.away_team}
              </h1>
              <p className="mt-4 text-lg text-cyan-100">
                {OUTCOME_LABELS[bestValueMatch.prediction]}
                <span className="ml-3 text-white/45">
                  AI {percentage(
                    outcomeProbability(
                      bestValueMatch,
                      bestValueMatch.prediction
                    )
                  )}
                </span>
              </p>
              <p className="mt-4 max-w-3xl text-sm leading-6 text-white/55">
                {run.reason}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                <p className="text-xs text-white/45">Investment</p>
                <p className="mt-2 text-2xl font-black">
                  {yen(run.investment_yen)}
                </p>
              </div>
              <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                <p className="text-xs text-white/45">Tickets</p>
                <p className="mt-2 text-2xl font-black">
                  {run.ticket_count}口
                </p>
              </div>
              <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                <p className="text-xs text-white/45">Value Index</p>
                <p className="mt-2 text-2xl font-black text-emerald-300">
                  {decimal(run.estimated_portfolio_value_index, 3)}
                </p>
              </div>
              <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                <p className="text-xs text-white/45">Best Edge</p>
                <p className="mt-2 text-2xl font-black text-cyan-200">
                  {percentage(bestValueMatch.best_edge)}
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
            <p className="text-xs text-white/45">Confidence</p>
            <p className="mt-2 text-lg font-bold">
              H {run.high_confidence_matches} / M{" "}
              {run.medium_confidence_matches} / L{" "}
              {run.low_confidence_matches}
            </p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
            <p className="text-xs text-white/45">Available Budget</p>
            <p className="mt-2 text-lg font-bold">
              {yen(run.available_budget_yen)}
            </p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
            <p className="text-xs text-white/45">Rollover</p>
            <p className="mt-2 text-lg font-bold">
              {yen(run.rollover_before_yen)} →{" "}
              {yen(run.rollover_after_yen)}
            </p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
            <p className="text-xs text-white/45">Generated Tickets</p>
            <p className="mt-2 text-lg font-bold">{tickets.length}口</p>
          </div>
        </section>

        <section className="mt-10">
          <div className="mb-5 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="text-sm text-cyan-300">Project Alpha v6</p>
              <h2 className="mt-1 text-2xl font-black">
                AI × Market Intelligence
              </h2>
            </div>
            <p className="text-sm text-white/45">
              RF 60% / LightGBM 40% Ensemble
            </p>
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            {matches.map((match) => (
              <MatchCard
                key={`${match.round_id}-${match.toto_match_no}`}
                match={match}
              />
            ))}
          </div>
        </section>

        {tickets.length > 0 && (
          <section className="mt-10 rounded-3xl border border-white/10 bg-white/[0.035] p-6">
            <div className="mb-5">
              <p className="text-sm text-cyan-300">Optimized Portfolio</p>
              <h2 className="mt-1 text-2xl font-black">
                上位10口プレビュー
              </h2>
            </div>

            <div className="overflow-x-auto">
              <table className="min-w-full text-left text-sm">
                <thead className="text-xs text-white/40">
                  <tr className="border-b border-white/10">
                    <th className="px-3 py-3">#</th>
                    <th className="px-3 py-3">Picks</th>
                    <th className="px-3 py-3">Value Index</th>
                    <th className="px-3 py-3">Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {tickets.slice(0, 10).map((ticket) => (
                    <tr
                      key={ticket.ticket_number}
                      className="border-b border-white/[0.06]"
                    >
                      <td className="px-3 py-3 text-white/45">
                        {ticket.ticket_number}
                      </td>
                      <td className="px-3 py-3 font-mono tracking-wider">
                        {ticket.picks}
                      </td>
                      <td className="px-3 py-3 text-emerald-300">
                        {decimal(ticket.conservative_value_index, 3)}
                      </td>
                      <td className="px-3 py-3">
                        {yen(ticket.ticket_cost_yen)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </div>
    </main>
  );
}
