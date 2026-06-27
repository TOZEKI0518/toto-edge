import { EdgeRanking } from "@/components/EdgeRanking";
import { MatchCard } from "@/components/MatchCard";
import { sampleMatches } from "@/data/sampleMatches";
import { mapDbMatch } from "@/lib/db-mapper";
import { supabaseSelect } from "@/lib/supabase-rest";
import type { TotoMatch } from "@/types/toto";

async function loadMatches(): Promise<{ matches: TotoMatch[]; mode: string; round: string }> {
  try {
    const rows = await supabaseSelect<any>(
      "matches",
      "select=*&order=round_no.desc,match_no.asc&limit=13"
    );
    if (rows.length > 0) {
      const matches = rows.map(mapDbMatch);
      return { matches, mode: "Supabase連携", round: matches[0].round };
    }
  } catch (error) {
    console.error(error);
  }
  return { matches: sampleMatches, mode: "サンプルデータ", round: sampleMatches[0].round };
}

export default async function Home() {
  const { matches, mode, round } = await loadMatches();

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top_left,#064e3b_0,#020617_34%,#020617_100%)] px-4 py-8 md:px-8">
      <div className="mx-auto max-w-6xl">
        <header className="mb-8 flex flex-col justify-between gap-4 md:flex-row md:items-end">
          <div>
            <p className="text-sm font-semibold tracking-[0.35em] text-emerald-300">TOTO EDGE</p>
            <h1 className="mt-3 text-4xl font-black text-white md:text-6xl">期待値で選ぶ<br className="md:hidden" />toto分析MVP</h1>
            <p className="mt-4 max-w-2xl text-slate-300">勝敗を単純に当てるのではなく、AI予想確率と投票率のズレから「狙い目」を見つけます。Supabase未設定時はサンプルデータで表示します。</p>
          </div>
          <div className="rounded-3xl border border-slate-800 bg-slate-900/70 p-4 text-sm text-slate-300">
            <p>対象回：<span className="font-bold text-white">{round}</span></p>
            <p>試合数：<span className="font-bold text-white">{matches.length}試合</span></p>
            <p>モード：<span className="font-bold text-emerald-300">{mode}</span></p>
          </div>
        </header>

        <div className="mb-6">
          <EdgeRanking matches={matches} />
        </div>

        <section className="grid gap-5 lg:grid-cols-2">
          {matches.map((match) => <MatchCard key={`${match.round}-${match.id}`} match={match} />)}
        </section>
      </div>
    </main>
  );
}
