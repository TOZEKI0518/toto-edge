import { supabaseSelect } from "@/lib/supabase-rest";
import type {
  CurrentRoundDashboardData,
  TotoMatchPrediction,
  TotoRoundRun,
  TotoRoundTicket,
} from "@/types/currentRoundDashboard";

export async function getLatestCurrentRoundDashboard(): Promise<CurrentRoundDashboardData> {
  const latestRuns = await supabaseSelect<TotoRoundRun>(
    "toto_round_runs",
    "select=*&order=round_id.desc&limit=1"
  );

  const run = latestRuns[0] ?? null;

  if (!run) {
    return { run: null, matches: [], tickets: [] };
  }

  const roundId = encodeURIComponent(String(run.round_id));

  const [matches, tickets] = await Promise.all([
    supabaseSelect<TotoMatchPrediction>(
      "toto_round_match_predictions",
      `select=*&round_id=eq.${roundId}&order=toto_match_no.asc`
    ),
    supabaseSelect<TotoRoundTicket>(
      "toto_round_tickets",
      `select=*&round_id=eq.${roundId}&order=ticket_number.asc`
    ),
  ]);

  return { run, matches, tickets };
}
