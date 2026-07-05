import { getJleagueStandings } from "@/services/jleagueStandingService";

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

type StandingHistoryRow = {
  round_no: number;
  team_name: string;
  rank?: number;
  points?: number;
  goal_difference?: number;
  matches?: number;
  wins?: number;
  draws?: number;
  losses?: number;
  goals_for?: number;
  goals_against?: number;
};

export async function saveCurrentStandingsSnapshot(roundNo: number) {
  if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
    throw new Error("Supabase environment variables are not set.");
  }

  const standings = await getJleagueStandings();

  const rows: StandingHistoryRow[] = standings.map((team) => ({
    round_no: roundNo,
    team_name: team.teamName,
    rank: team.rank,
    points: team.points,
    goal_difference: team.goalDifference,
    matches: team.matches,
    wins: team.wins,
    draws: team.draws,
    losses: team.losses,
    goals_for: team.goalsFor,
    goals_against: team.goalsAgainst,
  }));

  const res = await fetch(`${SUPABASE_URL}/rest/v1/standings_history`, {
    method: "POST",
    headers: {
      apikey: SUPABASE_SERVICE_ROLE_KEY,
      Authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
      "Content-Type": "application/json",
      Prefer: "return=minimal",
    },
    body: JSON.stringify(rows),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to save standings snapshot: ${res.status} ${text}`);
  }

  return {
    roundNo,
    savedCount: rows.length,
  };
}