import { getJleagueStandings } from "@/services/jleagueStandingService";
import { supabaseSelect } from "@/lib/supabase-rest";
import type { TeamStanding } from "@/types/standing";

const SUPABASE_URL = process.env.SUPABASE_URL ?? process.env.NEXT_PUBLIC_SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

type StandingHistoryRow = {
  round_no: number | null;
  snapshot_date: string;
  league: string | null;
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

type TotoRoundDateRow = {
  round_no: number;
  round_date: string;
};

function getTodayJstDate() {
  const now = new Date();
  const jst = new Date(now.getTime() + 9 * 60 * 60 * 1000);
  return jst.toISOString().slice(0, 10);
}

function assertSupabaseWriteConfig() {
  if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
    throw new Error("Supabase write environment variables are not set. SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required.");
  }

  return {
    url: SUPABASE_URL,
    key: SUPABASE_SERVICE_ROLE_KEY,
  };
}

function toRows(
  standings: TeamStanding[],
  roundNo: number | null,
  snapshotDate: string
): StandingHistoryRow[] {
  return standings.map((team) => {
    const withLeague = team as typeof team & { league?: string };

    return {
      round_no: roundNo,
      snapshot_date: snapshotDate,
      league: withLeague.league ?? null,
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
    };
  });
}

async function deleteExistingRoundSnapshot(roundNo: number) {
  const config = assertSupabaseWriteConfig();

  const res = await fetch(
    `${config.url}/rest/v1/standings_history?round_no=eq.${roundNo}`,
    {
      method: "DELETE",
      headers: {
        apikey: config.key,
        Authorization: `Bearer ${config.key}`,
        Prefer: "return=minimal",
      },
      cache: "no-store",
    }
  );

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to delete old standings snapshot: ${res.status} ${text}`);
  }
}

async function insertStandingRows(rows: StandingHistoryRow[]) {
  const config = assertSupabaseWriteConfig();

  const res = await fetch(`${config.url}/rest/v1/standings_history`, {
    method: "POST",
    headers: {
      apikey: config.key,
      Authorization: `Bearer ${config.key}`,
      "Content-Type": "application/json",
      Prefer: "return=minimal",
    },
    body: JSON.stringify(rows),
    cache: "no-store",
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to save standings snapshot: ${res.status} ${text}`);
  }
}

export async function getRoundDate(roundNo: number): Promise<string | null> {
  const rows = await supabaseSelect<TotoRoundDateRow>(
    "toto_round_dates",
    `select=round_no,round_date&round_no=eq.${roundNo}&limit=1`
  );

  return rows[0]?.round_date ?? null;
}

export async function listRoundDates(limit = 300): Promise<TotoRoundDateRow[]> {
  return supabaseSelect<TotoRoundDateRow>(
    "toto_round_dates",
    `select=round_no,round_date&order=round_no.desc&limit=${limit}`
  );
}

export async function saveStandingsSnapshot(
  standings: TeamStanding[],
  roundNo: number | null,
  snapshotDate: string
) {
  if (roundNo !== null) {
    await deleteExistingRoundSnapshot(roundNo);
  }

  const rows = toRows(standings, roundNo, snapshotDate);
  await insertStandingRows(rows);

  return {
    roundNo,
    snapshotDate,
    savedCount: rows.length,
  };
}

export async function saveCurrentStandingsSnapshot(
  roundNo?: number,
  snapshotDate?: string
) {
  const resolvedSnapshotDate =
    snapshotDate ??
    (typeof roundNo === "number" ? await getRoundDate(roundNo) : null) ??
    getTodayJstDate();

  const standings = await getJleagueStandings();

  return saveStandingsSnapshot(
    standings,
    typeof roundNo === "number" ? roundNo : null,
    resolvedSnapshotDate
  );
}
