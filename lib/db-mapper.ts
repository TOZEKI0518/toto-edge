import type { TotoMatch } from "@/types/toto";

type DbMatch = {
  id: number;
  round_no: string;
  match_no: number;
  kickoff: string | null;
  league: string | null;
  venue: string | null;
  weather: string | null;
  home_team: string;
  away_team: string;
  home_rank: number | null;
  away_rank: number | null;
  home_points: number | null;
  away_points: number | null;
  home_last5: string | null;
  away_last5: string | null;
  home_goals_for: number | null;
  home_goals_against: number | null;
  away_goals_for: number | null;
  away_goals_against: number | null;
  home_win_rate: number | null;
  away_win_rate: number | null;
  home_rest_days: number | null;
  away_rest_days: number | null;
  home_injury_level: number | null;
  away_injury_level: number | null;
  vote_home: number | null;
  vote_draw: number | null;
  vote_away: number | null;
};

function weatherLabel(value: string | null): TotoMatch["weather"] {
  if (value === "晴" || value === "曇" || value === "雨" || value === "強雨") return value;
  return "曇";
}

export function mapDbMatch(row: DbMatch): TotoMatch {
  return {
    id: row.match_no,
    round: row.round_no,
    kickoff: row.kickoff?.replace("T", " ").slice(0, 16) ?? "未定",
    league: row.league ?? "J",
    venue: row.venue ?? "未定",
    weather: weatherLabel(row.weather),
    home: {
      name: row.home_team,
      rank: row.home_rank ?? 10,
      points: row.home_points ?? 0,
      last5: row.home_last5 ?? "DDDDD",
      goalsFor: row.home_goals_for ?? 0,
      goalsAgainst: row.home_goals_against ?? 0,
      homeWinRate: Number(row.home_win_rate ?? 0.4),
      restDays: row.home_rest_days ?? 5,
      injuryLevel: row.home_injury_level ?? 0,
    },
    away: {
      name: row.away_team,
      rank: row.away_rank ?? 10,
      points: row.away_points ?? 0,
      last5: row.away_last5 ?? "DDDDD",
      goalsFor: row.away_goals_for ?? 0,
      goalsAgainst: row.away_goals_against ?? 0,
      awayWinRate: Number(row.away_win_rate ?? 0.3),
      restDays: row.away_rest_days ?? 5,
      injuryLevel: row.away_injury_level ?? 0,
    },
    voteRate: {
      home: Number(row.vote_home ?? 33),
      draw: Number(row.vote_draw ?? 34),
      away: Number(row.vote_away ?? 33),
    },
  };
}
