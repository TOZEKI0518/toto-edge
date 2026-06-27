import type { RawTotoMatch } from "@/types/rawToto";
import type { TotoMatch } from "@/types/toto";

const toNumber = (value: string | number | undefined, fallback = 0) => {
  if (value === undefined || value === "") return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
};

const toId = (name: string) =>
  name
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "-")
    .replace(/[・]/g, "-");

export function normalizeMatch(raw: RawTotoMatch): TotoMatch {
  const matchNo = toNumber(raw.matchNo);

  return {
    id: `${raw.totoRound}-${matchNo}`,
    totoRound: raw.totoRound,
    matchNo,
    kickoffAt: raw.kickoffAt,
    venue: raw.venue,
    homeTeam: {
      id: toId(raw.homeShortName),
      name: raw.homeShortName,
      shortName: raw.homeShortName,
      league: "J1",
    },
    awayTeam: {
      id: toId(raw.awayShortName),
      name: raw.awayShortName,
      shortName: raw.awayShortName,
      league: "J1",
    },
    homeStats: {
      teamId: toId(raw.homeShortName),
      rank: toNumber(raw.homeRank),
      points: 0,
      wins: 0,
      draws: 0,
      losses: 0,
      goalsFor: 0,
      goalsAgainst: 0,
      goalDifference: toNumber(raw.homeGoalDiff),
      homeWinRate: toNumber(raw.homeWinRate),
      awayWinRate: 0,
      recentFormPoints: toNumber(raw.homeRecentPoints),
    },
    awayStats: {
      teamId: toId(raw.awayShortName),
      rank: toNumber(raw.awayRank),
      points: 0,
      wins: 0,
      draws: 0,
      losses: 0,
      goalsFor: 0,
      goalsAgainst: 0,
      goalDifference: toNumber(raw.awayGoalDiff),
      homeWinRate: 0,
      awayWinRate: toNumber(raw.awayWinRate),
      recentFormPoints: toNumber(raw.awayRecentPoints),
    },
    weather: {
      condition: raw.weather ?? "未取得",
      temperature: 25,
      windSpeed: 3,
    },
    voteRates:
      raw.voteHome !== undefined &&
      raw.voteDraw !== undefined &&
      raw.voteAway !== undefined
        ? {
            home: toNumber(raw.voteHome),
            draw: toNumber(raw.voteDraw),
            away: toNumber(raw.voteAway),
          }
        : undefined,
  };
}

export function normalizeMatches(rows: RawTotoMatch[]): TotoMatch[] {
  return rows.map(normalizeMatch);
}