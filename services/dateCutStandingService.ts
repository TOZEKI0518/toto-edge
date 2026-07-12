import { normalizeTeamName } from "@/lib/teamNameNormalizer";
import { supabaseSelect } from "@/lib/supabase-rest";
import { loadManyRoundFixtures } from "@/services/totoFixtureCacheService";
import type { HeadToHeadStats, TeamStanding } from "@/types/standing";
import { isJLeagueFixture } from "@/services/leagueClassifier";

type TotoRoundDateRow = {
  round_no: number;
  round_date: string;
};

type TeamStats = {
  teamName: string;
  matches: number;
  wins: number;
  draws: number;
  losses: number;
  points: number;
  goalsFor: number;
  goalsAgainst: number;
  goalDifference: number;

  homeMatches: number;
  homeWins: number;
  homeDraws: number;
  homeLosses: number;
  awayMatches: number;
  awayWins: number;
  awayDraws: number;
  awayLosses: number;

  homeGoalsFor: number;
  homeGoalsAgainst: number;
  awayGoalsFor: number;
  awayGoalsAgainst: number;

  eloRating: number;
  homeEloRating: number;
  awayEloRating: number;

  recentResults: Array<"W" | "D" | "L">;
  homeRecentResults: Array<"W" | "D" | "L">;
  awayRecentResults: Array<"W" | "D" | "L">;

  headToHead: Map<string, HeadToHeadStats>;
};

const INITIAL_ELO = 1500;
const ELO_K_FACTOR = 24;
const HOME_ELO_BONUS = 25;
const SIDE_ELO_K_FACTOR = 18;
const DEFAULT_PRIOR_ROUND_LIMIT = 60;
const LEAGUE_AVG_GOALS = 1.35;

const standingsCache = new Map<
  number,
  Awaited<ReturnType<typeof buildDateCutStandingsInternal>>
>();

function expectedScore(ratingA: number, ratingB: number) {
  return 1 / (1 + Math.pow(10, (ratingB - ratingA) / 400));
}

function actualScores(totoResult: string) {
  if (totoResult === "1") return { home: 1, away: 0 };
  if (totoResult === "2") return { home: 0, away: 1 };
  return { home: 0.5, away: 0.5 };
}

function round1(value: number) {
  return Math.round(value * 10) / 10;
}

function safeRate(numerator: number, denominator: number) {
  if (!denominator || denominator <= 0) return 0;
  return numerator / denominator;
}

function applyEloResult(home: TeamStats, away: TeamStats, totoResult: string) {
  const expectedHome = expectedScore(
    home.eloRating + HOME_ELO_BONUS,
    away.eloRating
  );
  const expectedAway = 1 - expectedHome;
  const actual = actualScores(totoResult);

  home.eloRating = round1(
    home.eloRating + ELO_K_FACTOR * (actual.home - expectedHome)
  );
  away.eloRating = round1(
    away.eloRating + ELO_K_FACTOR * (actual.away - expectedAway)
  );

  const expectedHomeSide = expectedScore(
    home.homeEloRating,
    away.awayEloRating
  );
  const expectedAwaySide = 1 - expectedHomeSide;

  home.homeEloRating = round1(
    home.homeEloRating +
      SIDE_ELO_K_FACTOR * (actual.home - expectedHomeSide)
  );
  away.awayEloRating = round1(
    away.awayEloRating +
      SIDE_ELO_K_FACTOR * (actual.away - expectedAwaySide)
  );
}

function uniqByRound(rows: TotoRoundDateRow[]) {
  const map = new Map<number, TotoRoundDateRow>();

  for (const row of rows) {
    if (Number.isInteger(row.round_no) && row.round_date) {
      map.set(row.round_no, row);
    }
  }

  return Array.from(map.values());
}

export async function getRoundDateFromMaster(
  roundNo: number
): Promise<string | null> {
  const rows = await supabaseSelect<TotoRoundDateRow>(
    "toto_round_dates",
    `select=round_no,round_date&round_no=eq.${roundNo}&limit=1`
  );

  return rows[0]?.round_date ?? null;
}

async function listPriorRoundDates(
  targetDate: string,
  targetRoundNo: number,
  limit = DEFAULT_PRIOR_ROUND_LIMIT
) {
  const rows = await supabaseSelect<TotoRoundDateRow>(
    "toto_round_dates",
    `select=round_no,round_date&round_date=lt.${targetDate}&round_no=lt.${targetRoundNo}&order=round_date.desc&limit=${limit}`
  );

  return uniqByRound(rows).sort((a, b) => a.round_no - b.round_no);
}

function ensureStats(statsMap: Map<string, TeamStats>, rawTeamName: string) {
  const teamName = normalizeTeamName(rawTeamName);
  const existing = statsMap.get(teamName);

  if (existing) return existing;

  const stats: TeamStats = {
    teamName,
    matches: 0,
    wins: 0,
    draws: 0,
    losses: 0,
    points: 0,
    goalsFor: 0,
    goalsAgainst: 0,
    goalDifference: 0,

    homeMatches: 0,
    homeWins: 0,
    homeDraws: 0,
    homeLosses: 0,
    awayMatches: 0,
    awayWins: 0,
    awayDraws: 0,
    awayLosses: 0,

    homeGoalsFor: 0,
    homeGoalsAgainst: 0,
    awayGoalsFor: 0,
    awayGoalsAgainst: 0,

    eloRating: INITIAL_ELO,
    homeEloRating: INITIAL_ELO,
    awayEloRating: INITIAL_ELO,

    recentResults: [],
    homeRecentResults: [],
    awayRecentResults: [],

    headToHead: new Map(),
  };

  statsMap.set(teamName, stats);
  return stats;
}

function ensureHeadToHead(
  stats: TeamStats,
  opponentName: string
): HeadToHeadStats {
  const opponent = normalizeTeamName(opponentName);
  const existing = stats.headToHead.get(opponent);

  if (existing) return existing;

  const created: HeadToHeadStats = {
    matches: 0,
    wins: 0,
    draws: 0,
    losses: 0,
    homeMatches: 0,
    homeWins: 0,
    homeDraws: 0,
    homeLosses: 0,
    weightedScore: 0,
    lastResults: [],
  };

  stats.headToHead.set(opponent, created);
  return created;
}

function pushHeadToHeadResult(
  record: HeadToHeadStats,
  result: "W" | "D" | "L",
  isHomeSide: boolean
) {
  record.matches += 1;

  if (result === "W") record.wins += 1;
  if (result === "D") record.draws += 1;
  if (result === "L") record.losses += 1;

  if (isHomeSide) {
    record.homeMatches += 1;
    if (result === "W") record.homeWins += 1;
    if (result === "D") record.homeDraws += 1;
    if (result === "L") record.homeLosses += 1;
  }

  const resultScore = result === "W" ? 1 : result === "L" ? -1 : 0;
  record.weightedScore =
    Math.round((record.weightedScore * 0.72 + resultScore) * 1000) / 1000;

  record.lastResults.push(result);
  if (record.lastResults.length > 10) record.lastResults.shift();
}

function applyHeadToHeadResult(
  home: TeamStats,
  away: TeamStats,
  totoResult: string
) {
  const homeRecord = ensureHeadToHead(home, away.teamName);
  const awayRecord = ensureHeadToHead(away, home.teamName);

  if (totoResult === "1") {
    pushHeadToHeadResult(homeRecord, "W", true);
    pushHeadToHeadResult(awayRecord, "L", false);
    return;
  }

  if (totoResult === "2") {
    pushHeadToHeadResult(homeRecord, "L", true);
    pushHeadToHeadResult(awayRecord, "W", false);
    return;
  }

  pushHeadToHeadResult(homeRecord, "D", true);
  pushHeadToHeadResult(awayRecord, "D", false);
}

function pushRecentResult(
  stats: TeamStats,
  result: "W" | "D" | "L",
  side?: "home" | "away"
) {
  stats.recentResults.push(result);
  if (stats.recentResults.length > 5) stats.recentResults.shift();

  const sideResults =
    side === "home"
      ? stats.homeRecentResults
      : side === "away"
      ? stats.awayRecentResults
      : null;

  if (sideResults) {
    sideResults.push(result);
    if (sideResults.length > 5) sideResults.shift();
  }
}

function formPoints(results: Array<"W" | "D" | "L">) {
  return results.reduce((sum, result) => {
    if (result === "W") return sum + 3;
    if (result === "D") return sum + 1;
    return sum;
  }, 0);
}

function applyGoalProxy(home: TeamStats, away: TeamStats, totoResult: string) {
  if (totoResult === "1") {
    home.goalsFor += 1;
    home.homeGoalsFor += 1;

    away.goalsAgainst += 1;
    away.awayGoalsAgainst += 1;
    return;
  }

  if (totoResult === "2") {
    away.goalsFor += 1;
    away.awayGoalsFor += 1;

    home.goalsAgainst += 1;
    home.homeGoalsAgainst += 1;
    return;
  }

  home.goalsFor += 1;
  home.goalsAgainst += 1;
  home.homeGoalsFor += 1;
  home.homeGoalsAgainst += 1;

  away.goalsFor += 1;
  away.goalsAgainst += 1;
  away.awayGoalsFor += 1;
  away.awayGoalsAgainst += 1;
}

function refreshGoalDifference(stats: TeamStats) {
  stats.goalDifference = stats.goalsFor - stats.goalsAgainst;
}

function applyFixtureResult(
  statsMap: Map<string, TeamStats>,
  homeTeam: string,
  awayTeam: string,
  totoResult: string
) {
  const home = ensureStats(statsMap, homeTeam);
  const away = ensureStats(statsMap, awayTeam);

  applyEloResult(home, away, totoResult);
  applyHeadToHeadResult(home, away, totoResult);
  applyGoalProxy(home, away, totoResult);

  home.matches += 1;
  away.matches += 1;

  home.homeMatches += 1;
  away.awayMatches += 1;

  if (totoResult === "1") {
    home.wins += 1;
    home.homeWins += 1;
    home.points += 3;

    away.losses += 1;
    away.awayLosses += 1;

    pushRecentResult(home, "W", "home");
    pushRecentResult(away, "L", "away");
  } else if (totoResult === "2") {
    away.wins += 1;
    away.awayWins += 1;
    away.points += 3;

    home.losses += 1;
    home.homeLosses += 1;

    pushRecentResult(home, "L", "home");
    pushRecentResult(away, "W", "away");
  } else {
    home.draws += 1;
    home.homeDraws += 1;
    home.points += 1;

    away.draws += 1;
    away.awayDraws += 1;
    away.points += 1;

    pushRecentResult(home, "D", "home");
    pushRecentResult(away, "D", "away");
  }

  refreshGoalDifference(home);
  refreshGoalDifference(away);
}

function toHeadToHeadObject(stats: TeamStats): Record<string, HeadToHeadStats> {
  return Object.fromEntries(Array.from(stats.headToHead.entries()));
}

function toStandings(statsMap: Map<string, TeamStats>): TeamStanding[] {
  const sorted = Array.from(statsMap.values()).sort((a, b) => {
    if (b.points !== a.points) return b.points - a.points;
    if (b.goalDifference !== a.goalDifference) {
      return b.goalDifference - a.goalDifference;
    }
    if (b.goalsFor !== a.goalsFor) return b.goalsFor - a.goalsFor;
    if (b.wins !== a.wins) return b.wins - a.wins;
    return a.teamName.localeCompare(b.teamName, "ja");
  });

  return sorted.map((stats, index) => {
    const homeWinRate = safeRate(stats.homeWins, stats.homeMatches);
    const awayWinRate = safeRate(stats.awayWins, stats.awayMatches);

    const homeGoalsPerMatch = safeRate(
      stats.homeGoalsFor,
      stats.homeMatches
    );
    const awayGoalsPerMatch = safeRate(
      stats.awayGoalsFor,
      stats.awayMatches
    );

    const homeGoalsAgainstPerMatch = safeRate(
      stats.homeGoalsAgainst,
      stats.homeMatches
    );
    const awayGoalsAgainstPerMatch = safeRate(
      stats.awayGoalsAgainst,
      stats.awayMatches
    );

    const totalGoalsPerMatch = safeRate(stats.goalsFor, stats.matches);
    const totalGoalsAgainstPerMatch = safeRate(
      stats.goalsAgainst,
      stats.matches
    );

    const attackRating = safeRate(totalGoalsPerMatch, LEAGUE_AVG_GOALS);
    const defenseRating =
      totalGoalsAgainstPerMatch === 0
        ? 2
        : LEAGUE_AVG_GOALS / totalGoalsAgainstPerMatch;

    return {
      rank: index + 1,
      teamName: stats.teamName,
      points: stats.points,
      goalDifference: stats.goalDifference,
      matches: stats.matches,
      wins: stats.wins,
      draws: stats.draws,
      losses: stats.losses,
      goalsFor: stats.goalsFor,
      goalsAgainst: stats.goalsAgainst,

      homeMatches: stats.homeMatches,
      homeWins: stats.homeWins,
      homeDraws: stats.homeDraws,
      homeLosses: stats.homeLosses,

      awayMatches: stats.awayMatches,
      awayWins: stats.awayWins,
      awayDraws: stats.awayDraws,
      awayLosses: stats.awayLosses,

      homeGoalsFor: stats.homeGoalsFor,
      homeGoalsAgainst: stats.homeGoalsAgainst,
      awayGoalsFor: stats.awayGoalsFor,
      awayGoalsAgainst: stats.awayGoalsAgainst,

      homeWinRate,
      awayWinRate,

      homeGoalsPerMatch,
      awayGoalsPerMatch,
      homeGoalsAgainstPerMatch,
      awayGoalsAgainstPerMatch,

      attackRating,
      defenseRating,

      eloRating: Math.round(stats.eloRating),
      homeEloRating: Math.round(stats.homeEloRating),
      awayEloRating: Math.round(stats.awayEloRating),

      recentFormPoints: formPoints(stats.recentResults),
      recentFormMatches: stats.recentResults.length,
      recentFormLabel: stats.recentResults.join(""),

      homeRecentFormPoints: formPoints(stats.homeRecentResults),
      homeRecentFormMatches: stats.homeRecentResults.length,
      homeRecentFormLabel: stats.homeRecentResults.join(""),

      awayRecentFormPoints: formPoints(stats.awayRecentResults),
      awayRecentFormMatches: stats.awayRecentResults.length,
      awayRecentFormLabel: stats.awayRecentResults.join(""),

      headToHead: toHeadToHeadObject(stats),
    };
  });
}

async function buildDateCutStandingsInternal(roundNo: number) {
  const targetDate = await getRoundDateFromMaster(roundNo);

  if (!targetDate) {
    return {
      targetDate: null,
      standings: [] as TeamStanding[],
      sourceRounds: [] as number[],
    };
  }

  const priorRounds = await listPriorRoundDates(targetDate, roundNo);

  const loaded = await loadManyRoundFixtures(
    priorRounds.map((row) => row.round_no),
    { maxMissingFetches: 25, concurrency: 5 }
  );

  const statsMap = new Map<string, TeamStats>();

  for (const round of loaded) {
    for (const fixture of round.fixtures) {
      if (!/^[012]$/.test(fixture.totoResult ?? "")) continue;
      if (!isJLeagueFixture(fixture)) continue;

      applyFixtureResult(
        statsMap,
        fixture.homeTeam,
        fixture.awayTeam,
        fixture.totoResult ?? ""
      );
    }
  }

  return {
    targetDate,
    standings: toStandings(statsMap),
    sourceRounds: loaded
      .filter((round) => round.fixtures.length > 0)
      .map((round) => round.roundNo),
  };
}

export async function buildDateCutStandings(roundNo: number) {
  const cached = standingsCache.get(roundNo);
  if (cached) return cached;

  const result = await buildDateCutStandingsInternal(roundNo);
  standingsCache.set(roundNo, result);

  return result;
}