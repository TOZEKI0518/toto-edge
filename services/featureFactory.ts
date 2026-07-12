// services/featureFactory.ts

import type { PredictionInput } from "@/types/fixture";
import type { FeatureVector, MatchResult } from "@/types/feature";

const safeNumber = (value: number | undefined | null, fallback = 0) =>
  typeof value === "number" && Number.isFinite(value) ? value : fallback;

const safeRate = (wins?: number, matches?: number) => {
  if (!matches || matches <= 0) return 0;
  return safeNumber(wins) / matches;
};

const countWinsFromLabel = (label?: string) =>
  (label ?? "").split("").filter((item) => item === "W").length;

const lastN = (label: string | undefined, n: number) =>
  (label ?? "").slice(-n);

const pointsFromLabel = (label: string | undefined) =>
  (label ?? "").split("").reduce((sum, item) => {
    if (item === "W") return sum + 3;
    if (item === "D") return sum + 1;
    return sum;
  }, 0);

const unbeatenStreakFromLabel = (label: string | undefined) => {
  const results = (label ?? "").split("").reverse();
  let streak = 0;

  for (const result of results) {
    if (result === "W" || result === "D") streak += 1;
    else break;
  }

  return streak;
};

const losingStreakFromLabel = (label: string | undefined) => {
  const results = (label ?? "").split("").reverse();
  let streak = 0;

  for (const result of results) {
    if (result === "L") streak += 1;
    else break;
  }

  return streak;
};

export function buildFeatureVector(params: {
  roundNo: number;
  snapshotDate: string;
  input: PredictionInput;
  result?: MatchResult;
}): FeatureVector {
  const { roundNo, snapshotDate, input, result } = params;

  const home = input.homeStanding;
  const away = input.awayStanding;

  const homeElo = safeNumber(home?.eloRating, 1500);
  const awayElo = safeNumber(away?.eloRating, 1500);

  const homeHomeElo = safeNumber(home?.homeEloRating, homeElo);
  const awayAwayElo = safeNumber(away?.awayEloRating, awayElo);

  const homeFormPoints = safeNumber(home?.recentFormPoints);
  const awayFormPoints = safeNumber(away?.recentFormPoints);

  const homeRank = safeNumber(home?.rank);
  const awayRank = safeNumber(away?.rank);

  const homePoints = safeNumber(home?.points);
  const awayPoints = safeNumber(away?.points);

  const homeGoalsFor = safeNumber(home?.goalsFor);
  const awayGoalsFor = safeNumber(away?.goalsFor);

  const homeGoalsAgainst = safeNumber(home?.goalsAgainst);
  const awayGoalsAgainst = safeNumber(away?.goalsAgainst);

  const homeGoalDifference = safeNumber(home?.goalDifference);
  const awayGoalDifference = safeNumber(away?.goalDifference);

  const homeHomeWinRate = safeNumber(home?.homeWinRate);
  const awayAwayWinRate = safeNumber(away?.awayWinRate);

  const homeHomeGoalsPerMatch = safeNumber(home?.homeGoalsPerMatch);
  const awayAwayGoalsPerMatch = safeNumber(away?.awayGoalsPerMatch);

  const homeHomeGoalsAgainstPerMatch = safeNumber(
    home?.homeGoalsAgainstPerMatch
  );
  const awayAwayGoalsAgainstPerMatch = safeNumber(
    away?.awayGoalsAgainstPerMatch
  );

  const homeAttackRating = safeNumber(home?.attackRating);
  const awayAttackRating = safeNumber(away?.attackRating);

  const homeDefenseRating = safeNumber(home?.defenseRating);
  const awayDefenseRating = safeNumber(away?.defenseRating);

  const absEloDiff = Math.abs(homeElo - awayElo);
  const absRankDiff = Math.abs(homeRank - awayRank);
  const absPointsDiff = Math.abs(homePoints - awayPoints);

  const absAttackDiff = Math.abs(
    homeAttackRating - awayAttackRating
  );

  const absDefenseDiff = Math.abs(
    homeDefenseRating - awayDefenseRating
  );

  const balanceScore =
    absEloDiff * 0.35 +
    absRankDiff * 0.25 +
    absAttackDiff * 0.20 +
    absDefenseDiff * 0.20;
  
  const h2h = home?.headToHead?.[away?.teamName ?? input.fixture.awayTeam];

  const homeLast3Label = lastN(home?.recentFormLabel, 3);
  const awayLast3Label = lastN(away?.recentFormLabel, 3);

  const homeLast3Points = pointsFromLabel(homeLast3Label);
  const awayLast3Points = pointsFromLabel(awayLast3Label);

  const homeLast3Wins = countWinsFromLabel(homeLast3Label);
  const awayLast3Wins = countWinsFromLabel(awayLast3Label);

  const homeUnbeatenStreak = unbeatenStreakFromLabel(home?.recentFormLabel);
  const awayUnbeatenStreak = unbeatenStreakFromLabel(away?.recentFormLabel);

  const homeLosingStreak = losingStreakFromLabel(home?.recentFormLabel);
  const awayLosingStreak = losingStreakFromLabel(away?.recentFormLabel);

  const month = Number(snapshotDate.slice(5, 7)) || 0;

  const combinedGoalsPerMatch =
    homeHomeGoalsPerMatch +
    awayAwayGoalsPerMatch;

  const expectedGoalGap = Math.abs(
      homeHomeGoalsPerMatch -
      awayAwayGoalsPerMatch
  );

  const attackDefenseRatio =
      (homeAttackRating + awayAttackRating) /
      (homeDefenseRating + awayDefenseRating + 0.0001);

  return {
    roundNo,
    snapshotDate,

    league: input.fixture.league ?? home?.league ?? away?.league ?? "Unknown",

    homeTeam: input.fixture.homeTeam,
    awayTeam: input.fixture.awayTeam,

    homeElo,
    awayElo,
    eloDiff: homeElo - awayElo,

    homeHomeElo,
    awayAwayElo,
    homeAwayDiff: homeHomeElo - awayAwayElo,

    homeFormPoints,
    awayFormPoints,
    formDiff: homeFormPoints - awayFormPoints,

    homeLast5Wins: countWinsFromLabel(home?.recentFormLabel),
    awayLast5Wins: countWinsFromLabel(away?.recentFormLabel),

    homeRank,
    awayRank,
    rankDiff: awayRank - homeRank,

    homePoints,
    awayPoints,
    pointsDiff: homePoints - awayPoints,

    homeGoalsFor,
    awayGoalsFor,

    homeGoalsAgainst,
    awayGoalsAgainst,

    homeGoalDifference,
    awayGoalDifference,

    goalDifferenceDiff: homeGoalDifference - awayGoalDifference,

    homeWinRate: safeRate(home?.wins, home?.matches),
    awayWinRate: safeRate(away?.wins, away?.matches),

    h2hHomeWins: safeNumber(h2h?.wins),
    h2hDraws: safeNumber(h2h?.draws),
    h2hAwayWins: safeNumber(h2h?.losses),

    month,
    seasonRound: roundNo,

    homeHomeWinRate,
    awayAwayWinRate,
    homeAwayWinRateDiff: homeHomeWinRate - awayAwayWinRate,

    homeHomeGoalsPerMatch,
    awayAwayGoalsPerMatch,
    homeAwayGoalsPerMatchDiff:
      homeHomeGoalsPerMatch - awayAwayGoalsPerMatch,

    homeHomeGoalsAgainstPerMatch,
    awayAwayGoalsAgainstPerMatch,
    homeAwayGoalsAgainstPerMatchDiff:
      homeHomeGoalsAgainstPerMatch - awayAwayGoalsAgainstPerMatch,

    homeAttackRating,
    awayAttackRating,
    attackRatingDiff: homeAttackRating - awayAttackRating,

    homeDefenseRating,
    awayDefenseRating,
    defenseRatingDiff: homeDefenseRating - awayDefenseRating,

    absEloDiff,
    absRankDiff,
    absPointsDiff,

    absAttackDiff,
    absDefenseDiff,

    balanceScore,

    homeLast3Points,
    awayLast3Points,
    last3PointsDiff: homeLast3Points - awayLast3Points,

    homeLast3Wins,
    awayLast3Wins,
    last3WinsDiff: homeLast3Wins - awayLast3Wins,

    homeUnbeatenStreak,
    awayUnbeatenStreak,
    unbeatenStreakDiff: homeUnbeatenStreak - awayUnbeatenStreak,

    homeLosingStreak,
    awayLosingStreak,
    losingStreakDiff: homeLosingStreak - awayLosingStreak,

    combinedGoalsPerMatch,
    expectedGoalGap,
    attackDefenseRatio,
    
    result,
  };
}

export function buildFeatureVectors(params: {
  roundNo: number;
  snapshotDate: string;
  inputs: PredictionInput[];
  resultByMatchNo?: Map<number, MatchResult>;
}): FeatureVector[] {
  return params.inputs.map((input) =>
    buildFeatureVector({
      roundNo: params.roundNo,
      snapshotDate: params.snapshotDate,
      input,
      result: params.resultByMatchNo?.get(input.fixture.matchNo),
    })
  );
}