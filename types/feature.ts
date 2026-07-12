// types/feature.ts

export type MatchResult = "H" | "D" | "A";

export interface FeatureVector {
  // ===== Meta =====
  roundNo: number;
  snapshotDate: string;

  league: string;

  homeTeam: string;
  awayTeam: string;

  // ===== Overall Elo =====
  homeElo: number;
  awayElo: number;
  eloDiff: number;

  // ===== Home / Away Elo =====
  homeHomeElo: number;
  awayAwayElo: number;
  homeAwayDiff: number;

  // ===== Form =====
  homeFormPoints: number;
  awayFormPoints: number;
  formDiff: number;

  homeLast5Wins: number;
  awayLast5Wins: number;

  // ===== Standing =====
  homeRank: number;
  awayRank: number;
  rankDiff: number;

  homePoints: number;
  awayPoints: number;
  pointsDiff: number;

  // ===== Goals =====
  homeGoalsFor: number;
  awayGoalsFor: number;

  homeGoalsAgainst: number;
  awayGoalsAgainst: number;

  homeGoalDifference: number;
  awayGoalDifference: number;

  goalDifferenceDiff: number;

  // ===== Home / Away Win Rate =====
  homeWinRate: number;
  awayWinRate: number;

  // ===== H2H =====
  h2hHomeWins: number;
  h2hDraws: number;
  h2hAwayWins: number;

  // ===== Calendar =====
  month: number;
  seasonRound: number;

  // ===== Target =====
  result?: MatchResult;

  homeHomeWinRate: number;
  awayAwayWinRate: number;
  homeAwayWinRateDiff: number;

  homeHomeGoalsPerMatch: number;
  awayAwayGoalsPerMatch: number;
  homeAwayGoalsPerMatchDiff: number;

  homeHomeGoalsAgainstPerMatch: number;
  awayAwayGoalsAgainstPerMatch: number;
  homeAwayGoalsAgainstPerMatchDiff: number;

  homeAttackRating: number;
  awayAttackRating: number;
  attackRatingDiff: number;

  homeDefenseRating: number;
  awayDefenseRating: number;
  defenseRatingDiff: number;

  absEloDiff: number;
  absRankDiff: number;
  absPointsDiff: number;

  absAttackDiff: number;
  absDefenseDiff: number;

  balanceScore: number;

  homeLast3Points: number;
  awayLast3Points: number;
  last3PointsDiff: number;

  homeLast3Wins: number;
  awayLast3Wins: number;
  last3WinsDiff: number;

  homeUnbeatenStreak: number;
  awayUnbeatenStreak: number;
  unbeatenStreakDiff: number;

  homeLosingStreak: number;
  awayLosingStreak: number;
  losingStreakDiff: number;

  combinedGoalsPerMatch: number;
  expectedGoalGap: number;
  attackDefenseRatio: number;
}