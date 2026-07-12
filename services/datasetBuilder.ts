// services/datasetBuilder.ts

import type { FeatureVector } from "@/types/feature";

const CSV_COLUMNS: (keyof FeatureVector)[] = [
  "roundNo",
  "snapshotDate",
  "league",
  "homeTeam",
  "awayTeam",

  "homeElo",
  "awayElo",
  "eloDiff",

  "homeHomeElo",
  "awayAwayElo",
  "homeAwayDiff",

  "homeFormPoints",
  "awayFormPoints",
  "formDiff",

  "homeLast5Wins",
  "awayLast5Wins",

  "homeRank",
  "awayRank",
  "rankDiff",

  "homePoints",
  "awayPoints",
  "pointsDiff",

  "homeGoalsFor",
  "awayGoalsFor",

  "homeGoalsAgainst",
  "awayGoalsAgainst",

  "homeGoalDifference",
  "awayGoalDifference",
  "goalDifferenceDiff",

  "homeWinRate",
  "awayWinRate",

  "homeHomeWinRate",
  "awayAwayWinRate",
  "homeAwayWinRateDiff",

  "homeHomeGoalsPerMatch",
  "awayAwayGoalsPerMatch",
  "homeAwayGoalsPerMatchDiff",

  "homeHomeGoalsAgainstPerMatch",
  "awayAwayGoalsAgainstPerMatch",
  "homeAwayGoalsAgainstPerMatchDiff",

  "homeAttackRating",
  "awayAttackRating",
  "attackRatingDiff",

  "homeDefenseRating",
  "awayDefenseRating",
  "defenseRatingDiff",

  "absEloDiff",
  "absRankDiff",
  "absPointsDiff",

  "absAttackDiff",
  "absDefenseDiff",

  "balanceScore",

  "homeLast3Points",
  "awayLast3Points",
  "last3PointsDiff",

  "homeLast3Wins",
  "awayLast3Wins",
  "last3WinsDiff",

  "homeUnbeatenStreak",
  "awayUnbeatenStreak",
  "unbeatenStreakDiff",

  "homeLosingStreak",
  "awayLosingStreak",
  "losingStreakDiff",

  "h2hHomeWins",
  "h2hDraws",
  "h2hAwayWins",

  "month",
  "seasonRound",

  "combinedGoalsPerMatch",
  "expectedGoalGap",
  "attackDefenseRatio",

  "result",
];

const escapeCsvValue = (value: unknown) => {
  if (value === undefined || value === null) return "";

  const text = String(value);

  if (
    text.includes(",") ||
    text.includes("\n") ||
    text.includes("\r") ||
    text.includes('"')
  ) {
    return `"${text.replace(/"/g, '""')}"`;
  }

  return text;
};

export function featureVectorsToCsv(features: FeatureVector[]): string {
  const header = CSV_COLUMNS.join(",");

  const rows = features.map((feature) =>
    CSV_COLUMNS.map((column) => escapeCsvValue(feature[column])).join(",")
  );

  return [header, ...rows].join("\n");
}

export function buildDatasetSummary(features: FeatureVector[]) {
  const total = features.length;

  const withResult = features.filter((feature) => feature.result).length;

  const home = features.filter((feature) => feature.result === "H").length;
  const draw = features.filter((feature) => feature.result === "D").length;
  const away = features.filter((feature) => feature.result === "A").length;

  const jLeague = features.filter((feature) =>
    feature.league.toLowerCase().includes("j")
  ).length;

  return {
    total,
    withResult,
    withoutResult: total - withResult,
    resultBreakdown: {
      home,
      draw,
      away,
    },
    jLeague,
  };
}