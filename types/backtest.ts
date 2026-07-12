import type { FixturePrediction, MatchOutcome, PredictionFactor } from "@/types/prediction";

export type BacktestDataSource = "date_cut" | "round_snapshot" | "snapshot" | "current" | "none";

export type BacktestMatchResult = {
  matchNo: number;
  homeTeam: string;
  awayTeam: string;
  predictedOutcome: MatchOutcome;
  actualOutcome: MatchOutcome;
  probability: number;
  confidence: FixturePrediction["confidence"];
  totalScore: number;
  isHit: boolean;
  league?: string;
  factors?: PredictionFactor[];
};

export type BacktestRoundResult = {
  round: string;
  totalFixtures: number;
  totalMatches: number;
  hitCount: number;
  hitRate: number;
  coverageRate: number;
  matches: BacktestMatchResult[];
  unmatchedTeams?: string[];
  snapshotDate?: string | null;
  snapshotRoundNo?: number | null;
  dataSource?: BacktestDataSource;
  sourceRoundCount?: number;
};

export type BacktestInspectResult = {
  roundNumber: string;
  round: string;
  totalFixtures: number;
  totalMatches: number;
  hitCount: number;
  hitRate: number;
  coverageRate: number;
  isJleagueCandidate: boolean;
  snapshotDate?: string | null;
  snapshotRoundNo?: number | null;
  dataSource?: BacktestDataSource;
  sourceRoundCount?: number;
};
