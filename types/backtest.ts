import type { FixturePrediction, MatchOutcome } from "@/types/prediction";

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
};

export type BacktestRoundResult = {
  round: string;
  totalMatches: number;
  hitCount: number;
  hitRate: number;
  matches: BacktestMatchResult[];
};