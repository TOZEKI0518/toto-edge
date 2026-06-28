export type MatchOutcome = "HOME" | "DRAW" | "AWAY";

export type PredictionFactor = {
  label: string;
  score: number;
  description: string;
};

export type FixturePrediction = {
  matchNo: number;
  homeTeam: string;
  awayTeam: string;
  outcome: MatchOutcome;
  probability: number;
  confidence: "S" | "A" | "B" | "C";
  totalScore: number;
  factors: PredictionFactor[];
};