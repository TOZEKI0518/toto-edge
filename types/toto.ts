export type Outcome = "home" | "draw" | "away";
export type TeamStats = {
  name: string;
  rank: number;
  points: number;
  last5: string;
  goalsFor: number;
  goalsAgainst: number;
  homeWinRate?: number;
  awayWinRate?: number;
  restDays: number;
  injuryLevel: number;
};
export type TotoMatch = {
  id: number;
  round: string;
  kickoff: string;
  league: string;
  venue: string;
  weather: "晴" | "曇" | "雨" | "強雨";
  home: TeamStats;
  away: TeamStats;
  voteRate: Record<Outcome, number>;
};
export type Reason = { label: string; value: number; note: string };
export type Prediction = {
  matchId: number;
  probabilities: Record<Outcome, number>;
  pick: Outcome;
  confidence: "S" | "A" | "B" | "C";
  edge: number;
  reasons: Reason[];
};
