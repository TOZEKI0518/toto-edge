export type MatchResult = "HOME" | "DRAW" | "AWAY" | "UNKNOWN";

export type PredictionPick = "HOME" | "DRAW" | "AWAY";

export type ConfidenceRank = "S" | "A" | "B+" | "B" | "C";

export type Team = {
  id: string;
  name: string;
  shortName: string;
  league: "J1" | "J2" | "J3";
};

export type TeamStats = {
  teamId: string;
  rank: number;
  points: number;
  wins: number;
  draws: number;
  losses: number;
  goalsFor: number;
  goalsAgainst: number;
  goalDifference: number;
  homeWinRate: number;
  awayWinRate: number;
  recentFormPoints: number;
};

export type TotoMatch = {
  id: string;
  totoRound: string;
  matchNo: number;
  kickoffAt: string;
  venue: string;
  homeTeam: Team;
  awayTeam: Team;
  homeStats: TeamStats;
  awayStats: TeamStats;
  weather?: {
    condition: string;
    temperature: number;
    windSpeed: number;
  };
  voteRates?: {
    home: number;
    draw: number;
    away: number;
  };
  result?: {
    homeScore: number;
    awayScore: number;
    outcome: MatchResult;
  };
};

export type PredictionReason = {
  label: string;
  impact: number;
  description: string;
};

export type MatchPrediction = {
  matchId: string;
  pick: PredictionPick;
  confidence: ConfidenceRank;
  probabilities: {
    home: number;
    draw: number;
    away: number;
  };
  expectedValues?: {
    home: number;
    draw: number;
    away: number;
  };
  reasons: PredictionReason[];
};