export type TotoFixture = {
  matchNo: number;
  homeTeam: string;
  awayTeam: string;
  kickoffAt?: string;
  venue?: string;
};

export type PredictionInput = {
  fixture: TotoFixture;
  homeStanding?: {
    rank: number;
    teamName: string;
    points: number;
    goalDifference: number;
  };
  awayStanding?: {
    rank: number;
    teamName: string;
    points: number;
    goalDifference: number;
  };
};