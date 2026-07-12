import type { TeamStanding } from "@/types/standing";

export type TotoFixture = {
  matchNo: number;
  homeTeam: string;
  awayTeam: string;
  kickoffAt?: string;
  venue?: string;
  totoResult?: string;
  league?: string;
};

export type PredictionInput = {
  fixture: TotoFixture;
  homeStanding?: TeamStanding;
  awayStanding?: TeamStanding;
};