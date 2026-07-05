export type TeamStanding = {
  rank: number;
  teamName: string;
  points: number;
  goalDifference: number;

  matches?: number;
  wins?: number;
  draws?: number;
  losses?: number;
  goalsFor?: number;
  goalsAgainst?: number;
};