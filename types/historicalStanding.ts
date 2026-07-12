export type HistoricalStandingRequest = {
  year: number;
  competitionId: number;
  sectionId: number;
  league: "J1" | "J2" | "J3";
};

export type HistoricalStandingConfig = {
  roundNo: string;
  snapshotLabel: string;
  requests: HistoricalStandingRequest[];
};