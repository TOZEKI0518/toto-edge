export type TotoRoundSummary = {
  round: string;
  salesStart: string;
  salesEnd: string;
  status: "open" | "closed" | "unknown";
};